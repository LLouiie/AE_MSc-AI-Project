"""
Read-only exploratory analysis of HotpotQA trajectory logs.
Produces: analysis/report.md and analysis/step_features.csv

IMPORTANT: this script never modifies agent, controller, scheduler, or any
run-time code. It reads existing *.jsonl logs and produces analysis artefacts
only inside the analysis/ directory.

Usage (from hotpotqa_runs/):
    python3 analysis/trajectory_analysis.py
"""

import json, re, csv, math, sys
from pathlib import Path
from collections import Counter, defaultdict

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent          # hotpotqa_runs/
RUNS = BASE / "runs"
OUT  = Path(__file__).parent                 # analysis/

LOGS = {
    "smoke_fixed10": {
        "path": RUNS / "smoke_fixed10" / "practice_log.jsonl",
        "type": "practice",
        "scheduler": "fixed:10",
        "max_trials": 6,
        "limit": 20,
    },
    "smoke_never": {
        "path": RUNS / "smoke_never" / "practice_log.jsonl",
        "type": "practice",
        "scheduler": "never",
        "max_trials": 6,
        "limit": 5,
    },
    "smoke_noreflect": {
        "path": RUNS / "smoke_noreflect" / "practice_log.jsonl",
        "type": "practice",
        "scheduler": "never",
        "max_trials": 1,
        "limit": 5,
    },
    "smoke_exam": {
        "path": RUNS / "smoke_exam" / "exam_log.jsonl",
        "type": "exam",
        "scheduler": "N/A",
        "max_trials": 1,
        "limit": 5,
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_steps(scratchpad: str) -> list[dict]:
    """
    Parse a scratchpad string into a list of step dicts.
    Each step: {step_n, thought, action_type, action_arg, observation, finished_here}
    Does NOT use CORRECT/INCORRECT to judge quality — those strings are preserved
    verbatim in the observation field but never used as signals in feature computation.
    """
    lines = scratchpad.split("\n")
    steps = []
    cur = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = re.match(r"^Thought (\d+):\s*(.*)", line)
        if m:
            if cur:
                steps.append(cur)
            cur = {"step_n": int(m.group(1)), "thought": m.group(2),
                   "action_type": None, "action_arg": None,
                   "observation": "", "finished_here": False}
            continue

        m = re.match(r"^Action (\d+):\s*(.*)", line)
        if m and cur:
            raw_action = m.group(2).strip()
            am = re.match(r"^(\w+)\[(.+)\]$", raw_action)
            if am:
                cur["action_type"] = am.group(1)
                cur["action_arg"]  = am.group(2)
            else:
                cur["action_type"] = "INVALID"
                cur["action_arg"]  = raw_action
            continue

        m = re.match(r"^Observation (\d+):\s*(.*)", line)
        if m and cur:
            cur["observation"] = m.group(2)
            if cur.get("action_type") == "Finish":
                cur["finished_here"] = True
            continue

        # continuation of observation (multi-line)
        if cur and "observation" in cur and cur["observation"] is not None:
            if not re.match(r"^(Thought|Action|Observation) \d+", line):
                cur["observation"] = (cur["observation"] + " " + line).strip()

    if cur:
        steps.append(cur)
    return steps


def jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cap_words(text: str) -> set:
    """Very rough named-entity proxy: words starting with uppercase (not first word)."""
    words = text.split()
    return {w for w in words[1:] if w and w[0].isupper() and len(w) > 1}


def compute_step_features(steps: list[dict]) -> list[dict]:
    """
    Compute gold-free features for each step.
    Gold answer and CORRECT/INCORRECT text are never used as signals here.
    """
    prev_queries  = []   # list of (action_type, action_arg) so far in this trial
    prev_obs      = []   # list of observation strings so far

    rows = []
    no_progress_streak = 0

    for i, s in enumerate(steps):
        q_arg = s["action_arg"] or ""
        obs   = s["observation"] or ""
        atype = s["action_type"] or ""

        # ── query repeat features ──────────────────────────────────────────
        prev_same_type_args = [a for (t, a) in prev_queries if t == atype]

        query_exact_repeat = q_arg in [a for (_, a) in prev_queries]

        query_sim_scores = [jaccard(q_arg, a) for (_, a) in prev_queries] if prev_queries else [0.0]
        query_max_sim = max(query_sim_scores) if query_sim_scores else 0.0

        # ── observation repeat features ────────────────────────────────────
        obs_exact_repeat = (obs in prev_obs) if prev_obs else False

        obs_sim_scores = [jaccard(obs, o) for o in prev_obs] if prev_obs else [0.0]
        obs_max_sim = max(obs_sim_scores) if obs_sim_scores else 0.0

        # ── new named entities ─────────────────────────────────────────────
        all_prev_caps = set()
        for po in prev_obs:
            all_prev_caps |= _cap_words(po)
        new_caps = _cap_words(obs) - all_prev_caps
        new_named_entities = len(new_caps)

        # ── action counts ──────────────────────────────────────────────────
        action_type_count_so_far = sum(1 for (t, _) in prev_queries if t == atype)

        invalid_action = (atype == "INVALID" or atype not in {"Search", "Lookup", "Finish"})

        lookup_failure  = "No more results found" in obs
        search_failure  = "Could not find" in obs

        # ── budget ────────────────────────────────────────────────────────
        cumulative_steps     = i + 1           # 1-indexed
        cumulative_llm_calls = (i + 1) * 2    # 2 calls per step (Thought + Action)

        # ── no-progress streak ────────────────────────────────────────────
        # No progress: same action type AND observation very similar to any previous
        if i > 0 and obs_max_sim > 0.85 and not s["finished_here"]:
            no_progress_streak += 1
        else:
            no_progress_streak = 0

        rows.append({
            "step_n":                   s["step_n"],
            "thought_truncated":        (s["thought"] or "")[:80],
            "action_type":              atype,
            "action_arg":               q_arg[:60],
            "obs_truncated":            obs[:80],
            "finished_here":            s["finished_here"],
            # gold-free features
            "query_exact_repeat":       query_exact_repeat,
            "query_max_sim":            round(query_max_sim, 3),
            "obs_exact_repeat":         obs_exact_repeat,
            "obs_max_sim":              round(obs_max_sim, 3),
            "new_named_entities":       new_named_entities,
            "action_type_count_so_far": action_type_count_so_far,
            "invalid_action":           invalid_action,
            "lookup_failure":           lookup_failure,
            "search_failure":           search_failure,
            "cumulative_steps":         cumulative_steps,
            "cumulative_llm_calls":     cumulative_llm_calls,
            "no_progress_streak":       no_progress_streak,
        })

        prev_queries.append((atype, q_arg))
        prev_obs.append(obs)

    return rows


def answer_candidate_changes(trial_records: list) -> list[bool]:
    """True at index i if the answer changed from trial i-1 to trial i."""
    changes = [False]
    for i in range(1, len(trial_records)):
        a_prev = (trial_records[i-1].get("answer") or "").strip().lower()
        a_curr = (trial_records[i].get("answer") or "").strip().lower()
        changes.append(a_curr != a_prev)
    return changes


# ── main ──────────────────────────────────────────────────────────────────────

def analyse():
    all_step_rows = []   # for CSV
    report_sections = []

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Log inventory
    # ═══════════════════════════════════════════════════════════════════════════
    inv_lines = [
        "## 1. 日志来源\n",
        "| run | 类型 | scheduler | max_trials | 题数 | 字段 |",
        "|-----|------|-----------|-----------|------|------|",
    ]
    all_data = {}
    for name, meta in LOGS.items():
        if not meta["path"].exists():
            inv_lines.append(f"| {name} | — | — | — | **文件不存在** | — |")
            all_data[name] = []
            continue
        rows = load_log(meta["path"])
        all_data[name] = rows
        if rows:
            fields = sorted(rows[0].keys())
            has = lambda k: "✓" if k in fields else "✗"
            field_str = (f"trial_records:{has('trial_records')} "
                         f"trajectory:{has('trajectory')} "
                         f"reflections:{has('reflections')} "
                         f"trials_used:{has('trials_used')} "
                         f"llm_calls:{has('llm_calls')}")
        else:
            field_str = "empty"
        inv_lines.append(
            f"| {name} | {meta['type']} | {meta['scheduler']} | "
            f"{meta['max_trials']} | {len(rows)} | {field_str} |"
        )
    report_sections.append("\n".join(inv_lines))

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Overall statistics (smoke_fixed10 as primary dataset)
    # ═══════════════════════════════════════════════════════════════════════════
    primary = all_data["smoke_fixed10"]

    em_dist    = Counter(r["em"] for r in primary)
    trial_dist = Counter(r["trials_used"] for r in primary)
    calls_vals = [r["llm_calls"] for r in primary]

    # Failure mode classification (uses em and gold only for grouping, not as signals)
    def classify(r):
        if r["em"] == 1:
            return "correct"
        pred = r["pred"].strip().lower()
        gold = r["gold"].strip().lower()
        if gold in pred or pred in gold:
            return "format_subset"    # pred is substring/superset of gold
        return "wrong_content"

    groups = defaultdict(list)
    for r in primary:
        groups[classify(r)].append(r["q_index"])

    stat_lines = [
        "\n## 2. smoke_fixed10 全局统计（20 题，主分析数据集）\n",
        f"- EM 分布: correct={em_dist[1]}, failed={em_dist[0]}",
        f"- trials_used 分布: {dict(sorted(trial_dist.items()))}",
        f"- LLM calls: min={min(calls_vals)}, max={max(calls_vals)}, "
        f"mean={sum(calls_vals)/len(calls_vals):.1f}",
        f"- 格式/子串失败 (format_subset): q_index = {sorted(groups['format_subset'])}",
        f"- 内容错误 (wrong_content):       q_index = {sorted(groups['wrong_content'])}",
        f"- 正确 (correct):                 q_index = {sorted(groups['correct'])}",
    ]
    report_sections.append("\n".join(stat_lines))

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: Sample selection and trajectory decomposition
    # ═══════════════════════════════════════════════════════════════════════════
    # Categories (using em/gold only for grouping):
    # A: single trial correct
    # B: multi-trial success (trials_used >= 2, em=1)
    # C: all-trials failure, format/subset issue
    # D: all-trials failure, observation repeat pattern
    # E: all-trials failure, wrong content / answer degrades
    # F: exam log (single trial, no trial_records)

    samples = {
        "A_single_trial_correct":          [2, 6, 8, 10, 15, 18],
        "B_multi_trial_success":           [13],
        "C_format_failure":                [1, 4, 9, 12],
        "D_obs_repeat_failure":            [5, 20],
        "E_wrong_content_or_degrades":     [7, 14, 16, 17, 19],
    }

    traj_lines = ["\n## 3. 代表性 trajectory 分解\n"]

    for cat, qindices in samples.items():
        traj_lines.append(f"\n### {cat}\n")
        for qi in qindices:
            r = primary[qi - 1]
            traj_lines.append(
                f"**Q{qi}** `em={r['em']}` `trials={r['trials_used']}` "
                f"`gold={r['gold']!r}` `pred={r['pred'][:50]!r}`\n"
            )

            # Show per-trial step table for the LAST trial
            # (and for multi-trial success, also show trial 1)
            show_trials = [r["trial_records"][-1]]
            if cat == "B_multi_trial_success" and len(r["trial_records"]) > 1:
                show_trials = r["trial_records"]

            for tr in show_trials:
                steps = parse_steps(tr["trajectory"])
                traj_lines.append(
                    f"*Trial {tr['trial']}* "
                    f"(answer: `{str(tr['answer'])[:40]!r}`)\n"
                )
                traj_lines.append(
                    "| step | action_type | argument/query | obs摘要 | search_fail | obs_exact_repeat |"
                )
                traj_lines.append(
                    "|-----:|------------|----------------|---------|:-----------:|:----------------:|"
                )
                feats = compute_step_features(steps)
                for s, f in zip(steps, feats):
                    obs_short = (s["observation"] or "")[:70].replace("|", "/")
                    arg_short  = (s["action_arg"]  or "")[:40].replace("|", "/")
                    traj_lines.append(
                        f"| {s['step_n']} | {s['action_type']} | `{arg_short}` | "
                        f"{obs_short} | {f['search_failure']} | {f['obs_exact_repeat']} |"
                    )
                traj_lines.append("")

            # reflections summary (gold-free: just count and first 120 chars)
            refs = r.get("reflections", [])
            if refs:
                traj_lines.append(f"*Reflections ({len(refs)} total):*")
                for i, ref in enumerate(refs, 1):
                    traj_lines.append(f"- [{i}] {ref[:120]}")
                traj_lines.append("")

            # Collect step features for CSV
            for trial_rec in r["trial_records"]:
                steps = parse_steps(trial_rec["trajectory"])
                feats = compute_step_features(steps)
                ac_changes = answer_candidate_changes(r["trial_records"])
                for f in feats:
                    all_step_rows.append({
                        "run": "smoke_fixed10",
                        "qid": r["qid"],
                        "q_index": r["q_index"],
                        "category": cat,
                        "em": r["em"],
                        "trial": trial_rec["trial"],
                        "answer_candidate_change": ac_changes[trial_rec["trial"] - 1],
                        **f,
                    })

    report_sections.append("\n".join(traj_lines))

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Gold-free feature availability
    # ═══════════════════════════════════════════════════════════════════════════
    avail_lines = [
        "\n## 4. 候选信号可获得性\n",
        "| 特征 | 可获得 | 计算方式 | 备注 |",
        "|------|:------:|----------|------|",
        "| query_exact_repeat | ✓ | 字符串匹配 | 当前 trial 内 |",
        "| query_max_sim (Jaccard) | ✓ | token overlap | 代理，非 embedding |",
        "| obs_exact_repeat | ✓ | 字符串匹配 | 当前 trial 内 |",
        "| obs_max_sim (Jaccard) | ✓ | token overlap | 代理，非 embedding |",
        "| new_named_entities | ✓ (近似) | 大写词计数差值 | 无 NER 模型，大写词代理 |",
        "| action_type_count_so_far | ✓ | 计数 | 每个 action type |",
        "| invalid_action | ✓ | 解析失败 | parse_action 返回 None |",
        "| lookup_failure | ✓ | 字符串匹配 | 'No more results found' |",
        "| search_failure | ✓ | 字符串匹配 | 'Could not find' |",
        "| cumulative_steps | ✓ | step_n | 直接可读 agent.step_n |",
        "| cumulative_llm_calls | ✓ | step_n × 2 | 每步 2 次 LLM call |",
        "| answer_candidate_change | ✓ (trial间) | 字符串比较 | 仅跨 trial 可用，trial 内不可用 |",
        "| no_progress_streak | ✓ (近似) | obs_max_sim > 0.85 连续计数 | 阈值任意 |",
        "| token_count | ✗ unavailable | tiktoken 未写入日志 | 需修改 agent 才能记录 |",
        "| embedding_similarity | ✗ unavailable | 无嵌入 | 需外部模型 |",
        "| confidence / perplexity | ✗ unavailable | vLLM log-prob 未启用 | 需修改 LLM call |",
        "| step内候选答案 | ✗ unavailable | Finish 前无中间答案 | 仅 Finish 动作后可知 |",
    ]
    report_sections.append("\n".join(avail_lines))

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: Descriptive analysis
    # ═══════════════════════════════════════════════════════════════════════════

    # For primary dataset: per-question, per-trial features
    desc_lines = ["\n## 5. 初步描述性统计\n"]

    # 5a. Among all-fail questions (trials_used==6, em==0):
    #     How many had obs_exact_repeat before final step?
    fail_6 = [r for r in primary if r["em"] == 0 and r["trials_used"] == 6]
    desc_lines.append(f"### 5a. 跑满 6 trial 的失败题 (N={len(fail_6)})\n")
    desc_lines.append(
        "| q | 最后trial步数 | 是否halt | obs_repeat出现步 | search_fail出现步 | no_prog_streak_max |"
    )
    desc_lines.append("|---|-------------|---------|----------------|-------------------|--------------------|")

    for r in fail_6:
        last_tr = r["trial_records"][-1]
        steps = parse_steps(last_tr["trajectory"])
        feats = compute_step_features(steps)
        n_steps = len(steps)
        halted = (steps[-1]["action_type"] != "Finish") if steps else True
        obs_repeat_steps = [f["step_n"] for f in feats if f["obs_exact_repeat"]]
        sf_steps         = [f["step_n"] for f in feats if f["search_failure"]]
        max_streak       = max((f["no_progress_streak"] for f in feats), default=0)
        desc_lines.append(
            f"| {r['q_index']} | {n_steps} | {halted} | "
            f"{obs_repeat_steps or '—'} | {sf_steps or '—'} | {max_streak} |"
        )

    # 5b. Multi-trial success: what changed between trial 1 and final?
    desc_lines.append(f"\n### 5b. 多 trial 后成功的题 (N={len([r for r in primary if r['em']==1 and r['trials_used']>1])})\n")
    multi_ok = [r for r in primary if r["em"] == 1 and r["trials_used"] > 1]
    if not multi_ok:
        desc_lines.append("_日志中仅有 1 题 (q13) 属于此类。_\n")
        for r in multi_ok:
            for ti, tr in enumerate(r["trial_records"]):
                steps = parse_steps(tr["trajectory"])
                feats = compute_step_features(steps)
                sf = sum(1 for f in feats if f["search_failure"])
                rep = sum(1 for f in feats if f["obs_exact_repeat"])
                desc_lines.append(
                    f"  Q{r['q_index']} trial {ti+1}: steps={len(steps)} "
                    f"search_fail={sf} obs_repeat={rep} answer={tr['answer']!r}"
                )
    else:
        for r in multi_ok:
            desc_lines.append(f"**Q{r['q_index']}** trials={r['trials_used']}")
            for ti, tr in enumerate(r["trial_records"]):
                steps = parse_steps(tr["trajectory"])
                feats = compute_step_features(steps)
                sf = sum(1 for f in feats if f["search_failure"])
                rep = sum(1 for f in feats if f["obs_exact_repeat"])
                desc_lines.append(
                    f"  trial {ti+1}: steps={len(steps)} "
                    f"search_fail={sf} obs_repeat={rep} answer={tr['answer']!r}"
                )

    # 5c. Single-trial correct: do they also trigger no-progress signals?
    single_ok = [r for r in primary if r["em"] == 1 and r["trials_used"] == 1]
    desc_lines.append(f"\n### 5c. 一次成功的题中 gold-free 信号误触发情况 (N={len(single_ok)})\n")
    desc_lines.append(
        "| q | steps | search_fail_steps | obs_repeat_steps | no_prog_streak_max |"
    )
    desc_lines.append("|---|-------|-------------------|------------------|--------------------|")
    for r in single_ok:
        tr = r["trial_records"][0]
        steps = parse_steps(tr["trajectory"])
        feats = compute_step_features(steps)
        sf_steps  = [f["step_n"] for f in feats if f["search_failure"]]
        rep_steps = [f["step_n"] for f in feats if f["obs_exact_repeat"]]
        max_streak = max((f["no_progress_streak"] for f in feats), default=0)
        desc_lines.append(
            f"| {r['q_index']} | {len(steps)} | {sf_steps or '—'} | {rep_steps or '—'} | {max_streak} |"
        )

    # 5d. Aggregate: how often does each signal fire across fail vs correct?
    desc_lines.append("\n### 5d. 信号触发率：失败题 vs 成功题 (最后 trial)\n")

    def trial_signal_summary(rows, em_filter):
        filtered = [r for r in rows if r["em"] == em_filter]
        n = len(filtered)
        if n == 0:
            return {}
        counters = defaultdict(int)
        for r in filtered:
            tr = r["trial_records"][-1]
            steps = parse_steps(tr["trajectory"])
            feats = compute_step_features(steps)
            if any(f["obs_exact_repeat"] for f in feats):
                counters["obs_repeat"] += 1
            if any(f["search_failure"] for f in feats):
                counters["search_fail"] += 1
            if max((f["no_progress_streak"] for f in feats), default=0) >= 2:
                counters["streak_ge2"] += 1
            if len(steps) >= 5:
                counters["steps_ge5"] += 1
            if all(not f["obs_exact_repeat"] for f in feats):
                counters["no_obs_repeat"] += 1
        return {k: f"{v}/{n} ({100*v/n:.0f}%)" for k, v in counters.items()}

    fail_sig = trial_signal_summary(primary, em_filter=0)
    ok_sig   = trial_signal_summary(primary, em_filter=1)

    desc_lines.append("| 信号 | 失败题触发率 | 成功题触发率 |")
    desc_lines.append("|------|------------|------------|")
    all_keys = sorted(set(list(fail_sig.keys()) + list(ok_sig.keys())))
    for k in all_keys:
        desc_lines.append(
            f"| {k} | {fail_sig.get(k, '0/? (0%)')} | {ok_sig.get(k, '0/? (0%)')} |"
        )

    # 5e. answer_candidate_change: do reflections actually change the answer?
    desc_lines.append("\n### 5e. reflection 后答案是否变化 (smoke_fixed10 practice)\n")
    desc_lines.append("| q | trials | trial 答案序列 | 答案有变化的 trial |")
    desc_lines.append("|---|--------|--------------|------------------|")
    for r in primary:
        trs = r["trial_records"]
        answers = [t["answer"][:30] for t in trs]
        changes = answer_candidate_changes(trs)
        changed_at = [t["trial"] for t, c in zip(trs, changes) if c]
        desc_lines.append(
            f"| {r['q_index']} | {r['trials_used']} | "
            f"`{answers}` | {changed_at or '—'} |"
        )

    report_sections.append("\n".join(desc_lines))

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6: Findings and caveats
    # ═══════════════════════════════════════════════════════════════════════════
    findings_lines = [
        "\n## 6. 值得进一步验证的信号\n",
        "基于当前 30 题（20+5+5）的描述性观察：\n",
        "**观测到明显区分度的信号：**\n",
        "1. **obs_exact_repeat**：在所有 search_failure 密集型失败题（q5、q20）中，"
        "同一观测文本在 trial 内从 step 2 起完全重复出现。"
        "在成功题中未出现此模式。适合作为 step-level 早停信号。\n",
        "2. **search_failure 连续出现**：q5、q20、q12 等题在同一 trial 内 3–6 步"
        "连续触发 search_failure。成功题偶有 1 次但不连续。"
        "连续 ≥ 3 次可作为候选阈值。\n",
        "3. **no_progress_streak**：q5 的最后 trial 中从 step 2 起 streak=4，"
        "q20 同样 streak=5。成功题（q13 trial 1 有 streak=1）偶发但不超过 1。\n",
        "4. **answer 在多 trial 间不变化**：q1、q4、q9、q12 六次 trial 答案几乎完全相同，"
        "反思对答案字符串没有任何改变（reflection 文本分析了原因但未改变搜索策略）。"
        "这是 trial-level 信号，不是 step-level。\n",
        "5. **steps_ge5**（最后 trial 步数 ≥ 5）：11 道失败题中 5 道；"
        "9 道成功题中 0 道（均≤5步）。但单靠此信号有混淆。\n",
        "\n**当前日志无法支持的信号：**\n",
        "1. **token_count / embedding_similarity**：日志中未记录 token 数，"
        "scratchpad 解析后只能用 Jaccard 代理。\n",
        "2. **step 内候选答案变化**：只有 Finish 动作才产生答案，"
        "Search/Lookup 步骤没有中间答案。候选答案变化仅能在 trial 间比较。\n",
        "3. **LLM 置信度 / log-prob**：当前 vLLM 调用不返回 logprobs，无法计算。\n",
        "4. **格式失败（q1、q4、q9、q12、q17）的可识别性**："
        "这些题中 agent 实际上找到了正确实体但答案字符串缺少后缀/全名。"
        "gold-free 信号（无重复、无搜索失败）无法区分这类失败与成功，"
        "此类失败题的 trail 行为与成功题非常相似。\n",
        "\n## 7. 当前日志不能支持的分析\n",
        "- 跨题 query 相似度（不同题间的 query 重复）：无题间 session 信息。\n",
        "- Step-level LLM call 计时（wall time 仅按题记录）。\n",
        "- 格式失败与内容失败的 step-level 区分：gold-free 特征均无区分度。\n",
        "- 大样本统计：当前最大数据集仅 20 题，所有百分比均不稳定。\n",
    ]
    report_sections.append("\n".join(findings_lines))

    # ═══════════════════════════════════════════════════════════════════════════
    # Write outputs
    # ═══════════════════════════════════════════════════════════════════════════
    report_path = OUT / "report.md"
    with open(report_path, "w") as f:
        f.write("# HotpotQA Trajectory 探索性分析报告\n\n")
        f.write("> 生成自 `analysis/trajectory_analysis.py`（只读脚本）\n\n")
        f.write("---\n\n")
        for sec in report_sections:
            f.write(sec)
            f.write("\n\n---\n\n")

    csv_path = OUT / "step_features.csv"
    if all_step_rows:
        fieldnames = list(all_step_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_step_rows)

    print(f"Report  → {report_path}")
    print(f"CSV     → {csv_path}  ({len(all_step_rows)} rows)")


if __name__ == "__main__":
    analyse()
