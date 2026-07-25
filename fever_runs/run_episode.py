"""
run_episode.py — single-claim episode runner for FEVER.

Mirrors hotpotqa_runs/run_episode.py's design:
  - No reflection, no retry, no consolidation, no scheduler, no RulePool.
  - No inter-episode state (rules / reflections / affect).
  - Gold label is stored in agent.key but only read after run() returns,
    for offline EM computation inside this runner.

Usage (from fever_runs/):
    export WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (jy625@imperial.ac.uk)'
    python3 run_episode.py \
        --split practice \
        --run-name episode_smoke \
        [--limit 3] \
        [--max-steps 7]

Task data comes from data_loader.py's get_practice_tasks()/get_exam_tasks()
(the fixed seed=233 canonical-100 split over paper_dev.jsonl's first 7405
rows, 70/30 practice/exam) — not a raw file path, since FEVER's task
selection is index-based over one file, not a separately-ordered game/doc
store the way ALFWorld's task list vs. environment iteration order was
(see alfworld_runs_ae/environment.py's make_single_task_env for that bug).
"""

import json, os, sys, time, argparse, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hotpotqa_runs'))
import llm
from llm import AnyOpenAILLM

from environment import WikiSearchDocstore, normalize_label
from agents import FEVERReactAgent
from data_loader import get_practice_tasks, get_exam_tasks

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--split",     choices=["practice", "exam"], default="practice")
p.add_argument("--run-name",  required=True,          help="output directory name")
p.add_argument("--limit",     type=int, default=None, help="cap number of claims")
p.add_argument("--max-steps", type=int, default=7,    help="max ReAct steps per episode")
p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct"))
p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
args = p.parse_args()

RUN_DIR = os.path.join("runs", args.run_name)
os.makedirs(RUN_DIR, exist_ok=True)
LOG = os.path.join(RUN_DIR, "episode_log.jsonl")
MODEL = args.model
BASE_URL = args.base_url

# Fail fast, before any claim is processed: no anonymous default User-Agent.
WIKIPEDIA_USER_AGENT = os.environ.get("WIKIPEDIA_USER_AGENT", "").strip()
if not WIKIPEDIA_USER_AGENT:
    raise SystemExit(
        "ERROR: FEVER retrieval requires the WIKIPEDIA_USER_AGENT "
        "environment variable to be set, e.g.\n"
        "  export WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (jy625@imperial.ac.uk)'\n"
        "Refusing to send anonymous requests to the Wikipedia API."
    )

# ── config snapshot ───────────────────────────────────────────────────────────
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
json.dump({**vars(args), "model": MODEL, "git_commit": commit,
           "started": time.strftime("%F %T")},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=2)

# ── data ──────────────────────────────────────────────────────────────────────
tasks = get_practice_tasks() if args.split == "practice" else get_exam_tasks()
if args.limit:
    tasks = tasks[: args.limit]

done = set()
if os.path.exists(LOG):
    done = {json.loads(l)["qid"] for l in open(LOG)}
    print(f"checkpoint: {len(done)} done, resuming")

# ── LLM (react only; no reflect_llm needed) ───────────────────────────────────
react_llm = AnyOpenAILLM(
    temperature=0, max_tokens=100, model_name=MODEL,
    model_kwargs={"stop": "\n"},
    openai_api_key="EMPTY", openai_api_base=BASE_URL,
)

# ── episode loop ──────────────────────────────────────────────────────────────
with open(LOG, "a") as fout:
    for qi, task in enumerate(tasks, 1):
        if task["id"] in done:
            continue

        # Fresh docstore per claim -> no state leaks across episodes.
        docstore = WikiSearchDocstore(user_agent=WIKIPEDIA_USER_AGENT)

        agent = FEVERReactAgent(
            claim=task["question"],
            key=task["answer"],
            docstore=docstore,
            max_steps=args.max_steps,
            react_llm=react_llm,
        )

        c0 = llm.call_counter
        t0 = time.time()
        agent.run(reset=True)

        # ── offline evaluation (first time gold is used) ──────────────────
        pred = agent.answer
        gold = task["answer"]
        em = int(normalize_label(pred) == normalize_label(gold))

        record = {
            "qid":        task["id"],
            "q_index":    qi,
            "claim":      task["question"],
            "gold":       gold,
            "pred":       pred,
            "em":         em,
            "llm_calls":  llm.call_counter - c0,
            "wall_s":     round(time.time() - t0, 1),
            "trajectory": agent.scratchpad,
            "steps_used": agent.step_n - 1,
            "finished":   agent.is_finished(),
            "halted":     not agent.is_finished(),
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{qi}/{len(tasks)}] em={em} pred={pred!r} gold={gold!r} "
              f"steps={record['steps_used']} finished={record['finished']} "
              f"calls={record['llm_calls']}")

# ── summary ───────────────────────────────────────────────────────────────────
rows = [json.loads(l) for l in open(LOG)]
n = len(rows)
print(f"\n===== EPISODE DONE =====")
print(f"N={n}  "
      f"EM={sum(r['em'] for r in rows)/n:.3f}  "
      f"finished={sum(r['finished'] for r in rows)}/{n}  "
      f"halted={sum(r['halted'] for r in rows)}/{n}  "
      f"avg_calls={sum(r['llm_calls'] for r in rows)/n:.1f}")
