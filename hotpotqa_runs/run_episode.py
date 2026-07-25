"""
run_episode.py — single-question episode runner.

Each HotpotQA question is treated as an independent episode:
  - No consolidation, no scheduler, no RulePool.
  - --strategy react (default): no reflection/retry, no inter-episode state.
    online_feedback=False: CORRECT/INCORRECT never written to scratchpad;
    is_correct() never called during run(); gold never reaches any LLM
    prompt. Gold is stored in agent.key but only accessed after run()
    returns, for offline EM/F1 computation inside this runner.
  - --strategy reflexion: up to --max-trials attempts per question, each
    failed trial's trajectory reflected on (ReactReflectAgent) before the
    next retry, matching ae/baselines/reflexion.py's ALFWorld pattern
    (rules_text pinned to "" so RulePool/consolidation is never imported).
    Reflexion inherently needs is_correct() mid-run to decide whether to
    retry, so gold reaches the loop (not the LLM prompt) here.

Usage (from hotpotqa_runs/):
    python3 run_episode.py \
        --data /path/to/questions.json \
        --run-name episode_smoke \
        [--strategy react|reflexion] [--max-trials 4] \
        [--limit 3] \
        [--max-steps 6]

Question data format (same fields for practice/exam splits), either:
    .json   list of { "id": str, "question": str, "answer": str,
                       "context": { "title": [...], "sentences": [[...], ...] } }
    .joblib pandas DataFrame with the same per-row fields (e.g. Reflexion's
            official hotpot-qa-distractor-sample.joblib), converted to a list
            of row-dicts via DataFrame.to_dict("records").
"""

import json, os, sys, time, argparse, subprocess
import llm
from environment import DistractorDocstore
from wiki_docstore import WikipediaDocstore
from agents import ReactAgent, ReactReflectAgent, normalize_answer
from llm import AnyOpenAILLM

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--data",      required=True,          help="JSON or .joblib question file")
p.add_argument("--run-name",  required=True,          help="output directory name")
p.add_argument("--limit",     type=int, default=None, help="cap number of questions")
p.add_argument("--max-steps", type=int, default=6,    help="max ReAct steps per episode")
p.add_argument("--strategy",  choices=["react", "reflexion"], default="react")
p.add_argument("--max-trials", type=int, default=4,    help="reflexion only: max retry trials")
p.add_argument("--retrieval", choices=["wikipedia", "distractor"], default="wikipedia",
               help="wikipedia: live MediaWiki API, question-only, ignores ex['context']; "
                    "distractor: local DistractorDocstore built from ex['context']")
p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct"))
p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
args = p.parse_args()

RUN_DIR = os.path.join("runs", args.run_name)
os.makedirs(RUN_DIR, exist_ok=True)
LOG            = os.path.join(RUN_DIR, "episode_log.jsonl")
RETRIEVAL_LOG  = os.path.join(RUN_DIR, "retrieval_log.jsonl")
MODEL = args.model
BASE_URL = args.base_url
DATA_FORMAT = "joblib" if args.data.lower().endswith(".joblib") else "json"

# Fail fast, before any question is processed: no anonymous default User-Agent.
WIKIPEDIA_USER_AGENT = os.environ.get("WIKIPEDIA_USER_AGENT", "").strip()
if args.retrieval == "wikipedia" and not WIKIPEDIA_USER_AGENT:
    raise SystemExit(
        "ERROR: --retrieval wikipedia requires the WIKIPEDIA_USER_AGENT "
        "environment variable to be set, e.g.\n"
        "  export WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (jy625@imperial.ac.uk)'\n"
        "Refusing to send anonymous requests to the Wikipedia API."
    )

# ── config snapshot ───────────────────────────────────────────────────────────
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
json.dump({**vars(args), "model": MODEL, "git_commit": commit,
           "data_path": os.path.abspath(args.data), "data_format": DATA_FORMAT,
           "retrieval_mode": args.retrieval,
           "user_agent": WIKIPEDIA_USER_AGENT if args.retrieval == "wikipedia" else None,
           "started": time.strftime("%F %T")},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=2)

# ── data ──────────────────────────────────────────────────────────────────────
if DATA_FORMAT == "joblib":
    import joblib
    # hotpot-qa-distractor-sample.joblib was pickled by an older pandas that
    # had pandas.core.indexes.numeric (Int64Index etc.); modern pandas
    # removed that module, so loading raises ModuleNotFoundError unless we
    # register a compatibility shim first.
    import sys, types
    import pandas as pd
    _shim = types.ModuleType("pandas.core.indexes.numeric")
    for _name in ["Int64Index", "UInt64Index", "Float64Index", "NumericIndex"]:
        setattr(_shim, _name, pd.Index)
    sys.modules.setdefault("pandas.core.indexes.numeric", _shim)
    questions = joblib.load(args.data)
    if hasattr(questions, "to_dict"):
        questions = questions.to_dict("records")
else:
    questions = json.load(open(args.data))
if args.limit:
    questions = questions[: args.limit]

done = set()
if os.path.exists(LOG):
    done = {json.loads(l)["qid"] for l in open(LOG)}
    print(f"checkpoint: {len(done)} done, resuming")

# ── LLM ────────────────────────────────────────────────────────────────────────
react_llm = AnyOpenAILLM(
    temperature=0, max_tokens=100, model_name=MODEL,
    model_kwargs={"stop": "\n"},
    openai_api_key="EMPTY", openai_api_base=BASE_URL,
)
reflect_llm = None
if args.strategy == "reflexion":
    reflect_llm = AnyOpenAILLM(
        temperature=0, max_tokens=250, model_name=MODEL,
        openai_api_key="EMPTY", openai_api_base=BASE_URL,
    )

# ── episode loop ──────────────────────────────────────────────────────────────
with open(LOG, "a") as fout:
    for qi, ex in enumerate(questions, 1):
        if ex["id"] in done:
            continue

        if args.retrieval == "wikipedia":
            # Question-only: never touch ex["context"] or ex["supporting_facts"].
            # Fresh docstore per question -> no state leaks across episodes.
            docstore = WikipediaDocstore(
                user_agent=WIKIPEDIA_USER_AGENT,
                log_path=RETRIEVAL_LOG,
            )
        else:
            docstore = DistractorDocstore(
                titles=list(ex["context"]["title"]),
                sentences=list(ex["context"]["sentences"]),
            )

        c0 = llm.call_counter
        t0 = time.time()

        if args.strategy == "react":
            # online_feedback=False:
            #   • step() Finish branch skips is_correct() entirely
            #   • scratchpad never receives "Answer is CORRECT/INCORRECT"
            #   • gold (agent.key) is stored but never read during run()
            agent = ReactAgent(
                question=ex["question"],
                key=ex["answer"],          # stored only; never accessed during run()
                max_steps=args.max_steps,
                docstore=docstore,
                react_llm=react_llm,
                online_feedback=False,     # ← gold leakage cutoff
            )
            agent.run(reset=True)
            trials_used = 1
        else:
            agent = ReactReflectAgent(
                question=ex["question"],
                key=ex["answer"],
                max_steps=args.max_steps,
                docstore=docstore,
                react_llm=react_llm,
                reflect_llm=reflect_llm,
                rules_text="",              # never imports consolidation.py/schedulers.py
            )
            for trial in range(args.max_trials):
                agent.run(reset=True)
                if agent.is_correct():
                    break
            trials_used = trial + 1

        # ── offline evaluation (first time gold is used for react; already
        #    used mid-loop by ReactReflectAgent.run() for reflexion) ───────
        pred = agent.answer
        gold = ex["answer"]
        pt     = normalize_answer(pred).split()
        gt     = normalize_answer(gold).split()
        common = set(pt) & set(gt)
        em = int(normalize_answer(pred) == normalize_answer(gold))
        if not common:
            f1 = 0.0
        else:
            prec = len(common) / len(pt) if pt else 0.0
            rec  = len(common) / len(gt) if gt else 0.0
            f1   = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        steps_used = agent.step_n - 1   # step_n starts at 1, incremented after each step
        finished   = agent.is_finished()
        halted     = not finished

        record = {
            "qid":        ex["id"],
            "q_index":    qi,
            "question":   ex["question"],
            "gold":       gold,
            "pred":       pred,
            "em":         em,
            "f1":         f1,
            "trials_used": trials_used,
            "llm_calls":  llm.call_counter - c0,
            "wall_s":     round(time.time() - t0, 1),
            "trajectory": agent.scratchpad,
            "steps_used": steps_used,
            "finished":   finished,
            "halted":     halted,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{qi}/{len(questions)}] em={em} f1={f1:.3f} trials={trials_used} "
              f"steps={steps_used} finished={finished} halted={halted} "
              f"calls={record['llm_calls']}")

# ── summary ───────────────────────────────────────────────────────────────────
rows = [json.loads(l) for l in open(LOG)]
n = len(rows)
print(f"\n===== EPISODE DONE =====")
print(f"N={n}  "
      f"EM={sum(r['em'] for r in rows)/n:.3f}  "
      f"F1={sum(r['f1'] for r in rows)/n:.3f}  "
      f"finished={sum(r['finished'] for r in rows)}/{n}  "
      f"halted={sum(r['halted'] for r in rows)}/{n}  "
      f"avg_calls={sum(r['llm_calls'] for r in rows)/n:.1f}")
