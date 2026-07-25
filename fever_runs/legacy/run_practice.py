import json, os, sys, time, argparse, subprocess

# moved into legacy/, one extra directory level below the original run_practice.py location
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))                          # fever_runs: environment.py, agents.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'hotpotqa_runs'))          # llm.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'hotpotqa_runs', 'legacy'))  # schedulers.py, consolidation.py
import llm
from llm import AnyOpenAILLM
from schedulers import build_scheduler
from consolidation import RulePool, consolidate

from environment import WikiSearchDocstore, FEVEREnv
from agents import FEVERReactReflectAgent

p = argparse.ArgumentParser()
p.add_argument("--sequence", required=True,
               help="Path to JSON file with [{id, question, answer}] records "
                    "(use data_loader.py to generate practice_fever70.json)")
p.add_argument("--scheduler", required=True,
               help="never | fixed:N | fixed:end")
p.add_argument("--max-trials", type=int, default=6,
               help="Max reflection trials per claim (1 = no reflection)")
p.add_argument("--run-name", required=True,
               help="Output directory name under fever_runs/runs/")
p.add_argument("--limit", type=int, default=None,
               help="Smoke test: stop after N claims")
args = p.parse_args()

RUN_DIR = os.path.join("runs", args.run_name)
os.makedirs(RUN_DIR, exist_ok=True)
LOG = os.path.join(RUN_DIR, "practice_log.jsonl")
MODEL    = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()
json.dump({**vars(args), "model": MODEL, "max_steps": 7,
           "git_commit": commit, "started": time.strftime("%F %T")},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=2)

sched = build_scheduler(args.scheduler)
pool  = RulePool(os.path.join(RUN_DIR, "rules.jsonl"))

tasks = json.load(open(args.sequence))
if args.limit:
    tasks = tasks[:args.limit]

done = set()
if os.path.exists(LOG):
    done = {json.loads(l)["qid"] for l in open(LOG)}
    print(f"checkpoint: already done {len(done)} tasks, resuming")

react_llm = AnyOpenAILLM(temperature=0, max_tokens=100, model_name=MODEL,
                          model_kwargs={"stop": "\n"},
                          openai_api_key="EMPTY", openai_api_base=BASE_URL)
reflect_llm = AnyOpenAILLM(temperature=0, max_tokens=250, model_name=MODEL,
                            openai_api_key="EMPTY", openai_api_base=BASE_URL)

pending_traj = []

with open(LOG, "a") as fout:
    for qi, task in enumerate(tasks, 1):
        qid = str(task["id"])
        if qid in done:
            continue

        docstore = WikiSearchDocstore()
        agent = FEVERReactReflectAgent(
            claim=task["question"],
            key=task["answer"],
            docstore=docstore,
            max_steps=7,
            react_llm=react_llm,
            reflect_llm=reflect_llm,
            rules_text=pool.render(),
        )

        c0 = llm.call_counter
        t0 = time.time()
        trial_records = []

        for trial in range(args.max_trials):
            agent.run(reset=True, reflect_strategy='reflexion')
            trial_records.append({
                "trial": trial + 1,
                "em": int(agent.is_correct()),
                "trajectory": agent.scratchpad,
                "answer": agent.answer,
            })
            if agent.is_correct():
                break

        traj_text = trial_records[-1]["trajectory"]
        pending_traj.append(traj_text)

        state = {"q_index": qi, "em": int(agent.is_correct()),
                 "trials_used": trial + 1}
        decision = sched.decide(state)
        if decision == "consolidate":
            consolidate(pool, pending_traj, reflect_llm, qi)
            pending_traj = []

        fout.write(json.dumps({
            "qid": qid, "q_index": qi,
            "question": task["question"],
            "gold": task["answer"], "pred": agent.answer,
            "em": state["em"],
            "trials_used": state["trials_used"],
            "llm_calls": llm.call_counter - c0,
            "wall_s": round(time.time() - t0, 1),
            "reflections": list(agent.reflections),
            "trajectory": traj_text,
            "trial_records": trial_records,
            "decision": decision, "rules_size": len(pool.rules),
        }, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{qi}/{len(tasks)}] em={state['em']} "
              f"trials={state['trials_used']} {decision} pool={len(pool.rules)}")

if args.scheduler == "fixed:end" and pending_traj:
    consolidate(pool, pending_traj, reflect_llm, len(tasks))
    print("end-only consolidation done")
