"""
ALFWorld exam run: single-trial evaluation on held-out 34 tasks,
optionally with frozen rules from a practice run.

Usage:
    cd ~/projects/reflexion/alfworld_runs_ae
    python run_exam.py --rules-from runs/smoke_af10/rules.jsonl --run-name exam_af10
"""

import json, os, sys, time, argparse, subprocess

# moved into legacy/, one extra directory level below the original run_exam.py location
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))                  # alfworld_runs_ae: environment.py, agents.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'hotpotqa_runs'))  # llm.py
import llm
from llm import AnyOpenAILLM

from environment import make_alfworld_env, get_exam_tasks, get_task_type, TASKS_FILE
from agents import ALFWorldAgent

p = argparse.ArgumentParser()
p.add_argument("--rules-from", default=None,
               help="Path to rules.jsonl from a practice run")
p.add_argument("--run-name", required=True)
p.add_argument("--limit", type=int, default=None)
p.add_argument("--tasks-file", default=TASKS_FILE)
args = p.parse_args()

RUN_DIR = os.path.join("runs", args.run_name)
os.makedirs(RUN_DIR, exist_ok=True)
LOG = os.path.join(RUN_DIR, "exam_log.jsonl")
MODEL    = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True,
                        cwd=os.path.dirname(__file__)).stdout.strip()


def _load_rules(path):
    if not path or not os.path.exists(path):
        return ""
    lines = [json.loads(l) for l in open(path)]
    if not lines:
        return ""
    pool = lines[-1]["pool"]
    return "\n".join(f"{i+1}. {r['text']}" for i, r in enumerate(pool))


rules_text = _load_rules(args.rules_from)
json.dump({**vars(args), "model": MODEL, "rules_loaded": bool(rules_text),
           "git_commit": commit, "started": time.strftime("%F %T")},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=2)

tasks = get_exam_tasks(args.tasks_file)
if args.limit:
    tasks = tasks[:args.limit]

done = set()
if os.path.exists(LOG):
    done = {json.loads(l)["env_name"] for l in open(LOG)}
    print(f"checkpoint: already done {len(done)} tasks, resuming")

act_llm = AnyOpenAILLM(temperature=0, max_tokens=50, model_name=MODEL,
                        model_kwargs={"stop": ['\n']},
                        openai_api_key="EMPTY", openai_api_base=BASE_URL)

env = make_alfworld_env()
agent = ALFWorldAgent(act_llm)

n_success = 0
with open(LOG, "a") as fout:
    for qi, task in enumerate(tasks, 1):
        env_name = task.get('env_name', f'task_{qi}')
        if env_name in done:
            continue

        goal = task['goal']
        task_type = get_task_type(env_name)

        c0 = llm.call_counter
        t0 = time.time()
        traj, success = agent.run(env, goal, task_type,
                                  rules_text=rules_text, to_print=True)
        n_success += int(success)

        fout.write(json.dumps({
            "env_name": env_name, "q_index": qi, "goal": goal,
            "success": int(success),
            "llm_calls": llm.call_counter - c0,
            "wall_s": round(time.time() - t0, 1),
            "trajectory": traj,
        }, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{qi}/{len(tasks)}] success={success} "
              f"acc={n_success}/{qi}={n_success/qi:.2%}")

env.close()
print(f"\nFinal: {n_success}/{len(tasks)} = {n_success/len(tasks):.2%}")
