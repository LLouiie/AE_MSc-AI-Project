"""
ALFWorld practice run with Reflexion + ExpeL-style consolidation.

Setup:
    alfworld-download
    export ALFWORLD_DATA=~/alfworld_data   # or wherever it downloaded

Usage:
    cd ~/projects/reflexion/alfworld_runs_ae
    python run_practice.py --scheduler fixed:10 --max-trials 4 --run-name smoke_af10 --limit 10
"""

import json, os, sys, time, argparse, subprocess

# moved into legacy/, one extra directory level below the original run_practice.py location
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))                          # alfworld_runs_ae: environment.py, agents.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'hotpotqa_runs'))          # llm.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'hotpotqa_runs', 'legacy'))  # schedulers.py, consolidation.py
import llm
from llm import AnyOpenAILLM
from schedulers import build_scheduler
from consolidation import RulePool, consolidate

from environment import (
    make_alfworld_env, get_practice_tasks, get_exam_tasks, get_task_type, TASKS_FILE
)
from agents import ALFWorldReflectAgent

p = argparse.ArgumentParser()
p.add_argument("--scheduler", required=True, help="never | fixed:N | fixed:end")
p.add_argument("--max-trials", type=int, default=4,
               help="Max Reflexion trials per task (1 = no reflection)")
p.add_argument("--run-name", required=True, help="Output dir under alfworld_runs_ae/runs/")
p.add_argument("--limit", type=int, default=None, help="Smoke test: stop after N tasks")
p.add_argument("--tasks-file", default=TASKS_FILE,
               help="Path to alfworld_tasks_suffix.json")
args = p.parse_args()

RUN_DIR = os.path.join("runs", args.run_name)
os.makedirs(RUN_DIR, exist_ok=True)
LOG = os.path.join(RUN_DIR, "practice_log.jsonl")
MODEL    = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True,
                        cwd=os.path.dirname(__file__)).stdout.strip()
json.dump({**vars(args), "model": MODEL, "max_steps": 50,
           "git_commit": commit, "started": time.strftime("%F %T")},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=2)

sched = build_scheduler(args.scheduler)
pool  = RulePool(os.path.join(RUN_DIR, "rules.jsonl"))

tasks = get_practice_tasks(args.tasks_file)
if args.limit:
    tasks = tasks[:args.limit]

done = set()
if os.path.exists(LOG):
    done = {json.loads(l)["env_name"] for l in open(LOG)}
    print(f"checkpoint: already done {len(done)} tasks, resuming")

act_llm = AnyOpenAILLM(temperature=0, max_tokens=50, model_name=MODEL,
                        model_kwargs={"stop": ['\n']},
                        openai_api_key="EMPTY", openai_api_base=BASE_URL)
reflect_llm = AnyOpenAILLM(temperature=0, max_tokens=300, model_name=MODEL,
                            openai_api_key="EMPTY", openai_api_base=BASE_URL)

env = make_alfworld_env()
pending_traj = []

with open(LOG, "a") as fout:
    for qi, task in enumerate(tasks, 1):
        env_name = task.get('env_name', f'task_{qi}')
        if env_name in done:
            continue

        goal = task['goal']
        task_type = get_task_type(env_name)

        agent = ALFWorldReflectAgent(act_llm=act_llm, reflect_llm=reflect_llm)
        c0 = llm.call_counter
        t0 = time.time()
        trial_records = []

        for trial in range(args.max_trials):
            success = agent.run_trial(env, goal, task_type,
                                      rules_text=pool.render(), to_print=True)
            trial_records.append({
                "trial": trial + 1,
                "success": int(success),
                "trajectory": agent.last_trajectory,
            })
            if success:
                break
            if trial < args.max_trials - 1:
                reflection = agent.reflect(goal, task_type)
                print(f"  Reflection: {reflection[:120]}...")

        traj_text = trial_records[-1]["trajectory"]
        if agent.is_success:
            pending_traj.append(traj_text)

        state = {"q_index": qi, "em": int(agent.is_success), "trials_used": trial + 1}
        decision = sched.decide(state)
        if decision == "consolidate":
            consolidate(pool, pending_traj, reflect_llm, qi)
            pending_traj = []

        fout.write(json.dumps({
            "env_name": env_name, "q_index": qi, "goal": goal,
            "success": int(agent.is_success),
            "trials_used": trial + 1,
            "llm_calls": llm.call_counter - c0,
            "wall_s": round(time.time() - t0, 1),
            "reflections": agent.reflections,
            "trajectory": traj_text,
            "trial_records": trial_records,
            "decision": decision, "rules_size": len(pool.rules),
        }, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{qi}/{len(tasks)}] success={agent.is_success} "
              f"trials={trial+1} {decision} pool={len(pool.rules)}")

env.close()

if args.scheduler == "fixed:end" and pending_traj:
    consolidate(pool, pending_traj, reflect_llm, len(tasks))
    print("end-only consolidation done")
