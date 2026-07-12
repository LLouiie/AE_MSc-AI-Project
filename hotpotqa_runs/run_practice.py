import json, os, sys, time, argparse, subprocess
import llm
from environment import DistractorDocstore
from agents import ReactReflectAgent, ReflexionStrategy, normalize_answer
from llm import AnyOpenAILLM
from schedulers import build_scheduler
from consolidation import RulePool, consolidate

p = argparse.ArgumentParser()
p.add_argument("--sequence", required=True)        # splits/practice_random_s42.json 等
p.add_argument("--scheduler", required=True)       # never | fixed:10 | fixed:end
p.add_argument("--max-trials", type=int, default=6)  # 1 = No Reflection
p.add_argument("--run-name", required=True)        # 输出目录名
p.add_argument("--limit", type=int, default=None)  # 烟雾测试: --limit 20
args = p.parse_args()

RUN_DIR = os.path.join("runs", args.run_name)
os.makedirs(RUN_DIR, exist_ok=True)
LOG = os.path.join(RUN_DIR, "practice_log.jsonl")
MODEL = "Qwen/Qwen2.5-32B-Instruct"
BASE_URL = "http://localhost:8000/v1"

# ---- config 留档: 一个 run 一份完整配置 ----
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
json.dump({**vars(args), "model": MODEL, "max_steps": 6,
           "git_commit": commit, "started": time.strftime("%F %T")},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=2)

sched = build_scheduler(args.scheduler)
pool = RulePool(os.path.join(RUN_DIR, "rules.jsonl"))

questions = json.load(open(args.sequence))
if args.limit:
    questions = questions[:args.limit]

done = set()
if os.path.exists(LOG):
    done = {json.loads(l)["qid"] for l in open(LOG)}
    print(f"checkpoint: 已完成 {len(done)} 题, 续跑")

react_llm = AnyOpenAILLM(temperature=0, max_tokens=100, model_name=MODEL,
                         model_kwargs={"stop": "\n"},
                         openai_api_key="EMPTY", openai_api_base=BASE_URL)
reflect_llm = AnyOpenAILLM(temperature=0, max_tokens=250, model_name=MODEL,
                           openai_api_key="EMPTY", openai_api_base=BASE_URL)

pending_traj = []   # 自上次巩固以来的轨迹, 巩固后清空

with open(LOG, "a") as fout:
    for qi, ex in enumerate(questions, 1):
        if ex["id"] in done:
            continue
        docstore = DistractorDocstore(titles=list(ex["context"]["title"]),
                                      sentences=list(ex["context"]["sentences"]))
        agent = ReactReflectAgent(question=ex["question"], key=ex["answer"],
                                  max_steps=6, docstore=docstore,
                                  react_llm=react_llm, reflect_llm=reflect_llm,
                                  rules_text=pool.render())
        c0 = llm.call_counter
        t0 = time.time()
        trial_records = []
        for trial in range(args.max_trials):
            agent.run(reset=True, reflect_strategy=ReflexionStrategy.REFLEXION)
            trial_records.append({
                "trial": trial + 1,
                "em": int(agent.is_correct()),
                "trajectory": getattr(agent, "scratchpad", ""),
                "answer": agent.answer,
            })
            if agent.is_correct():
                break
        traj_text = trial_records[-1]["trajectory"]   # 最后一轮的 ReAct 轨迹
        pending_traj.append(traj_text)

        state = {"q_index": qi, "em": int(agent.is_correct()),
                 "trials_used": trial + 1}
        decision = sched.decide(state)
        if decision == "consolidate":
            consolidate(pool, pending_traj, reflect_llm, qi)
            pending_traj = []

        fout.write(json.dumps({
            "qid": ex["id"], "q_index": qi, "question": ex["question"],
            "gold": ex["answer"], "pred": agent.answer, "em": state["em"],
            "trials_used": state["trials_used"],
            "llm_calls": llm.call_counter - c0,
            "wall_s": round(time.time() - t0, 1),
            "reflections": list(agent.reflections),
            "trajectory": traj_text,
            "trial_records": trial_records,
            "decision": decision, "rules_size": len(pool.rules),
        }, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{qi}/{len(questions)}] em={state['em']} "
              f"trials={state['trials_used']} {decision} pool={len(pool.rules)}")

# end-only(k=∞): 练习全部结束后统一巩固一次
if args.scheduler == "fixed:end" and pending_traj:
    consolidate(pool, pending_traj, reflect_llm, len(questions))
    print("end-only 巩固完成")