import json, os, sys, time, argparse, subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # moved into legacy/, parent has llm.py/environment.py/agents.py
import llm
from environment import DistractorDocstore
from agents import ReactReflectAgent, ReflexionStrategy, normalize_answer
from llm import AnyOpenAILLM

p = argparse.ArgumentParser()
p.add_argument("--exam", required=True)          # splits/exam_s42.json
p.add_argument("--rules-from", required=True)    # 某个练习 run 的 rules.jsonl
p.add_argument("--run-name", required=True)
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

RUN_DIR = os.path.join("runs", args.run_name)
os.makedirs(RUN_DIR, exist_ok=True)
LOG = os.path.join(RUN_DIR, "exam_log.jsonl")
MODEL = "Qwen/Qwen2.5-32B-Instruct"
BASE_URL = "http://localhost:8000/v1"

# 读冻结规则池:取 rules.jsonl 最后一行(最终池快照)
rules_text = ""
if os.path.exists(args.rules_from):
    last = None
    for line in open(args.rules_from):
        if line.strip():
            last = json.loads(line)
    if last:
        rules_text = "\n".join(f"{i+1}. {r['text']}"
                               for i, r in enumerate(last["pool"]))
print(f"注入规则 {rules_text.count(chr(10))+1 if rules_text else 0} 条")

# config 留档
commit = subprocess.run(["git","rev-parse","--short","HEAD"],
                        capture_output=True, text=True).stdout.strip()
json.dump({**vars(args), "model": MODEL, "git_commit": commit,
           "n_rules": rules_text.count(chr(10))+1 if rules_text else 0,
           "started": time.strftime("%F %T")},
          open(os.path.join(RUN_DIR,"config.json"),"w"), indent=2)

questions = json.load(open(args.exam))
if args.limit:
    questions = questions[:args.limit]

done = set()
if os.path.exists(LOG):
    done = {json.loads(l)["qid"] for l in open(LOG)}
    print(f"checkpoint: 已完成 {len(done)} 题, 续跑")

# 只需 react_llm(单轮作答, 不反思)
react_llm = AnyOpenAILLM(temperature=0, max_tokens=100, model_name=MODEL,
                         model_kwargs={"stop":"\n"},
                         openai_api_key="EMPTY", openai_api_base=BASE_URL)

with open(LOG, "a") as fout:
    for qi, ex in enumerate(questions, 1):
        if ex["id"] in done:
            continue
        docstore = DistractorDocstore(titles=list(ex["context"]["title"]),
                                      sentences=list(ex["context"]["sentences"]))
        agent = ReactReflectAgent(question=ex["question"], key=ex["answer"],
                                  max_steps=6, docstore=docstore,
                                  react_llm=react_llm, reflect_llm=react_llm,
                                  rules_text=rules_text)
        c0 = llm.call_counter
        t0 = time.time()
        # 单轮: 只 run 一次, 不进 reflexion 重试循环
        agent.run(reset=True, reflect_strategy=ReflexionStrategy.NONE)

        pred = agent.answer
        pt, gt = normalize_answer(pred).split(), normalize_answer(ex["answer"]).split()
        common = set(pt) & set(gt)
        if not common: f1 = 0.0
        else:
            prec, rec = len(common)/len(pt) if pt else 0, len(common)/len(gt) if gt else 0
            f1 = round(2*prec*rec/(prec+rec), 4) if (prec+rec)>0 else 0.0

        fout.write(json.dumps({
            "qid": ex["id"], "q_index": qi, "question": ex["question"],
            "gold": ex["answer"], "pred": pred,
            "em": int(agent.is_correct()), "f1": f1,
            "llm_calls": llm.call_counter - c0,
            "wall_s": round(time.time()-t0, 1),
            "trajectory": getattr(agent, "scratchpad", ""),
        }, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{qi}/{len(questions)}] em={int(agent.is_correct())} f1={f1} calls={llm.call_counter-c0}")

rows = [json.loads(l) for l in open(LOG)]
n = len(rows)
print(f"\n===== EXAM DONE =====")
print(f"N={n}  EM={sum(r['em'] for r in rows)/n:.3f}  "
      f"F1={sum(r['f1'] for r in rows)/n:.3f}  "
      f"avg_calls={sum(r['llm_calls'] for r in rows)/n:.1f}")