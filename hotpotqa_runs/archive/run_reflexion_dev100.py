import json, os, time
import llm                          # ← 新增：访问 llm.call_counter
from environment import DistractorDocstore
from agents import ReactReflectAgent, ReflexionStrategy, normalize_answer
from llm import AnyOpenAILLM

MAX_TRIALS = 6
MAX_STEPS = 6
DATA_PATH = os.path.expanduser("~/projects/AE/data/hotpotqa/dev.json")
OUT_PATH = "results/reflexion_dev100.jsonl"
MODEL = "Qwen/Qwen2.5-32B-Instruct"
BASE_URL = "http://localhost:8001/v1"
API_KEY = "EMPTY"

os.makedirs("results", exist_ok=True)

dev = json.load(open(DATA_PATH))
print(f"加载 {len(dev)} 题")

done = set()
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        for line in f:
            done.add(json.loads(line)["qid"])
    print(f"已完成 {len(done)} 题，继续跑剩下的")

react_llm = AnyOpenAILLM(
    temperature=0, max_tokens=100, model_name=MODEL,
    model_kwargs={"stop": "\n"}, openai_api_key=API_KEY, openai_api_base=BASE_URL)
reflect_llm = AnyOpenAILLM(
    temperature=0, max_tokens=250, model_name=MODEL,
    openai_api_key=API_KEY, openai_api_base=BASE_URL)

with open(OUT_PATH, "a") as fout:
    for i, ex in enumerate(dev):
        qid = ex["id"]
        if qid in done:
            continue

        question = ex["question"]
        gold = ex["answer"]
        docstore = DistractorDocstore(
            titles=list(ex["context"]["title"]),
            sentences=list(ex["context"]["sentences"])
        )

        agent = ReactReflectAgent(
            question=question, key=gold, max_steps=MAX_STEPS,
            docstore=docstore, react_llm=react_llm, reflect_llm=reflect_llm,
        )

        call_before = llm.call_counter  # ← 新增：记住跑这题前的计数
        t0 = time.time()
        for trial in range(MAX_TRIALS):
            agent.run(reset=True, reflect_strategy=ReflexionStrategy.REFLEXION)
            if agent.is_correct():
                break
        wall = round(time.time() - t0, 1)

        pred_tokens = normalize_answer(agent.answer).split()
        gold_tokens = normalize_answer(gold).split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            f1 = 0.0
        else:
            prec = len(common) / len(pred_tokens) if pred_tokens else 0
            rec = len(common) / len(gold_tokens) if gold_tokens else 0
            f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        result = {
            "qid": qid, "question": question, "gold": gold,
            "pred": agent.answer, "em": int(agent.is_correct()),
            "f1": f1, "trials_used": trial + 1,
            "reflections": list(agent.reflections), "wall_s": wall,
            "llm_calls": llm.call_counter - call_before,  # ← 新增：这道题的 LLM 调用次数
        }
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()

        status = "✓" if result["em"] else "✗"
        print(f"[{len(done)+1}/{len(dev)}] {status}  trials={result['trials_used']}  f1={f1}  calls={result['llm_calls']}  wall={wall}s")
        done.add(qid)

rows = [json.loads(l) for l in open(OUT_PATH)]
n = len(rows)
em = sum(r["em"] for r in rows) / n
avg_f1 = sum(r["f1"] for r in rows) / n
avg_trials = sum(r["trials_used"] for r in rows) / n
avg_calls = sum(r["llm_calls"] for r in rows) / n
print(f"\n===== DONE =====")
print(f"N={n}  EM={em:.3f}  F1={avg_f1:.3f}  avg_trials={avg_trials:.2f}  avg_calls={avg_calls:.1f}")