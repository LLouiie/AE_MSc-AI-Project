import json, os, sys, time, argparse, subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hotpotqa_runs'))
import llm
from llm import AnyOpenAILLM

from environment import WikiSearchDocstore
from agents import FEVERReactAgent

p = argparse.ArgumentParser()
p.add_argument("--exam", required=True,
               help="Path to JSON file with [{id, question, answer}] exam tasks")
p.add_argument("--rules-from", default=None,
               help="Path to rules.jsonl from a practice run (frozen rule pool)")
p.add_argument("--run-name", required=True,
               help="Output directory name under fever_runs/runs/")
p.add_argument("--limit", type=int, default=None,
               help="Smoke test: stop after N claims")
args = p.parse_args()

RUN_DIR = os.path.join("runs", args.run_name)
os.makedirs(RUN_DIR, exist_ok=True)
LOG = os.path.join(RUN_DIR, "exam_log.jsonl")
MODEL    = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()


def _load_rules(rules_path: str) -> str:
    if not rules_path or not os.path.exists(rules_path):
        return ""
    lines = [json.loads(l) for l in open(rules_path)]
    if not lines:
        return ""
    final = lines[-1]["pool"]
    return "\n".join(f"{i+1}. {r['text']}" for i, r in enumerate(final))


rules_text = _load_rules(args.rules_from)
json.dump({**vars(args), "model": MODEL, "max_steps": 7,
           "rules_loaded": bool(rules_text), "git_commit": commit,
           "started": time.strftime("%F %T")},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=2)

tasks = json.load(open(args.exam))
if args.limit:
    tasks = tasks[:args.limit]

done = set()
if os.path.exists(LOG):
    done = {json.loads(l)["qid"] for l in open(LOG)}
    print(f"checkpoint: already done {len(done)} tasks, resuming")

react_llm = AnyOpenAILLM(temperature=0, max_tokens=100, model_name=MODEL,
                          model_kwargs={"stop": "\n"},
                          openai_api_key="EMPTY", openai_api_base=BASE_URL)

n_correct = 0
with open(LOG, "a") as fout:
    for qi, task in enumerate(tasks, 1):
        qid = str(task["id"])
        if qid in done:
            n_correct += json.loads([l for l in open(LOG)
                                     if json.loads(l)["qid"] == qid][0])["em"]
            continue

        docstore = WikiSearchDocstore()
        agent = FEVERReactAgent(
            claim=task["question"],
            key=task["answer"],
            docstore=docstore,
            max_steps=7,
            react_llm=react_llm,
            rules_text=rules_text,
        )

        c0 = llm.call_counter
        t0 = time.time()
        agent.run(reset=True)

        em = int(agent.is_correct())
        n_correct += em

        fout.write(json.dumps({
            "qid": qid, "q_index": qi,
            "question": task["question"],
            "gold": task["answer"], "pred": agent.answer,
            "em": em,
            "llm_calls": llm.call_counter - c0,
            "wall_s": round(time.time() - t0, 1),
            "trajectory": agent.scratchpad,
        }, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{qi}/{len(tasks)}] em={em} acc={n_correct}/{qi}={n_correct/qi:.2%}")

print(f"\nFinal accuracy: {n_correct}/{len(tasks)} = {n_correct/len(tasks):.2%}")
