import json, random

rows = [json.loads(l) for l in open("runs/smoke_fixed10/practice_log.jsonl")]
rng = random.Random(0)
correct = [r for r in rows if r["em"] == 1]
wrong = [r for r in rows if r["em"] == 0]
pick = rng.sample(correct, min(3, len(correct))) + rng.sample(wrong, min(3, len(wrong)))

for r in pick:
    print(f"\n{'='*70}")
    print(f"qid={r['qid']}  q{r['q_index']}  em={r['em']}  trials={r['trials_used']}  calls={r['llm_calls']}")
    print(f"Q: {r['question']}")
    print(f"gold: {r['gold']!r}")
    print(f"pred: {r['pred']!r}")
    if r.get('reflections'):
        for i, ref in enumerate(r['reflections'], 1):
            print(f"  [reflection {i}] {ref[:200]}")
    print(f"--- trajectory ---\n{r['trajectory'][:1500]}")