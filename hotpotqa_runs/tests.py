import joblib
from environment import DistractorDocstore
from agents import ReactReflectAgent

data = joblib.load('data/hotpot-qa-distractor-sample.joblib')

correct = 0
total = 10

for i in range(total):
    row = data.iloc[i]
    print(f"\n{'='*60}")
    print(f"Question {i+1}: {row['question']}")
    print(f"Gold Answer: {row['answer']}")

    docstore = DistractorDocstore(
        titles=list(row['context']['title']),
        sentences=list(row['context']['sentences'])
    )

    agent = ReactReflectAgent(
        question=row['question'],
        key=row['answer'],
        docstore=docstore
    )

    # 最多 3 轮 trial
    for trial in range(3):
        agent.run()
        if agent.is_correct():
            break

    print(f"Model Answer: {agent.answer}")
    print(f"Correct: {agent.is_correct()}")
    if agent.is_correct():
        correct += 1

print(f"\n{'='*60}")
print(f"Result: {correct}/{total} = {correct/total*100:.0f}%")