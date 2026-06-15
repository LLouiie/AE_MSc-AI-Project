import json
from agents import ReactReflectAgent
from environment import DistractorDocstore

# 拿一道真题构造 docstore(纯文件读取,不碰模型)
ex = json.load(open("/rds/general/user/jy625/home/projects/AE/data/hotpotqa/dev_full.json"))[0]
docstore = DistractorDocstore(titles=list(ex["context"]["title"]),
                              sentences=list(ex["context"]["sentences"]))

def build(rules_text):
    # react_llm/reflect_llm 传 None:只拼 prompt 不调用它们
    a = ReactReflectAgent(question=ex["question"], key=ex["answer"],
                          max_steps=6, docstore=docstore,
                          react_llm=None, reflect_llm=None,
                          rules_text=rules_text)
    return a._build_agent_prompt()

print("===== 空 rules =====")
p_empty = build("")
print(p_empty[-400:])   # 只看末尾, 看 rules 那行有没有消失

print("\n===== 有 rules =====")
p_rules = build("1. For comparison questions, the answer is often yes/no.\n2. Identify the bridge entity first.")
print(p_rules[-500:])
