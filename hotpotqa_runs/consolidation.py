import json, re

POOL_CAP = 20

EXTRACT_PROMPT = """(占位 — 明天按 ExpeL Appendix 原文精修, 含引用)
You are reviewing an agent's recent question-answering trajectories...
Existing rules:
{rules}
Trajectories:
{trajectories}
Output operations, one per line:
ADD: <new rule> | EDIT <n>: <revised rule> | UPVOTE <n> | DOWNVOTE <n>
"""


class RulePool:
    """ExpeL 式 rule 池: 每条带分数, UPVOTE+1 / DOWNVOTE-1, 归零删除, 容量上限。"""
    def __init__(self, path):
        self.path = path           # rules.jsonl, 追加式留痕
        self.rules = []            # [{"text": str, "score": int, "born": int}]

    def apply_ops(self, ops_text: str, q_index: int):
        for line in ops_text.splitlines():
            line = line.strip()
            if line.startswith("ADD:") and len(self.rules) < POOL_CAP:
                self.rules.append({"text": line[4:].strip(), "score": 2, "born": q_index})
            elif m := re.match(r"EDIT (\d+):\s*(.+)", line):
                i = int(m.group(1)) - 1
                if 0 <= i < len(self.rules):
                    self.rules[i]["text"] = m.group(2).strip()
            elif m := re.match(r"UPVOTE (\d+)", line):
                i = int(m.group(1)) - 1
                if 0 <= i < len(self.rules):
                    self.rules[i]["score"] += 1
            elif m := re.match(r"DOWNVOTE (\d+)", line):
                i = int(m.group(1)) - 1
                if 0 <= i < len(self.rules):
                    self.rules[i]["score"] -= 1
        self.rules = [r for r in self.rules if r["score"] > 0]
        with open(self.path, "a") as f:
            f.write(json.dumps({"q_index": q_index,
                                "pool": self.rules}, ensure_ascii=False) + "\n")

    def render(self) -> str:
        """注入 prompt 用的文本形态。"""
        if not self.rules:
            return ""
        return "\n".join(f"{i+1}. {r['text']}" for i, r in enumerate(self.rules))


def consolidate(pool: RulePool, trajectories: list, llm, q_index: int):
    """巩固入口: 所有方法共用, 唯一被调度器控制的就是何时调用它。"""
    prompt = EXTRACT_PROMPT.format(
        rules=pool.render() or "(none)",
        trajectories="\n---\n".join(trajectories))
    ops = llm(prompt)
    pool.apply_ops(ops, q_index)