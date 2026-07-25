# Adapted from ExpeL (Zhao et al. 2024), agent/expel.py and prompts/
import json, re

MAX_NUM_RULES = 20

_OP_RE = re.compile(r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)')

_EXTRACT_INSTRUCTION = (
    "You will be given successful tasks trials in which you were given access to a "
    "Docstore API environment and a question to answer."
)

_PROMPT_BODY = """{instruction}
Here are the trials:
{success_history}

Here are the EXISTING RULES:
{existing_rules}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:

<OPERATION> <RULE NUMBER>: <RULE>

The available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation.
"""

_SUFFIX_NOT_FULL = "Below are the operations you do to the above list of EXISTING RULES:\n"
_SUFFIX_FULL = (
    "Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful "
    "and different from EXISTING RULES. Below are the operations you do to the above list of "
    "EXISTING RULES:\n"
)


class RulePool:
    """ExpeL-style rule pool: each rule is {'text', 'count', 'born'}."""
    def __init__(self, path: str):
        self.path = path
        self.rules: list = []   # [{"text": str, "count": int, "born": int}]

    def _parse_ops(self, ops_text: str) -> dict:
        parsed = {op: [] for op in ['REMOVE', 'AGREE', 'EDIT', 'ADD']}
        for line in ops_text.splitlines():
            line = (line.strip()
                       .replace('UPVOTE', 'AGREE')
                       .replace('DOWNVOTE', 'REMOVE'))
            m = _OP_RE.match(line)
            if not m:
                continue
            op_str = m.group(1)       # e.g. "REMOVE 3" or "ADD 5"
            rule_text = m.group(2).strip()
            parts = op_str.split()
            op_type = parts[0]
            idx = int(parts[1]) - 1 if len(parts) > 1 else None  # 0-based
            parsed[op_type].append((idx, rule_text))
        return parsed

    def apply_ops(self, ops_text: str, q_index: int):
        parsed = self._parse_ops(ops_text)
        remove_strength = 3 if len(self.rules) >= MAX_NUM_RULES + 5 else 1

        for op in ['REMOVE', 'AGREE', 'EDIT', 'ADD']:
            for idx, text in parsed[op]:
                if op == 'REMOVE':
                    if idx is not None and 0 <= idx < len(self.rules):
                        self.rules[idx]['count'] -= remove_strength
                elif op == 'AGREE':
                    if idx is not None and 0 <= idx < len(self.rules):
                        self.rules[idx]['count'] += 1
                elif op == 'EDIT':
                    if idx is not None and 0 <= idx < len(self.rules):
                        self.rules[idx]['text'] = text
                        self.rules[idx]['count'] += 1
                elif op == 'ADD':
                    self.rules.append({'text': text, 'count': 2, 'born': q_index})

        self.rules = [r for r in self.rules if r['count'] > 0]
        self.rules.sort(key=lambda r: r['count'], reverse=True)

        with open(self.path, 'a') as f:
            f.write(json.dumps({'q_index': q_index, 'pool': self.rules},
                               ensure_ascii=False) + '\n')

    def render(self) -> str:
        if not self.rules:
            return ''
        return '\n'.join(f"{i+1}. {r['text']}" for i, r in enumerate(self.rules))

    def _build_prompt(self, trajectories: list) -> str:
        existing = self.render() or '(none)'
        suffix = _SUFFIX_FULL if len(self.rules) >= MAX_NUM_RULES else _SUFFIX_NOT_FULL
        body = _PROMPT_BODY.format(
            instruction=_EXTRACT_INSTRUCTION,
            success_history='\n---\n'.join(trajectories),
            existing_rules=existing,
        )
        return body + suffix


def consolidate(pool: RulePool, trajectories: list, llm, q_index: int):
    prompt = pool._build_prompt(trajectories)
    ops = llm(prompt)
    pool.apply_ops(ops, q_index)
