"""Prompt templates as simple callables (no langchain dependency)."""


class PromptTemplate:
    """Minimal drop-in replacement for langchain.prompts.PromptTemplate."""
    def __init__(self, template: str, input_variables=None):
        self.template = template
        self.input_variables = input_variables or []

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


REACT_INSTRUCTION = """\
Determine if there is an observation that SUPPORTS or REFUTES a Claim, \
or if there is NOT ENOUGH INFORMATION.
Use Search[entity] to retrieve a Wikipedia article, \
Lookup[keyword] to find the next sentence containing the keyword, \
or Finish[label] to submit your answer (SUPPORTS / REFUTES / NOT ENOUGH INFO).

{rules}Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Claim: {question}{scratchpad}"""

REACT_REFLECT_INSTRUCTION = """\
Determine if there is an observation that SUPPORTS or REFUTES a Claim, \
or if there is NOT ENOUGH INFORMATION.
Use Search[entity] to retrieve a Wikipedia article, \
Lookup[keyword] to find the next sentence containing the keyword, \
or Finish[label] to submit your answer (SUPPORTS / REFUTES / NOT ENOUGH INFO).

{rules}Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Claim: {question}{scratchpad}"""

REFLECT_INSTRUCTION = """\
You are an advanced reasoning agent that can improve based on self reflection.
You were given a claim to verify using Wikipedia and failed either because you \
submitted the wrong label or did not gather sufficient evidence.
In a few sentences, diagnose the failure and propose a concrete plan to fix it.

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Claim: {question}{scratchpad}

Reflection:"""

react_agent_prompt = PromptTemplate(
    template=REACT_INSTRUCTION,
    input_variables=["rules", "examples", "reflections", "question", "scratchpad"],
)

react_reflect_agent_prompt = PromptTemplate(
    template=REACT_REFLECT_INSTRUCTION,
    input_variables=["rules", "examples", "reflections", "question", "scratchpad"],
)

reflect_prompt = PromptTemplate(
    template=REFLECT_INSTRUCTION,
    input_variables=["examples", "question", "scratchpad"],
)

REFLECTION_HEADER = "You have attempted to verify this claim before, and have the following reflections:\n"
RULES_HEADER = "You have accumulated the following verification strategies through experience:\n"
