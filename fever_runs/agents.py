import re, os, sys
from typing import List
import tiktoken

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hotpotqa_runs'))
from llm import AnyOpenAILLM

from prompts import (
    react_agent_prompt, react_reflect_agent_prompt, reflect_prompt,
    REFLECTION_HEADER, RULES_HEADER,
)
from fewshots import FEVER_WEBTHINK, FEVER_REFLECTIONS
from environment import normalize_label

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY  = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL    = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")


class FEVERReactAgent:
    def __init__(
        self,
        claim: str,
        key: str,
        docstore,
        max_steps: int = 7,
        agent_prompt=react_agent_prompt,
        react_llm: AnyOpenAILLM = None,
        rules_text: str = "",
    ):
        self.question = claim
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = FEVER_WEBTHINK
        self.rules_text = rules_text
        self.docstore = docstore
        self.llm = react_llm or AnyOpenAILLM(
            temperature=0, max_tokens=100, model_name=MODEL,
            model_kwargs={"stop": "\n"},
            openai_api_key=API_KEY, openai_api_base=BASE_URL,
        )
        self.enc = tiktoken.encoding_for_model("text-davinci-003")
        self._reset()

    def run(self, reset=True) -> None:
        if reset:
            self._reset()
        while not self.is_halted() and not self.is_finished():
            self.step()

    def step(self) -> None:
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self._prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nAction {self.step_n}:'
        action = self._prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = _parse_action(action)
        print(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nObservation {self.step_n}: '
        if action_type == 'Finish':
            self.answer = argument
            self.scratchpad += 'Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Search':
            try:
                self.scratchpad += _fmt(self.docstore.search(argument))
            except Exception as e:
                self.scratchpad += f'Could not find that page, please try again.'

        elif action_type == 'Lookup':
            try:
                self.scratchpad += _fmt(self.docstore.lookup(argument))
            except Exception as e:
                self.scratchpad += str(e)

        else:
            self.scratchpad += ('Invalid Action. Valid Actions are Search[<entity>], '
                                'Lookup[<keyword>], Finish[SUPPORTS], '
                                'Finish[REFUTES], Finish[NOT ENOUGH INFO].')

        print(self.scratchpad.split('\n')[-1])
        self.step_n += 1

    def _prompt_agent(self) -> str:
        return _fmt(self.llm(self._build_prompt()))

    def _build_prompt(self) -> str:
        return self.agent_prompt.format(
            rules=self._fmt_rules(),
            examples=self.react_examples,
            reflections=getattr(self, 'reflections_str', ''),
            question=self.question,
            scratchpad=self.scratchpad,
        )

    def _fmt_rules(self) -> str:
        if not self.rules_text:
            return ''
        return RULES_HEADER + self.rules_text + '\n'

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return normalize_label(self.answer) == normalize_label(self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or
                (len(self.enc.encode(self._build_prompt())) > 3896)) and not self.finished

    def _reset(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad = ''


class FEVERReactReflectAgent(FEVERReactAgent):
    def __init__(self, claim, key, docstore, max_steps=7,
                 agent_prompt=react_reflect_agent_prompt,
                 reflect_prompt_tmpl=reflect_prompt,
                 react_llm=None, reflect_llm=None, rules_text=""):
        super().__init__(claim, key, docstore, max_steps, agent_prompt, react_llm, rules_text)
        self.reflect_prompt = reflect_prompt_tmpl
        self.reflect_examples = FEVER_REFLECTIONS
        self.reflect_llm = reflect_llm or AnyOpenAILLM(
            temperature=0, max_tokens=250, model_name=MODEL,
            openai_api_key=API_KEY, openai_api_base=BASE_URL,
        )
        self.reflections: List[str] = []
        self.reflections_str: str = ''

    def run(self, reset=True, reflect_strategy='reflexion') -> None:
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            self.reflect(reflect_strategy)
        FEVERReactAgent.run(self, reset)

    def reflect(self, strategy: str = 'reflexion') -> None:
        print('Reflecting...')
        if strategy == 'last_trial':
            self.reflections = [self.scratchpad]
            self.reflections_str = _fmt_last_attempt(self.question, self.reflections[0])
        elif strategy == 'reflexion':
            self.reflections.append(self._prompt_reflection())
            self.reflections_str = _fmt_reflections(self.reflections)
        else:
            raise ValueError(f"Unknown reflect_strategy: {strategy}")
        print(self.reflections_str)

    def _prompt_reflection(self) -> str:
        return _fmt(self.reflect_llm(self._build_reflect_prompt()))

    def _build_reflect_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            question=self.question,
            scratchpad=_truncate(self.scratchpad, self.enc),
        )


def _parse_action(action: str):
    import re
    m = re.match(r'^(\w[\w\s]*)\[(.+)\]$', action.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _fmt(s: str) -> str:
    return s.strip('\n').strip()


def _fmt_reflections(reflections: List[str]) -> str:
    if not reflections:
        return ''
    header = REFLECTION_HEADER
    return header + 'Reflection: '.join([r + '\n' for r in reflections])


def _fmt_last_attempt(question: str, scratchpad: str) -> str:
    return (f'You have attempted to verify this claim before:\n'
            f'Claim: {question}\n{scratchpad}\n(END PREVIOUS TRIAL)\n')


def _truncate(scratchpad: str, enc, max_tokens: int = 1600) -> str:
    lines = scratchpad.split('\n')
    while len(enc.encode('\n'.join(lines))) > max_tokens and lines:
        lines.pop(0)
    return '\n'.join(lines)
