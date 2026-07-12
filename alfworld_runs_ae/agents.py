import os, sys, json
from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hotpotqa_runs'))
from llm import AnyOpenAILLM

from environment import process_ob, get_task_type

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY  = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL    = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")

PROMPTS_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'alfworld_runs', 'prompts', 'alfworld_3prompts.json'
)
with open(PROMPTS_FILE) as f:
    _PROMPTS = json.load(f)

MAX_STEPS = 50
_EXHAUSTED_MSG = "(exhausted: repeated action)"


def _build_base_prompt(task_type: str, goal: str, reflections: List[str],
                       rules_text: str = "") -> str:
    """Build the static prefix shown before the interactive trajectory."""
    examples = _PROMPTS[f'react_{task_type}_1'] + _PROMPTS[f'react_{task_type}_0']
    prompt = f"Interact with a household to solve a task. Here are two examples.\n{examples}"

    if rules_text:
        prompt += f"\n\nYou have accumulated the following strategies through experience:\n{rules_text}\n"

    if reflections:
        prompt += "\n\nYour memory for the task below:"
        for i, r in enumerate(reflections[-3:]):   # keep at most 3 reflections
            prompt += f"\nTrial {i}:\n{r.strip()}"

    prompt += f"\nHere is the task:\n{goal}"
    return prompt


class ALFWorldAgent:
    """Single-trial ALFWorld agent using ReAct prompting."""

    def __init__(self, llm: AnyOpenAILLM = None):
        self.llm = llm or AnyOpenAILLM(
            temperature=0, max_tokens=50, model_name=MODEL,
            model_kwargs={"stop": ['\n']},
            openai_api_key=API_KEY, openai_api_base=BASE_URL,
        )

    def run(self, env, goal: str, task_type: str,
            reflections: List[str] = None, rules_text: str = "",
            to_print: bool = True) -> Tuple[str, bool]:
        """
        Run one trial.  Returns (trajectory_text, is_success).
        trajectory_text is suitable for Reflexion reflection and consolidation.
        """
        reflections = reflections or []
        base_prompt = _build_base_prompt(task_type, goal, reflections, rules_text)

        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        ob = process_ob(ob)

        history = []   # list of (action, observation) strings
        last_action = None
        is_exhausted = False

        if to_print:
            print(ob)

        for step in range(MAX_STEPS):
            trajectory_so_far = _build_trajectory(base_prompt, ob, history)
            action = self.llm(trajectory_so_far + '\n> ').strip()

            if action == last_action:
                is_exhausted = True
                break
            last_action = action

            observation, reward, done, info = env.step([action])
            observation = process_ob(observation[0])
            reward = info['won'][0]
            done = done[0]

            if action.startswith('think:'):
                observation = 'OK.'

            history.append((action, observation))

            if to_print:
                print(f'> {action}\n{observation}')

            if done:
                trajectory_text = _build_trajectory(base_prompt, ob, history)
                return trajectory_text, True

        trajectory_text = _build_trajectory(base_prompt, ob, history)
        if is_exhausted:
            trajectory_text += f'\n{_EXHAUSTED_MSG}'
        return trajectory_text, False


class ALFWorldReflectAgent:
    """Multi-trial agent with Reflexion and rule pool support."""

    def __init__(self, act_llm: AnyOpenAILLM = None, reflect_llm: AnyOpenAILLM = None):
        self.act_agent = ALFWorldAgent(act_llm)
        self.reflect_llm = reflect_llm or AnyOpenAILLM(
            temperature=0, max_tokens=300, model_name=MODEL,
            openai_api_key=API_KEY, openai_api_base=BASE_URL,
        )
        self.reflections: List[str] = []
        self.is_success: bool = False
        self.last_trajectory: str = ""

    def run_trial(self, env, goal: str, task_type: str,
                  rules_text: str = "", to_print: bool = True) -> bool:
        traj, success = self.act_agent.run(
            env, goal, task_type,
            reflections=self.reflections,
            rules_text=rules_text,
            to_print=to_print,
        )
        self.last_trajectory = traj
        self.is_success = success
        return success

    def reflect(self, goal: str, task_type: str) -> str:
        """Generate a reflection string from the last failed trajectory."""
        prompt = _build_reflect_prompt(goal, task_type, self.last_trajectory)
        reflection = self.reflect_llm(prompt).strip()
        self.reflections.append(self.last_trajectory + f'\nReflection: {reflection}')
        return reflection

    def reset(self):
        self.reflections = []
        self.is_success = False
        self.last_trajectory = ""


_REFLECT_INSTRUCTION = """\
You are an advanced reasoning agent that can improve based on self reflection.
You were given a household task to solve but failed. Review your last attempt trajectory and diagnose why it failed.
Write a short reflection (2-4 sentences) identifying what went wrong and what you should do differently next time.

Task: {goal}

Last attempt:
{trajectory}

Reflection:"""


def _build_reflect_prompt(goal: str, task_type: str, trajectory: str) -> str:
    return _REFLECT_INSTRUCTION.format(goal=goal, trajectory=trajectory[-3000:])


def _build_trajectory(base_prompt: str, initial_ob: str,
                      history: List[Tuple[str, str]]) -> str:
    s = base_prompt + '\n' + initial_ob
    for action, observation in history:
        s += f'\n> {action}\n{observation}'
    return s
