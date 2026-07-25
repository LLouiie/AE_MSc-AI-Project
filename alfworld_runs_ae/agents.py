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


def _batch_first(value):
    """Extract the batch_size=1 element from ALFWorld's list/tuple-shaped
    env.step()/reset() return values; unwraps numpy scalar/0-d array via
    .item(); passes plain scalars through untouched. Deliberately does NOT
    cast type here — `reward` stays numeric for logging, `done`/`won` get
    cast to bool by the caller via _to_bool_scalar below. Never uses bare
    truthiness on the list/tuple itself (a non-empty `[False]` is a truthy
    *container* holding a False *value* — those are not the same thing)."""
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else None
    if hasattr(value, "item"):  # numpy scalar / 0-d array
        value = value.item()
    return value


def _to_bool_scalar(value) -> bool:
    return bool(_batch_first(value))


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
    """Single-trial ALFWorld agent using ReAct prompting.

    `controller` is optional and defaults to None, which reproduces the
    original ReAct-only behavior exactly (no extra signal extraction, no
    directive injection, no extra branching in the hot loop beyond a few
    `if self.controller:` checks that are no-ops when it's None). Passing
    an ae.controllers.stateful_controller.StatefulController switches this
    into the "ae_full" run mode. The controller never resets the
    environment and never appends anything to `history` — its directive is
    spliced only into the ephemeral prompt string for one LLM call.
    """

    def __init__(self, llm: AnyOpenAILLM = None, controller=None):
        self.llm = llm or AnyOpenAILLM(
            temperature=0, max_tokens=50, model_name=MODEL,
            model_kwargs={"stop": ['\n']},
            openai_api_key=API_KEY, openai_api_base=BASE_URL,
        )
        self.controller = controller

    def run(self, env, goal: str, task_type: str,
            reflections: List[str] = None, rules_text: str = "",
            to_print: bool = True) -> Tuple[str, bool]:
        """
        Run one trial.  Returns (trajectory_text, is_success).
        trajectory_text is suitable for Reflexion reflection and consolidation.
        """
        reflections = reflections or []
        base_prompt = _build_base_prompt(task_type, goal, reflections, rules_text)

        if self.controller:
            self.controller.reset()

        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        ob = process_ob(ob)
        admissible_before = info.get('admissible_commands', [[]])[0]

        history = []   # list of (action, observation) strings
        last_action = None
        is_exhausted = False
        termination_reason = "max_steps"

        if to_print:
            print(ob)

        for step in range(MAX_STEPS):
            trajectory_so_far = _build_trajectory(base_prompt, ob, history)
            prompt_for_llm = trajectory_so_far
            if self.controller:
                directive = self.controller.active_directive()
                if directive:
                    prompt_for_llm += f"\n\n{directive}\n"
            action = self.llm(prompt_for_llm + '\n> ').strip()

            # ---- stuck-loop ("exhausted") early-termination check --------
            # ORDERING FIX (this revision): previously this check ran BEFORE
            # env.step()/controller.step(), so the very first repeated
            # action after an intervention fired ended the episode without
            # ever giving the controller (or the environment) a chance to
            # react to the just-injected directive — a 3-step patch got, at
            # most, one real action's worth of effect. react mode
            # (controller=None) must behave byte-for-byte as before: grace
            # is never available when there's no controller, so this always
            # takes the "break before stepping" branch in that case.
            repeated_action_detected = (action == last_action)
            legacy_termination_candidate = "exhausted_repeated" if repeated_action_detected else None
            termination_suppressed = False
            suppression_reason = None

            if repeated_action_detected:
                grace_available = bool(self.controller) and self.controller.recovery_grace_remaining > 0
                if not grace_available:
                    is_exhausted = True
                    termination_reason = "exhausted_repeated"
                    if self.controller:
                        self.controller.note_final_exhausted()
                    break
                termination_suppressed = True
                suppression_reason = f"recovery_grace_remaining={self.controller.recovery_grace_remaining}"
                self.controller.consume_grace_step()
            last_action = action

            is_think_action = action.startswith('think:')
            obs_raw, reward_raw, done_raw, info = env.step([action])
            observation = process_ob(_batch_first(obs_raw))
            # SUCCESS BUGFIX (found 2026-07-25 while building the AE
            # controller, unified this revision): `done=True` does NOT mean
            # the task succeeded — ALFWorld's own base_config.yaml sets
            # dagger.training.max_nb_steps_per_episode: 50, equal to this
            # file's MAX_STEPS, so simply using up the step budget without
            # finishing ALSO produces done=True on the last step (confirmed
            # empirically: 50 steps of a harmless no-op 'look' ends with
            # done=True, info['won'][0]=False). `won` (info['won']) is the
            # ONLY field this file uses to decide success now — `reward`
            # and `done` are kept only as separate logged/diagnostic
            # values, never as a stand-in for `won`. All three go through
            # _to_bool_scalar/_batch_first rather than raw list truthiness,
            # since a non-empty `[False]` is a truthy container around a
            # False value.
            won = _to_bool_scalar(info.get('won', [False]))
            done = _to_bool_scalar(done_raw)
            reward = _batch_first(reward_raw)
            admissible_after = info.get('admissible_commands', [[]])[0]

            if self.controller:
                recent_actions = [a for a, _ in history]
                recent_observations = [o for _, o in history]
                self.controller.step(
                    action=action,
                    observation=observation,
                    admissible_before=admissible_before,
                    recent_actions=recent_actions,
                    recent_observations=recent_observations,
                    max_steps=MAX_STEPS,
                    is_think_action=is_think_action,
                    legacy_termination_candidate=legacy_termination_candidate,
                    termination_suppressed=termination_suppressed,
                    suppression_reason=suppression_reason,
                    reward=reward,
                    done=done,
                    won=won,
                )
            admissible_before = admissible_after

            if is_think_action:
                observation = 'OK.'

            history.append((action, observation))

            if to_print:
                print(f'> {action}\n{observation}')

            if done:
                trajectory_text = _build_trajectory(base_prompt, ob, history)
                termination_reason = "success" if won else "env_done_without_success"
                self.termination_reason = termination_reason
                if self.controller:
                    self.controller.episode_termination_reason = termination_reason
                return trajectory_text, bool(won)

        trajectory_text = _build_trajectory(base_prompt, ob, history)
        if is_exhausted:
            trajectory_text += f'\n{_EXHAUSTED_MSG}'
        self.termination_reason = termination_reason
        if self.controller:
            self.controller.episode_termination_reason = termination_reason
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
