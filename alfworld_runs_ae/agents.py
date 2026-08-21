import os, sys, json
from typing import List, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hotpotqa_runs'))
from llm import AnyOpenAILLM
from openai import BadRequestError

from environment import process_ob, get_task_type
from output_parser import parse_agent_output
from alfworld_action_normalize import normalize_alfworld_action
from demo_config import DemoConfig, DEFAULT_DEMO_CONFIG, demo_count_phrase

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY  = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL    = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")

PROMPTS_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'alfworld_runs', 'prompts', 'alfworld_3prompts.json'
)
with open(PROMPTS_FILE) as f:
    _PROMPTS = json.load(f)

MAX_STEPS = 50

# Generation-path config (shared by react/reflexion/ae_full -- fixing the
# max_tokens=50 truncation bug here fixes it for every baseline that uses
# ALFWorldAgent, not just AE). Single source of truth: run_alfworld.py's
# build_llms() imports these same constants rather than hardcoding its own.
ACT_MAX_TOKENS = int(os.getenv("ALFWORLD_ACT_MAX_TOKENS", "256"))
# Empirically verified against the live Qwen3-8B endpoint (2026-07-25):
# stop=["\n"] (the old setting) hard-truncates generation after exactly one
# line, which cuts a "Thought: ...\nAction: ..." response off after the
# Thought line every time. Replaying history in the SAME Thought/Action/
# Observation shape it's asked to generate in (see _build_trajectory) gives
# the model a strong, local pattern to continue -- stopping right after the
# Action line and letting our own code supply "Observation: ..." next.
# First attempt without replaying history in this shape (only stating the
# instruction once, keeping the old "> action\nobservation" replay format)
# reliably reverted to the old style after turn 1 and ran to max_tokens
# hallucinating several further turns in one completion; matching the
# replay format to the instruction fixed this (confirmed empirically:
# finish_reason="stop", one clean Thought/Action pair, across 3+ turns).
ACT_STOP = ["\nObservation:", "\nThought:"]

DEFAULT_TERMINATION_POLICY = os.getenv("ALFWORLD_TERMINATION_POLICY", "fixed_horizon")
_TERMINATION_POLICIES = {"fixed_horizon", "legacy_early_stop"}

_EXHAUSTED_MSG = "(exhausted: repeated action)"
_PARSE_FAILURE_OBSERVATION = "Nothing happens."

# ---- shared context-length budget (react/reflexion/ae_full alike) --------
# One 40960 context budget for every method -- fixed 2026-07-27 after
# Reflexion x ALFWorld exam task 3 (pick_two_obj_and_place-SoapBar-None-
# GarbageCan-424) crashed vLLM with a 400 BadRequestError (prompt >40705
# tokens). Root cause was NOT "no last-3 reflection cap" (the action prompt
# already did reflections[-3:]) -- it was that ALFWorldReflectAgent stored
# self.last_trajectory + reflection (an entire ~50-step trial) as each
# "reflection" memory entry, instead of just the short reflection text the
# original alfworld_runs/generate_reflections.py::update_memory stores
# (env_configs[i]['memory'] += [reflection]). Three of those bloated
# entries stacked in the action prompt is what actually blew the budget.
# Fixed at the source (see ALFWorldReflectAgent.reflect()) -- the budget/
# truncation machinery below is a deliberate defense-in-depth layer, not
# the primary fix, per explicit spec: never raise vLLM's own max_model_len
# per-method, keep one shared 40960 budget for every baseline.
MAX_MODEL_LEN = int(os.getenv("ALFWORLD_MAX_MODEL_LEN", "40960"))
PROMPT_SAFETY_MARGIN = int(os.getenv("ALFWORLD_PROMPT_SAFETY_MARGIN", "512"))
REFLECT_MAX_TOKENS = int(os.getenv("ALFWORLD_REFLECT_MAX_TOKENS", "256"))

_TOKENIZERS: dict = {}


def _get_tokenizer(model_name: str):
    tok = _TOKENIZERS.get(model_name)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZERS[model_name] = tok
    return tok


def count_tokens(text: str, model_name: str) -> int:
    """Exact input token count via the model's own tokenizer -- not an
    estimate -- so the truncation loops below can check the real budget
    before every request, not guess from character counts."""
    return len(_get_tokenizer(model_name).encode(text))


class PromptBudgetExceededError(RuntimeError):
    """Raised when a prompt still exceeds the shared context budget after
    deterministic truncation AND one retry with a tighter budget. Callers
    (ae/runners/run_alfworld.py) catch this to mark the single task
    incomplete and move on -- this must never propagate into a bare
    sys.exit or crash the rest of the split/combo."""

    def __init__(self, message: str, tokens_before: int, tokens_after: int):
        super().__init__(message)
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after

_FORMAT_INSTRUCTION = (
    "\nRespond with exactly two lines in this format:\n"
    "Thought: <one concise reasoning step>\n"
    "Action: <one environment command>\n"
)


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
                       rules_text: str = "", demo_config: DemoConfig = None) -> str:
    """Build the static prefix shown before the interactive trajectory.

    `demo_config` selects which/how many alfworld_3prompts.json ICL
    examples to concatenate (see demo_config.py); defaults to
    DEFAULT_DEMO_CONFIG, which reproduces the original hardcoded two-shot
    prompt byte-for-byte."""
    demo_config = demo_config or DEFAULT_DEMO_CONFIG
    try:
        examples = "".join(
            _PROMPTS[f'react_{task_type}_{idx}'] for idx in demo_config.demo_indices
        )
    except KeyError as e:
        raise KeyError(
            f"demo_config references {e} for task_type={task_type!r}, but no such "
            f"ICL example exists in alfworld_3prompts.json"
        ) from e
    prompt = f"Interact with a household to solve a task. {demo_count_phrase(demo_config.num_demos)}\n{examples}"

    if rules_text:
        prompt += f"\n\nYou have accumulated the following strategies through experience:\n{rules_text}\n"

    if reflections:
        prompt += "\n\nYour memory for the task below:"
        for i, r in enumerate(reflections[-3:]):   # keep at most 3 reflections
            prompt += f"\nTrial {i}:\n{r.strip()}"

    prompt += f"\nHere is the task:\n{goal}\n{_FORMAT_INSTRUCTION}"
    return prompt


def _fit_action_prompt(
    task_type: str, goal: str, reflections: List[str], rules_text: str,
    initial_ob: str, history: List[Tuple[Optional[str], str, str]],
    model_name: str, max_new_tokens: int, task_id: Optional[str],
    extra_margin: int = 0, demo_config: DemoConfig = None,
) -> Tuple[str, dict]:
    """Deterministically assemble the action-generation prompt so it fits
    within MAX_MODEL_LEN - max_new_tokens - PROMPT_SAFETY_MARGIN tokens
    (minus `extra_margin`, used by the one-retry-with-a-tighter-budget path
    in ALFWorldAgent.run()). Priority order, per spec:
      1. drop the OLDEST of the (already-capped-at-3) reflections first,
         keeping the newest ones;
      2. if still over budget with zero reflections, drop the oldest
         history steps one at a time, always keeping the task description
         (in base_prompt) + the initial observation + the most recent
         action/observation steps.
    Returns (prompt_text, trunc_info) where trunc_info records tokens
    before/after and which categories were dropped, for logging.
    """
    budget = MAX_MODEL_LEN - max_new_tokens - PROMPT_SAFETY_MARGIN - extra_margin
    capped_reflections = reflections[-3:] if reflections else []
    dropped: List[str] = []

    tokens_before = None
    text = None
    n_used = len(capped_reflections)
    for n_used in range(len(capped_reflections), -1, -1):
        base_prompt = _build_base_prompt(task_type, goal, capped_reflections[-n_used:] if n_used else [], rules_text, demo_config)
        text = _build_trajectory(base_prompt, initial_ob, history)
        tokens = count_tokens(text, model_name)
        if tokens_before is None:
            tokens_before = tokens
        if tokens <= budget:
            if n_used < len(capped_reflections):
                dropped.append(f"dropped {len(capped_reflections) - n_used} oldest reflection(s)")
            return text, {
                "tokens_before": tokens_before, "tokens_after": tokens,
                "dropped": dropped, "task_id": task_id,
            }

    # Still over budget with zero reflections -- fall back to dropping the
    # oldest history (action/observation) steps, keeping the task
    # description + initial observation + most recent steps.
    hist = list(history)
    base_prompt = _build_base_prompt(task_type, goal, [], rules_text, demo_config)
    n_dropped_steps = 0
    while len(hist) > 0:
        hist.pop(0)  # drop oldest
        n_dropped_steps += 1
        text = _build_trajectory(base_prompt, initial_ob, hist)
        tokens = count_tokens(text, model_name)
        if tokens <= budget:
            dropped.append(f"dropped {n_dropped_steps} oldest history step(s)")
            return text, {
                "tokens_before": tokens_before, "tokens_after": tokens,
                "dropped": dropped, "task_id": task_id,
            }

    # Irreducible: even task description + initial observation + zero
    # history/reflections doesn't fit. Return best-effort text; the caller
    # (the LLM call site) is responsible for treating this as unfixable.
    tokens = count_tokens(text, model_name)
    dropped.append(f"dropped all {n_dropped_steps} history step(s); still over budget")
    return text, {
        "tokens_before": tokens_before, "tokens_after": tokens,
        "dropped": dropped, "task_id": task_id,
    }


def summarize_generation_log(step_log: List[dict]) -> dict:
    """Generation-path episode summary, computed uniformly from
    ALFWorldAgent.step_log for any baseline (react, reflexion, ae_full).
    Token counts are reported as "unavailable" rather than estimated if
    the endpoint never returned usage for any step in the episode."""
    truncation_count = sum(1 for s in step_log if s.get("was_truncated"))
    parse_failure_count = sum(1 for s in step_log if not s.get("parse_success", True))
    admissible_checkable = [s for s in step_log if s.get("action_is_admissible") is not None]
    valid_action_count = sum(1 for s in admissible_checkable if s["action_is_admissible"])
    invalid_action_count = sum(1 for s in admissible_checkable if not s["action_is_admissible"])
    admissible_action_rate = (
        valid_action_count / len(admissible_checkable) if admissible_checkable else None
    )

    usages = [s.get("usage") for s in step_log]
    if any(u == "unavailable" or u is None for u in usages):
        prompt_tokens = completion_tokens = total_tokens = "unavailable"
    else:
        prompt_tokens = sum(u["prompt_tokens"] for u in usages)
        completion_tokens = sum(u["completion_tokens"] for u in usages)
        total_tokens = sum(u["total_tokens"] for u in usages)

    return {
        "truncation_count": truncation_count,
        "parse_failure_count": parse_failure_count,
        "valid_action_count": valid_action_count,
        "invalid_action_count": invalid_action_count,
        "admissible_action_rate": admissible_action_rate,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


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

    `termination_policy`:
      - "fixed_horizon" (default): only won=True, an environment terminal,
        or MAX_STEPS end the episode. Repeated actions/observations and
        parse failures are recorded but never end the episode early —
        this is the policy react and ae_full both run the main pilot
        under, so neither baseline gets a recovery mechanism the other
        lacks.
      - "legacy_early_stop": the original exact-repeated-action early
        termination (with AE's recovery-grace suppression window, when a
        controller is attached). Kept for comparison, not deleted.
    """

    def __init__(self, llm: AnyOpenAILLM = None, controller=None,
                 termination_policy: str = None, demo_config: DemoConfig = None):
        self.llm = llm or AnyOpenAILLM(
            temperature=0, max_tokens=ACT_MAX_TOKENS, model_name=MODEL,
            model_kwargs={"stop": ACT_STOP},
            openai_api_key=API_KEY, openai_api_base=BASE_URL,
        )
        self.controller = controller
        self.termination_policy = termination_policy or DEFAULT_TERMINATION_POLICY
        if self.termination_policy not in _TERMINATION_POLICIES:
            raise ValueError(f"unknown termination_policy: {self.termination_policy!r}")
        self.demo_config = demo_config or DEFAULT_DEMO_CONFIG

    def run(self, env, goal: str, task_type: str,
            reflections: List[str] = None, rules_text: str = "",
            to_print: bool = True, task_id: Optional[str] = None) -> Tuple[str, bool]:
        """
        Run one trial.  Returns (trajectory_text, is_success).
        trajectory_text is suitable for Reflexion reflection and consolidation.
        """
        reflections = reflections or []
        base_prompt = _build_base_prompt(task_type, goal, reflections, rules_text, self.demo_config)
        fixed_horizon = self.termination_policy == "fixed_horizon"

        if self.controller:
            self.controller.reset()

        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        ob = process_ob(ob)
        admissible_before = info.get('admissible_commands', [[]])[0]

        history = []   # list of (action_text, observation) strings
        self.step_log: List[dict] = []
        last_action = None
        is_exhausted = False
        termination_reason = "max_steps"

        if to_print:
            print(ob)

        for step in range(MAX_STEPS):
            prompt_for_llm, trunc_info = _fit_action_prompt(
                task_type, goal, reflections, rules_text, ob, history,
                self.llm.model, ACT_MAX_TOKENS, task_id,
                demo_config=self.demo_config,
            )
            if self.controller:
                directive = self.controller.active_directive()
                if directive:
                    prompt_for_llm += f"\n\n{directive}\n"
            try:
                raw_generation = self.llm(prompt_for_llm + '\n')
            except BadRequestError as e:
                if "maximum context length" not in str(e):
                    raise
                # One retry with a much tighter budget before giving up --
                # never sys.exit, the caller decides what "give up" means.
                retry_prompt, retry_trunc_info = _fit_action_prompt(
                    task_type, goal, reflections, rules_text, ob, history,
                    self.llm.model, ACT_MAX_TOKENS, task_id,
                    extra_margin=PROMPT_SAFETY_MARGIN, demo_config=self.demo_config,
                )
                if self.controller:
                    directive = self.controller.active_directive()
                    if directive:
                        retry_prompt += f"\n\n{directive}\n"
                try:
                    raw_generation = self.llm(retry_prompt + '\n')
                    prompt_for_llm, trunc_info = retry_prompt, retry_trunc_info
                except BadRequestError as e2:
                    if "maximum context length" not in str(e2):
                        raise
                    raise PromptBudgetExceededError(
                        f"task_id={task_id} step={step}: prompt still exceeds context "
                        f"budget after deterministic truncation and one retry: {e2}",
                        tokens_before=trunc_info["tokens_before"],
                        tokens_after=retry_trunc_info["tokens_after"],
                    ) from e2
            meta = getattr(self.llm, "last_meta", None) or {}
            finish_reason = meta.get("finish_reason")
            was_truncated = bool(meta.get("was_truncated"))
            usage = meta.get("usage", "unavailable")

            parsed = parse_agent_output(raw_generation, admissible_before)
            parsed_action = parsed.parsed_action
            is_think_action = bool(parsed_action) and parsed_action.startswith('think:')

            # ---- ALFWorld action-protocol compatibility adapter -----------
            # parsed_action is the model's own wording (e.g. "put mug 1 in
            # fridge 1"), unchanged from the parser above and still what
            # `history`/the repeated-action termination check (below) see.
            # executed_action is the ONLY thing env.step() ever receives --
            # a pure string rewrite (currently just put->move; see
            # alfworld_action_normalize.py) applied at the environment
            # boundary, after parsing and before execution, never touching
            # raw_generation/Thought/history.
            executed_action, action_normalization = normalize_alfworld_action(parsed_action)

            # ---- stuck-loop ("exhausted") early-termination check --------
            # Only meaningful under termination_policy="legacy_early_stop".
            # Under "fixed_horizon" (the main-experiment default) an exact
            # repeat is still recorded in the log but never ends the
            # episode early -- react and ae_full get the exact same
            # horizon, so AE's recovery grace is not a termination
            # advantage the baseline lacks.
            repeated_action_detected = (
                parsed_action is not None and last_action is not None
                and parsed_action == last_action
            )
            legacy_termination_candidate = "exhausted_repeated" if repeated_action_detected else None
            termination_suppressed = False
            suppression_reason = None

            if repeated_action_detected and not fixed_horizon:
                grace_available = bool(self.controller) and self.controller.recovery_grace_remaining > 0
                if grace_available:
                    termination_suppressed = True
                    suppression_reason = f"recovery_grace_remaining={self.controller.recovery_grace_remaining}"
                    self.controller.consume_grace_step()
                else:
                    # AE-full only (self.controller is not None): grace can
                    # only ever be granted by _arm_intervention(), and
                    # interventions are unconditionally suppressed while
                    # the controller is still in its warmup window -- so an
                    # exact repeat landing inside (or at the boundary of)
                    # warmup would otherwise kill the episode at
                    # intervention_count=0, before the controller has ever
                    # had a single non-suppressed step to react (confirmed
                    # empirically, e.g. idx=106
                    # pick_heat_then_place_in_recep-Egg-...-GarbageCan-10).
                    # `<=` (not `<`) deliberately covers one extra step past
                    # the controller's own in_warmup boundary (step_index
                    # <= warmup_steps) so THIS iteration's upcoming
                    # controller.step() call -- the first call that is
                    # itself no longer in_warmup -- still gets to run
                    # instead of being pre-empted by this check. This is a
                    # one-time, bounded (<=warmup_steps+1 iterations total)
                    # window, not a recurring grace refresh: ReAct/Reflexion
                    # (self.controller is None) and any repeat past this
                    # window are completely unaffected.
                    controller_in_warmup = (
                        self.controller is not None
                        and self.controller.step_index <= self.controller.config.warmup_steps
                    )
                    if controller_in_warmup:
                        termination_suppressed = True
                        suppression_reason = (
                            f"warmup_active (completed_steps={self.controller.step_index}"
                            f"<=warmup_steps={self.controller.config.warmup_steps})"
                        )
                        self.controller.note_warmup_suppressed()
                    else:
                        is_exhausted = True
                        termination_reason = "exhausted_repeated"
                        if self.controller:
                            self.controller.note_final_exhausted()
                        break
            last_action = parsed_action

            # ---- environment step ------------------------------------
            # The environment only ever receives executed_action (parsed,
            # then put->move normalized), never raw_generation and never
            # the un-normalized parsed_action. When no action could be
            # extracted at all, env.step() is skipped entirely and a
            # synthesized no-op observation is used instead -- this still
            # consumes one step of the budget/history (a "normal failed
            # interaction", per spec) without ever handing the environment
            # text it never asked for.
            if executed_action is not None:
                obs_raw, reward_raw, done_raw, info = env.step([executed_action])
                observation = process_ob(_batch_first(obs_raw))
                won = _to_bool_scalar(info.get('won', [False]))
                done = _to_bool_scalar(done_raw)
                reward = _batch_first(reward_raw)
                admissible_after = info.get('admissible_commands', [[]])[0]
                action_is_admissible = (
                    None if is_think_action else executed_action.strip() in admissible_before
                )
            else:
                observation = _PARSE_FAILURE_OBSERVATION
                won = False
                done = False
                reward = 0.0
                admissible_after = admissible_before
                action_is_admissible = None

            step_record = {
                **parsed.as_dict(),
                "executed_action": executed_action,
                "action_normalized": action_normalization is not None,
                "action_normalization": action_normalization,
                "finish_reason": finish_reason,
                "was_truncated": was_truncated,
                "action_is_admissible": action_is_admissible,
                "usage": usage,
                "prompt_tokens_before_truncation": trunc_info["tokens_before"],
                "prompt_tokens_after_truncation": trunc_info["tokens_after"],
                "prompt_truncation_actions": trunc_info["dropped"],
            }
            self.step_log.append(step_record)

            if self.controller:
                recent_actions = [a for _, a, _ in history]
                recent_observations = [o for _, _, o in history]
                self.controller.step(
                    action=executed_action if executed_action is not None else "",
                    observation=observation,
                    admissible_before=admissible_before,
                    recent_actions=recent_actions,
                    recent_observations=recent_observations,
                    max_steps=MAX_STEPS,
                    is_think_action=is_think_action,
                    admissible_after=admissible_after,
                    legacy_termination_candidate=legacy_termination_candidate,
                    termination_suppressed=termination_suppressed,
                    suppression_reason=suppression_reason,
                    reward=reward,
                    done=done,
                    won=won,
                )
            admissible_before = admissible_after

            display_observation = 'OK.' if is_think_action else observation
            history_action_text = parsed_action if parsed_action is not None else raw_generation.strip()[:200]
            history.append((parsed.parsed_thought, history_action_text, display_observation))

            if to_print:
                print(f'Action: {history_action_text}\nObservation: {display_observation}')

            if done:
                trajectory_text = _build_trajectory(base_prompt, ob, history)
                termination_reason = "success" if won else "env_done_without_success"
                self.termination_reason = termination_reason
                self.last_ob = ob
                self.last_history = history
                if self.controller:
                    self.controller.episode_termination_reason = termination_reason
                return trajectory_text, bool(won)

            # ---- post-intervention exact-repeat forced termination -------
            # Only reached when the environment didn't already end the
            # episode this step (env_success/env_done above always takes
            # priority). Higher priority than legacy repeated-action
            # termination, recovery_grace, and cooldown: the controller's
            # own step() already bypassed all of those internally the
            # moment it detected this condition (see stateful_controller.py).
            if self.controller and self.controller.force_terminate_reason:
                is_exhausted = True
                termination_reason = self.controller.force_terminate_reason
                break

        trajectory_text = _build_trajectory(base_prompt, ob, history)
        if is_exhausted:
            trajectory_text += f'\n{_EXHAUSTED_MSG}'
        self.termination_reason = termination_reason
        self.last_ob = ob
        self.last_history = history
        if self.controller:
            self.controller.episode_termination_reason = termination_reason
        return trajectory_text, False


class ALFWorldReflectAgent:
    """Multi-trial agent using Reflexion and rule pool support."""

    def __init__(self, act_llm: AnyOpenAILLM = None, reflect_llm: AnyOpenAILLM = None,
                 termination_policy: str = None, demo_config: DemoConfig = None):
        self.act_agent = ALFWorldAgent(act_llm, termination_policy=termination_policy,
                                        demo_config=demo_config)
        self.reflect_llm = reflect_llm or AnyOpenAILLM(
            temperature=0, max_tokens=REFLECT_MAX_TOKENS, model_name=MODEL,
            openai_api_key=API_KEY, openai_api_base=BASE_URL,
        )
        self.reflections: List[str] = []
        self.is_success: bool = False
        self.last_trajectory: str = ""
        self.last_ob: str = ""
        self.last_history: List[Tuple[Optional[str], str, str]] = []
        self.last_reflect_trunc_info: Optional[dict] = None

    def run_trial(self, env, goal: str, task_type: str,
                  rules_text: str = "", to_print: bool = True,
                  task_id: Optional[str] = None) -> bool:
        traj, success = self.act_agent.run(
            env, goal, task_type,
            reflections=self.reflections,
            rules_text=rules_text,
            to_print=to_print,
            task_id=task_id,
        )
        self.last_trajectory = traj
        self.last_ob = self.act_agent.last_ob
        self.last_history = self.act_agent.last_history
        self.is_success = success
        return success

    def reflect(self, goal: str, task_type: str, task_id: Optional[str] = None) -> str:
        """Generate a reflection string from the last failed trajectory.

        Bug fixed 2026-07-27 (see MAX_MODEL_LEN docstring above): this used
        to store `self.last_trajectory + reflection` -- the ENTIRE ~50-step
        failed trial, not just the reflection -- as the memory entry fed
        back into the next trial's action prompt. The original Reflexion
        implementation (alfworld_runs/generate_reflections.py::
        update_memory: `env_configs[i]['memory'] += [reflection]`) only
        ever stores the short reflection text itself. Three of the bloated
        entries stacked in _build_base_prompt's reflections[-3:] is what
        actually blew the 40960-token budget on longer exam-split trials --
        capping the COUNT at 3 (already correct) never helped when each
        entry itself was a full trajectory."""
        reflect_model = self.reflect_llm.model
        reflect_budget = MAX_MODEL_LEN - REFLECT_MAX_TOKENS - PROMPT_SAFETY_MARGIN
        prompt, trunc_info = _fit_reflect_prompt(
            goal, task_type, self.last_ob, self.last_history, reflect_model, reflect_budget, task_id,
        )
        try:
            reflection_raw = self.reflect_llm(prompt)
        except BadRequestError as e:
            if "maximum context length" not in str(e):
                raise
            prompt, trunc_info = _fit_reflect_prompt(
                goal, task_type, self.last_ob, self.last_history, reflect_model,
                reflect_budget - PROMPT_SAFETY_MARGIN, task_id,
            )
            try:
                reflection_raw = self.reflect_llm(prompt)
            except BadRequestError as e2:
                if "maximum context length" not in str(e2):
                    raise
                raise PromptBudgetExceededError(
                    f"task_id={task_id}: reflection prompt still exceeds context budget "
                    f"after truncation and one retry: {e2}",
                    tokens_before=trunc_info["tokens_before"],
                    tokens_after=trunc_info["tokens_after"],
                ) from e2
        self.last_reflect_trunc_info = trunc_info
        reflection = reflection_raw.strip()
        self.reflections.append(reflection)
        return reflection

    def reset(self):
        self.reflections = []
        self.is_success = False
        self.last_trajectory = ""
        self.last_reflect_trunc_info = None


_REFLECT_INSTRUCTION = """\
You are an advanced reasoning agent that can improve based on self reflection.
You were given a household task to solve but failed. Review your last attempt trajectory and diagnose why it failed.
Write a short reflection (2-4 sentences) identifying what went wrong and what you should do differently next time.

Task: {goal}

Last attempt:
{trajectory}

Reflection:"""


def _fit_reflect_prompt(
    goal: str, task_type: str, initial_ob: str,
    history: List[Tuple[Optional[str], str, str]],
    model_name: str, budget: int, task_id: Optional[str],
) -> Tuple[str, dict]:
    """Deterministically assemble the reflection-generation prompt so it
    fits within `budget` tokens. Replaces the old char-count slice
    (trajectory[-3000:]) with exact tokenizer-based accounting; same
    drop-oldest-step priority as _fit_action_prompt's history fallback --
    task description (`goal`, always in the template) + initial
    observation are always kept, oldest action/observation steps drop
    first."""
    dropped: List[str] = []
    hist = list(history)
    tokens_before = None
    while True:
        trajectory_text = _build_trajectory("", initial_ob, hist).lstrip("\n")
        prompt = _REFLECT_INSTRUCTION.format(goal=goal, trajectory=trajectory_text)
        tokens = count_tokens(prompt, model_name)
        if tokens_before is None:
            tokens_before = tokens
        if tokens <= budget or not hist:
            if len(hist) < len(history):
                dropped.append(f"dropped {len(history) - len(hist)} oldest history step(s)")
            return prompt, {
                "tokens_before": tokens_before, "tokens_after": tokens,
                "dropped": dropped, "task_id": task_id,
            }
        hist.pop(0)


def _build_trajectory(base_prompt: str, initial_ob: str,
                      history: List[Tuple[Optional[str], str, str]]) -> str:
    """Replays history in the same Thought:/Action:/Observation: shape the
    prompt asks the model to generate in (see ACT_STOP's docstring for why
    this replay format matters, not just the one-time instruction line).
    A history entry with no parsed thought (the unlabeled bare-action
    fallback form) simply omits the Thought: line for that turn."""
    s = base_prompt + '\n' + initial_ob
    for thought, action, observation in history:
        if thought:
            s += f'\nThought: {thought}'
        s += f'\nAction: {action}\nObservation: {observation}'
    return s
