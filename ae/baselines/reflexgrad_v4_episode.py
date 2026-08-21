"""Real ALFWorld episode loop for reflexgrad_v4 / reflexion_only_reflexgrad_v4
(Phase A3) -- ONE shared function for both baselines, parameterized only by
which wrapper's build_engine to call (textgrad_enabled is the only
difference, per Phase A2/A2.2's shared-engine requirement).

Reuses:
  - alfworld_runs_ae/environment.py::process_ob (same observation cleanup
    every other baseline in this repo uses)
  - alfworld_runs_ae/output_parser.py::parse_agent_output (the actor's
    action parser -- same shared parser react/reflexion/ae_full/
    react_reflact_anchor all go through; a parse failure here is a normal,
    budget-consuming event, NOT a RoleParseError -- that "fail loudly"
    contract is specifically for the OTHER structured roles, see
    reflexgrad_v4_llm.py's module docstring)

zero-shot, single episode / single trial, max_agent_steps=15 (reproduction
spec sections 3/4). Success (won) comes only from ALFWorld's own
info["won"] -- never from an LLM judgment, same convention as every other
baseline in this repo.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alfworld_runs_ae"))

from environment import process_ob  # noqa: E402
from output_parser import parse_agent_output  # noqa: E402

from ae.baselines.reflexgrad_v4_llm import (  # noqa: E402
    ReflexGradAPIError, ReflexGradChatLLM, RoleParseError, parse_decomposer_todos,
    parse_evaluator_score, parse_todo_verifier_result,
)
from ae.baselines.reflexgrad_v4_prompts import (  # noqa: E402
    CAUSAL_DIAGNOSER_PROMPT_TEMPLATE, DECOMPOSER_PROMPT_TEMPLATE,
    EVALUATOR_PROMPT_TEMPLATE, GRADIENT_PROMPT_TEMPLATE, INITIAL_BASE_POLICY,
    LOSS_PROMPT_TEMPLATE, OPTIMIZER_PROMPT_TEMPLATE, PLAN_GENERATOR_PROMPT_TEMPLATE,
    TODO_VERIFIER_PROMPT_TEMPLATE, TRAJECTORY_ANALYZER_PROMPT_TEMPLATE,
    build_actor_prompt, manifest_prompt_provenance,
)

MAX_AGENT_STEPS = 15  # reproduction spec sections 3.2/4.1

# Per-role max_tokens -- disclosed decoding defaults, not paper-stated
# values (the paper does not specify a token budget per role). Recorded in
# the manifest, not silently assumed.
_MAX_TOKENS_BY_ROLE = {
    "actor": 256, "evaluator": 16, "todo_verifier": 8, "decomposer": 256,
    "loss": 256, "gradient": 256, "optimizer": 256,
    "trajectory_analyzer": 256, "causal_diagnoser": 128, "plan_generator": 200,
}


def _batch_first(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _to_bool_scalar(value) -> bool:
    return bool(_batch_first(value))


class _ActionParserWithAdmissible:
    """action_parser_fn for the engine -- ActionParserFn is Callable[[str],
    Optional[str]] (one arg), but alfworld_runs_ae's shared parser needs the
    current admissible_commands too. This holds the latest admissible list
    so the episode loop can update it each step without changing the
    engine's parser call signature."""

    def __init__(self):
        self.admissible_commands: Optional[List[str]] = None

    def __call__(self, raw: str) -> Optional[str]:
        return parse_agent_output(raw, self.admissible_commands).parsed_action


def run_episode(env, env_name: str, task_goal: str, chat_llm: ReflexGradChatLLM,
                 build_engine_fn, task_id: Optional[str] = None,
                 max_agent_steps: int = MAX_AGENT_STEPS, to_print: bool = False) -> dict:
    """One real ALFWorld episode. `build_engine_fn` is either
    ae.baselines.reflexgrad_v4.build_engine or
    ae.baselines.reflexion_only_reflexgrad_v4.build_engine -- the ONLY
    difference between the two baselines (textgrad_enabled); everything
    else in this function is identical for both, per the shared-engine
    requirement."""
    step_records: List[dict] = []

    def _record_role_call(step_record: dict, role: str, messages: list, raw: str, parsed) -> None:
        step_record.setdefault("role_calls", []).append({
            "role": role, "messages": messages, "raw_generation": raw,
            "parsed_result": parsed,
        })

    _current = {"step_record": None}

    def _call_role(role: str, prompt_text: str, parse_fn=None):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            raw = chat_llm.call(role, messages, _MAX_TOKENS_BY_ROLE[role])
        except ReflexGradAPIError:
            raise
        parsed = parse_fn(raw) if parse_fn else raw.strip()
        if _current["step_record"] is not None:
            _record_role_call(_current["step_record"], role, messages, raw, parsed)
        return parsed

    def decomposer_fn(task, initial_observation):
        prompt = DECOMPOSER_PROMPT_TEMPLATE.format(
            task_description=task, initial_observation=initial_observation)
        return _call_role("decomposer", prompt, parse_decomposer_todos)

    def evaluator_fn(task, obs, action, next_obs):
        prompt = EVALUATOR_PROMPT_TEMPLATE.format(
            task_description=task, o_t=obs, a_t=action, o_t_plus_1=next_obs)
        return _call_role("evaluator", prompt, parse_evaluator_score)

    def loss_fn(task, policy, last3):
        prompt = LOSS_PROMPT_TEMPLATE.format(
            task_description=task, policy_pi=policy, k=len(last3), tuples=last3)
        return _call_role("loss", prompt)

    def gradient_fn(loss, policy):
        prompt = GRADIENT_PROMPT_TEMPLATE.format(loss_ell=loss, policy_pi=policy)
        return _call_role("gradient", prompt)

    def optimizer_fn(policy, gradient):
        prompt = OPTIMIZER_PROMPT_TEMPLATE.format(policy_pi=policy, gradient_g=gradient)
        return _call_role("optimizer", prompt)

    def trajectory_analyzer_fn(task, last5):
        prompt = TRAJECTORY_ANALYZER_PROMPT_TEMPLATE.format(
            task_description=task, m=len(last5), window_W=last5)
        return _call_role("trajectory_analyzer", prompt)

    def causal_diagnoser_fn(analysis, policy):
        prompt = CAUSAL_DIAGNOSER_PROMPT_TEMPLATE.format(
            trajectory_analysis=analysis, policy_pi=policy)
        return _call_role("causal_diagnoser", prompt)

    def plan_generator_fn(cause, policy):
        prompt = PLAN_GENERATOR_PROMPT_TEMPLATE.format(causal_trace_d=cause, policy_pi=policy)
        return _call_role("plan_generator", prompt)

    def todo_verifier_fn(current_todo, prev_obs, action, next_obs):
        prompt = TODO_VERIFIER_PROMPT_TEMPLATE.format(
            subgoal=current_todo, prev_obs=prev_obs, action=action, curr_obs=next_obs)
        return _call_role("todo_verifier", prompt, parse_todo_verifier_result)

    parser = _ActionParserWithAdmissible()

    def actor_fn(task, observation, todo, policy, slow_plan, memory):
        prompt = build_actor_prompt(
            task_description=task,
            active_todo=todo.content if todo is not None else None,
            policy=policy, active_slow_plan=slow_plan, observation=observation,
        )
        messages = [{"role": "user", "content": prompt}]
        raw = chat_llm.call("actor", messages, _MAX_TOKENS_BY_ROLE["actor"])
        if _current["step_record"] is not None:
            _record_role_call(_current["step_record"], "actor", messages, raw, None)  # parsed filled in below
        return raw

    engine = build_engine_fn(
        decomposer_fn=decomposer_fn, actor_fn=actor_fn, evaluator_fn=evaluator_fn,
        loss_fn=loss_fn, gradient_fn=gradient_fn, optimizer_fn=optimizer_fn,
        trajectory_analyzer_fn=trajectory_analyzer_fn, causal_diagnoser_fn=causal_diagnoser_fn,
        plan_generator_fn=plan_generator_fn, action_parser_fn=parser,
        todo_verifier_fn=todo_verifier_fn,
    )
    engine.base_policy = INITIAL_BASE_POLICY

    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    initial_observation = process_ob(ob)
    parser.admissible_commands = info.get('admissible_commands', [[]])[0]

    observation = initial_observation
    won = False
    termination_reason = "max_agent_steps"
    api_error_message = None

    try:
        engine.decompose(task_goal, initial_observation)

        for _ in range(max_agent_steps):
            step_record = {"todos_before": _snapshot_todos(engine)}
            _current["step_record"] = step_record

            parsed_action, raw_actor_output = engine.choose_action(task_goal, observation)
            # Backfill the actor's parsed action into its role-call record
            # (actor_fn itself doesn't know the parse result -- the engine
            # parses AFTER calling actor_fn).
            for rc in step_record.get("role_calls", []):
                if rc["role"] == "actor" and rc["parsed_result"] is None:
                    rc["parsed_result"] = parsed_action

            if parsed_action is None:
                route = engine.after_parse_failure()
                step_record.update({
                    "action": None, "route": route, "cooldown_remaining": engine.cooldown_remaining,
                    "policy_after": engine.base_policy, "evaluator_score": None,
                    "todos_after": _snapshot_todos(engine),
                    "agent_steps": engine.agent_steps, "env_actions": engine.env_actions,
                })
                step_records.append(step_record)
                if to_print:
                    print(f"[parse_failure] raw={raw_actor_output!r}")
                continue

            obs_raw, _reward_raw, done_raw, info = env.step([parsed_action])
            next_observation = process_ob(_batch_first(obs_raw))
            won = _to_bool_scalar(info.get('won', [False]))
            parser.admissible_commands = info.get('admissible_commands', [[]])[0]

            route = engine.after_env_step(task_goal, observation, parsed_action,
                                           next_observation, env_success=won)
            evaluator_score = engine.memory.last(1)[0][3] if len(engine.memory) else None
            step_record.update({
                "action": parsed_action, "observation_before": observation,
                "observation_after": next_observation, "route": route,
                "cooldown_remaining": engine.cooldown_remaining,
                "policy_after": engine.base_policy, "evaluator_score": evaluator_score,
                "todos_after": _snapshot_todos(engine),
                "agent_steps": engine.agent_steps, "env_actions": engine.env_actions,
            })
            step_records.append(step_record)
            if to_print:
                print(f"Action: {parsed_action}\nObservation: {next_observation}\nroute={route}")

            observation = next_observation
            if won:
                termination_reason = "won"
                break
    except (ReflexGradAPIError, RoleParseError) as e:
        termination_reason = "api_error"
        api_error_message = str(e)
    finally:
        _current["step_record"] = None

    return {
        "baseline": build_engine_fn.__module__.rsplit(".", 1)[-1],
        "task_id": task_id, "env_name": env_name, "task": task_goal,
        "initial_observation": initial_observation,
        "success": int(won), "won": bool(won), "termination_reason": termination_reason,
        "api_error_message": api_error_message,
        "agent_steps": engine.agent_steps, "env_actions": engine.env_actions,
        "llm_calls_by_role": engine.calls.as_dict(),
        "api_attempts_by_role": dict(chat_llm.api_attempts_by_role),
        "policy_final": engine.base_policy,
        "active_slow_plan_final": engine.active_slow_plan,
        "step_log": step_records,
        "manifest_fields": {
            "max_agent_steps": max_agent_steps,
            "gradient_cadence_k": engine.config.gradient_cadence_k,
            "slow_window_m": engine.config.slow_window_m,
            "cooldown_steps": engine.config.cooldown_steps,
            "low_progress_threshold": engine.config.low_progress_threshold,
            "working_memory_size": engine.config.working_memory_size,
            "todo_max_attempts": engine.config.todo_max_attempts,
            "textgrad_enabled": engine.config.textgrad_enabled,
            "use_task_decomposer": engine.config.use_task_decomposer,
            "model": chat_llm.model, "temperature": chat_llm.temperature,
            "thinking_mode": "backend_default",
            "max_api_retries": chat_llm.max_api_retries,
            "max_tokens_by_role": dict(_MAX_TOKENS_BY_ROLE),
            "prompt_provenance": manifest_prompt_provenance(),
            "initial_base_policy": INITIAL_BASE_POLICY,
        },
    }


def _snapshot_todos(engine) -> List[dict]:
    return [
        {"content": t.content, "status": t.status, "attempts": t.attempts,
         "last_action": t.last_action, "failure_reasons": list(t.failure_reasons)}
        for t in engine.todos
    ]
