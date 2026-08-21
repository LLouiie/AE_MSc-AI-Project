"""react_reflact_anchor: MPO/ReflAct-v2-Appendix-G published-anchor ReAct baseline.

Distinct from ae/baselines/react.py (this repo's existing AE-project ReAct
baseline, which stays untouched and keeps running under the practice/exam
protocol -- see AE_Qwen3_ALFWorld_Baseline_Reproduction_Spec.md section one).
This module targets the *published-anchor* reproduction spec (Qwen3-8B x
ALFWorld ReAct, 65.7% single-run sanity anchor):

  - prompt/ICL source: MPO, vendored verbatim (see
    alfworld_runs_ae/mpo_prompts.py + external/mpo_reflact/SOURCE.md), not
    this repo's own alfworld_3prompts.json;
  - API mode: chat completions with Qwen3 thinking explicitly disabled via
    chat_template_kwargs, never raw /v1/completions (see
    hotpotqa_runs/llm.py::AnyOpenAILLM's docstring for why every OTHER
    baseline uses raw completions instead -- that reasoning does not carry
    over here, since MPO's own reference agent uses chat.completions and
    the reproduction spec requires the real chat-template mechanism, not an
    accidental side effect of avoiding the chat template entirely);
  - step budget: 30 agent-steps / 512 max_completion_tokens, both
    "inherited from the MPO repo" per the reproduction spec -- not values
    ReflAct v2 Appendix G states itself. Recorded as such in manifest_fields,
    never presented as paper-explicit.

Reuses this repo's own environment creation, task->env resolution and
success determination (info["won"]) -- see mpo_prompts.py's module
docstring for the one deliberate divergence from MPO's own environment code
(MPO's AlfWorldEnv.step() sets success from `done` alone; this repo keeps
its info["won"]-based convention instead).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alfworld_runs_ae"))

from environment import process_ob  # noqa: E402
from mpo_prompts import (  # noqa: E402
    MPO_SOURCE_COMMIT, MPO_SOURCE_REPO, build_first_turn_prompt, get_mpo_category,
    parse_mpo_action,
)

MAX_AGENT_STEPS = int(os.getenv("ANCHOR_REACT_MAX_AGENT_STEPS", "30"))
MAX_COMPLETION_TOKENS = int(os.getenv("ANCHOR_REACT_MAX_COMPLETION_TOKENS", "512"))
THINKING_CONTROL_METHOD = "chat_template_kwargs.enable_thinking=false"
API_MODE = "chat_completions"
CHAT_TEMPLATE = "qwen3_native"

# MPO's own parse-failure text (envs/alfworld_env_reference.py::step, the
# `except Exception` branch) -- reproduced verbatim, typo included, since
# rewriting it would be "rewriting prompt/environment-feedback content"
# ourselves, which is exactly what we were told not to do.
_PARSE_FAILURE_OBSERVATION = "Observation: Error Input. Your input must contains 'Action: '"


class NativeThinkBlockError(RuntimeError):
    """Raised when a completion contains a native <think> block despite
    requesting enable_thinking=False. Per explicit instruction this must
    surface as a stop-and-report condition -- never a silent fallback to
    raw completions or a silently-swallowed warning."""


class QwenChatAnchorLLM:
    """Chat-completions client for the published-anchor ReAct baseline only.

    Deliberately separate from hotpotqa_runs/llm.py::AnyOpenAILLM, which
    stays on raw /v1/completions for every other baseline in this repo (see
    that module's docstring for why). This class always calls
    client.chat.completions.create and always passes
    chat_template_kwargs={"enable_thinking": False} via extra_body -- the
    vLLM-documented mechanism for Qwen3's thinking switch on an OpenAI-
    compatible chat endpoint.
    """

    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY",
                 temperature: float = 0.0, max_tokens: int = MAX_COMPLETION_TOKENS,
                 client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client
        self.last_meta: dict = {}

    def __call__(self, messages: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        self.last_meta = {
            "finish_reason": choice.finish_reason,
            "was_truncated": choice.finish_reason == "length",
            "usage": (
                {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                 "total_tokens": usage.total_tokens}
                if usage is not None else "unavailable"
            ),
        }
        if "<think>" in text:
            raise NativeThinkBlockError(
                "completion contained a native <think> block despite "
                f"enable_thinking=False: {text[:200]!r}"
            )
        return text


def _batch_first(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _to_bool_scalar(value) -> bool:
    return bool(_batch_first(value))


def run_episode(env, env_name: str, chat_llm: QwenChatAnchorLLM,
                 task_id: Optional[str] = None,
                 max_agent_steps: int = MAX_AGENT_STEPS, to_print: bool = False) -> dict:
    """One published-anchor ReAct episode.

    Returns a dict with the agent_steps / env_actions / llm_calls_by_role
    split (reproduction spec section 5): agent_steps counts every decision
    attempt including parser failures, env_actions only counts real
    env.step() calls, llm_calls_by_role is a dict (one role, "actor", for
    this baseline).
    """
    mpo_category = get_mpo_category(env_name)

    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    ob = process_ob(ob)
    admissible_before = info.get('admissible_commands', [[]])[0]

    flat_prompt, _unused_chat_messages = build_first_turn_prompt(mpo_category, ob)
    messages = [{"role": "user", "content": flat_prompt}]
    first_turn_messages_snapshot = [dict(m) for m in messages]

    agent_steps = 0
    env_actions = 0
    llm_calls_by_role = {"actor": 0}
    step_log: List[dict] = []
    won = False
    done = False
    termination_reason = "max_agent_steps"

    for _ in range(max_agent_steps):
        agent_steps += 1
        raw_generation = chat_llm(messages)
        llm_calls_by_role["actor"] += 1
        meta = chat_llm.last_meta

        messages.append({"role": "assistant", "content": raw_generation})
        parsed_action = parse_mpo_action(raw_generation)

        if parsed_action is not None:
            env_actions += 1
            obs_raw, _reward_raw, done_raw, info = env.step([parsed_action])
            observation = process_ob(_batch_first(obs_raw))
            won = _to_bool_scalar(info.get('won', [False]))
            done = _to_bool_scalar(done_raw)
            admissible_after = info.get('admissible_commands', [[]])[0]
            action_is_admissible = parsed_action.strip() in admissible_before
            observation_for_history = f"Observation: {observation}"
        else:
            observation = _PARSE_FAILURE_OBSERVATION
            admissible_after = admissible_before
            action_is_admissible = None
            observation_for_history = observation

        messages.append({"role": "user", "content": observation_for_history})

        step_log.append({
            "raw_generation": raw_generation,
            "parsed_action": parsed_action,
            "parse_success": parsed_action is not None,
            "action_is_admissible": action_is_admissible,
            "observation": observation,
            "finish_reason": meta.get("finish_reason"),
            "was_truncated": meta.get("was_truncated"),
            "usage": meta.get("usage"),
        })
        admissible_before = admissible_after

        if to_print:
            print(f"Action: {parsed_action}\nObservation: {observation}")

        if done:
            termination_reason = "success" if won else "env_done_without_success"
            break

    return {
        "baseline": "react_reflact_anchor",
        "success": int(won),
        "termination_reason": termination_reason,
        "agent_steps": agent_steps,
        "env_actions": env_actions,
        "llm_calls_by_role": llm_calls_by_role,
        "step_log": step_log,
        "first_turn_messages": first_turn_messages_snapshot,
        "mpo_category": mpo_category,
        "manifest_fields": {
            "protocol_kind": "published_anchor",
            "api_mode": API_MODE,
            "chat_template": CHAT_TEMPLATE,
            "thinking_control_method": THINKING_CONTROL_METHOD,
            "mpo_source_repo": MPO_SOURCE_REPO,
            "mpo_source_commit": MPO_SOURCE_COMMIT,
            "max_agent_steps": max_agent_steps,
            "max_completion_tokens": chat_llm.max_tokens,
        },
    }
