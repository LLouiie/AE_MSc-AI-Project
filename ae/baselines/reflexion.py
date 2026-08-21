"""A2 Reflexion baseline: post-episode failure -> verbal reflection -> retry.

Thin wrapper around alfworld_runs_ae/agents.py::ALFWorldReflectAgent, with
rules_text pinned to "" so no RulePool/consolidation import is ever touched
(that pipeline is legacy as of legacy/rule-library-2026-07-25, see
AE_MIGRATION_AUDIT.md). Core mechanism verified against Shinn et al. 2023,
arXiv:2303.11366: reflect-on-failure, inject reflections into next trial's
prompt, stop at max_trials or first success — see BASELINE_RESEARCH.md
section 6. Adapted reproduction (different LLM backend), not exact; and not
the same code that produced the thesis's existing HotpotQA Reflexion number
(that one is AE/baselines/reflexion, the official langchain fork).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alfworld_runs_ae"))

from agents import ALFWorldReflectAgent  # noqa: E402


def run_episode(env, goal: str, task_type: str, act_llm, reflect_llm,
                 max_trials: int = 4, to_print: bool = False,
                 termination_policy: str = None, task_id: str = None, demo_config=None) -> dict:
    agent = ALFWorldReflectAgent(act_llm=act_llm, reflect_llm=reflect_llm,
                                  termination_policy=termination_policy, demo_config=demo_config)
    trial_records = []
    for trial in range(max_trials):
        success = agent.run_trial(env, goal, task_type, rules_text="", to_print=to_print, task_id=task_id)
        trial_records.append({
            "trial": trial + 1,
            "success": int(success),
            "trajectory": agent.last_trajectory,
            "step_log": list(agent.act_agent.step_log),
        })
        if success:
            break
        if trial < max_trials - 1:
            agent.reflect(goal, task_type, task_id=task_id)
            trial_records[-1]["reflect_trunc_info"] = agent.last_reflect_trunc_info

    return {
        "baseline": "reflexion",
        "success": int(agent.is_success),
        "trials_used": trial + 1,
        "reflections": list(agent.reflections),
        "trial_records": trial_records,
        "trajectory": agent.last_trajectory,
    }
