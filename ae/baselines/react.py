"""A1 ReAct baseline: no meta-controller, single trial.

Thin wrapper around alfworld_runs_ae/agents.py::ALFWorldAgent, which already
implements the paper's Thought/Action/Observation loop (Yao et al. 2022,
arXiv:2210.03629) against our own AnyOpenAILLM/vLLM backend — see
BASELINE_RESEARCH.md section 6 for the paper-vs-code consistency check.
Adapted reproduction (different LLM backend / no notebook), not exact.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alfworld_runs_ae"))

from agents import ALFWorldAgent  # noqa: E402


def run_episode(env, goal: str, task_type: str, act_llm, to_print: bool = False) -> dict:
    agent = ALFWorldAgent(act_llm)
    trajectory, success = agent.run(env, goal, task_type, to_print=to_print)
    return {
        "baseline": "react",
        "success": int(success),
        "trajectory": trajectory,
    }
