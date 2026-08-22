"""AE baseline (`ae_full` run mode): stateful intra-episode appraisal
controller on top of the same ALFWorldAgent ReAct loop react.py/reflexion.py
already use — see alfworld_runs_ae/agents.py::ALFWorldAgent's optional
`controller` param. No experience bank, no retrieval, no cross-episode
memory, no environment reset, no parameter updates: the controller is
constructed fresh per episode and discarded after run_episode() returns.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alfworld_runs_ae"))

from agents import ALFWorldAgent, summarize_generation_log  # noqa: E402

from ae.controllers.config import AEConfig  # noqa: E402
from ae.controllers.stateful_controller import StatefulController  # noqa: E402


def run_episode(env, goal: str, task_type: str, act_llm, config: AEConfig, to_print: bool = False,
                 termination_policy: str = None, task_id: str = None, demo_config=None,
                 ablation_mode: str = "full", random_seed: int = 42) -> dict:
    stable_task_seed = int.from_bytes(
        hashlib.sha256(f"{random_seed}:{task_id or 'unknown'}".encode()).digest()[:8], "big"
    )
    controller = StatefulController(
        config, ablation_mode=ablation_mode, random_seed=stable_task_seed
    )
    agent = ALFWorldAgent(act_llm, controller=controller, termination_policy=termination_policy,
                           demo_config=demo_config)
    trajectory, success = agent.run(env, goal, task_type, to_print=to_print, task_id=task_id)

    env_steps = len(controller.step_log)
    summary = controller.summary(
        success=success,
        env_steps=env_steps,
        termination_reason=controller.episode_termination_reason,
    )
    # agent.step_log (generation-path fields: raw_generation/parsed_*/
    # finish_reason/usage/...) and controller.step_log (AE signal/state/
    # intervention fields) advance in lockstep, one record per step -- merge
    # them index-wise so each step appears once with every field, rather
    # than making callers zip two parallel logs themselves.
    merged_step_log = [
        {**gen_rec, **ae_rec}
        for gen_rec, ae_rec in zip(agent.step_log, controller.step_log)
    ]
    return {
        "baseline": "ae_full",
        "ae_ablation_mode": ablation_mode,
        "ae_random_seed": random_seed,
        "success": int(success),
        "trajectory": trajectory,
        "ae_step_log": merged_step_log,
        "ae_summary": summary,
        "generation_summary": summarize_generation_log(agent.step_log),
    }
