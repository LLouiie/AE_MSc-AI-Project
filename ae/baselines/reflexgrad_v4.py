"""reflexgrad_v4: TextGrad (fast) + Reflexion (slow) both enabled.

Thin config wrapper around ae/baselines/reflexgrad_v4_engine.py -- the state
machine itself lives there and is shared verbatim with
reflexion_only_reflexgrad_v4.py; this file only supplies the
textgrad_enabled=True config value and default numeric parameters from the
reproduction spec (section 4.1).

Real role-callable wiring (actual LLM prompts for actor/evaluator/decomposer/
loss/gradient/optimizer/trajectory_analyzer/causal_diagnoser/plan_generator)
and the real ALFWorld run_episode loop are a later phase -- this phase
(A2) only builds and locks the shared engine against mock/scripted
callables (see ae/baselines/tests/test_reflexgrad_v4_engine.py).
"""

from __future__ import annotations

from typing import Optional

from ae.baselines.reflexgrad_v4_engine import (
    ActionParserFn, ReflexGradV4Config, ReflexGradV4Engine, RoleFn,
)

MAX_ENV_STEPS = 15  # spec section 4.1


def build_config(**overrides) -> ReflexGradV4Config:
    base = dict(
        working_memory_size=10, slow_window_m=5, low_progress_threshold=4,
        cooldown_steps=5, gradient_cadence_k=3,
        reflexion_enabled=True, textgrad_enabled=True, use_task_decomposer=True,
        # 4 is the current official ReflexGrad repo's implementation value,
        # not a number the v4 paper's text states explicitly -- see
        # ReflexGradV4Config.todo_max_attempts's docstring. Must be recorded
        # as such in the manifest whenever a real run uses it.
        todo_max_attempts=4,
    )
    base.update(overrides)
    return ReflexGradV4Config(**base)


def build_engine(decomposer_fn: RoleFn, actor_fn: RoleFn, evaluator_fn: RoleFn,
                  loss_fn: RoleFn, gradient_fn: RoleFn, optimizer_fn: RoleFn,
                  trajectory_analyzer_fn: RoleFn, causal_diagnoser_fn: RoleFn,
                  plan_generator_fn: RoleFn, action_parser_fn: Optional[ActionParserFn] = None,
                  todo_verifier_fn: Optional[RoleFn] = None,
                  **config_overrides) -> ReflexGradV4Engine:
    return ReflexGradV4Engine(
        config=build_config(**config_overrides),
        decomposer_fn=decomposer_fn, actor_fn=actor_fn, evaluator_fn=evaluator_fn,
        loss_fn=loss_fn, gradient_fn=gradient_fn, optimizer_fn=optimizer_fn,
        trajectory_analyzer_fn=trajectory_analyzer_fn,
        causal_diagnoser_fn=causal_diagnoser_fn, plan_generator_fn=plan_generator_fn,
        action_parser_fn=action_parser_fn, todo_verifier_fn=todo_verifier_fn,
    )
