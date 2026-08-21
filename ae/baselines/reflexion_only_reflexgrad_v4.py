"""reflexion_only_reflexgrad_v4: TextGrad (fast) disabled, Reflexion (slow)
kept enabled -- ReflexGrad v4's own Table 2 ablation, NOT the original
Reflexion method (see AE_Qwen3_ALFWorld_Baseline_Reproduction_Spec.md
section 3.1: single episode, zero-shot, no reset-and-retry trial loop --
do not call this result "Reflexion" unqualified, per standing instruction).

Thin config wrapper around ae/baselines/reflexgrad_v4_engine.py -- same
ReflexGradV4Engine class as reflexgrad_v4.py, not a copy of the state
machine. The only difference from reflexgrad_v4.py's config is
textgrad_enabled=False; every other field (working_memory_size,
slow_window_m, low_progress_threshold, cooldown_steps, gradient_cadence_k,
reflexion_enabled) is identical, per spec section 3.2 ("其余超参数相同").
"""

from __future__ import annotations

from typing import Optional

from ae.baselines.reflexgrad_v4 import build_config as _build_v4_config
from ae.baselines.reflexgrad_v4_engine import (
    ActionParserFn, ReflexGradV4Config, ReflexGradV4Engine, RoleFn,
)

MAX_ENV_STEPS = 15  # spec section 3.2, same as reflexgrad_v4


def build_config(**overrides) -> ReflexGradV4Config:
    overrides.setdefault("textgrad_enabled", False)
    return _build_v4_config(**overrides)


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
