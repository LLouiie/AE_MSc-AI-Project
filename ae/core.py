"""Shared data structures every baseline and controller in ae/ builds on.

StepContext is what a baseline hands to a controller after each environment
step; InterventionDecision is what the controller hands back. Every baseline
(react, reflexion, adapt, reflact, reflexgrad, ae) must be able to run behind
this same interface so comparisons stay apples-to-apples per-baseline
intervention/token accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


@dataclass
class StepContext:
    task_id: str
    step_index: int
    goal: str
    current_plan: Optional[str]
    action: str
    observation: str
    action_valid: bool
    tool_error: bool
    recent_actions: list[str] = field(default_factory=list)
    recent_observations: list[str] = field(default_factory=list)
    completed_subgoals: list[str] = field(default_factory=list)
    token_usage: int = 0


@dataclass
class AppraisalState:
    progress: float = 0.0
    frustration: float = 0.0
    uncertainty: float = 0.0
    surprise: float = 0.0
    confidence: float = 0.0
    recovery: float = 0.0
    cooldown_remaining: int = 0


class InterventionType(Enum):
    CONTINUE = "continue"
    VERIFY = "verify"
    REFLECT = "reflect"
    REPLAN = "replan"


@dataclass
class InterventionDecision:
    intervention: InterventionType
    reason: str
    scores: dict = field(default_factory=dict)


class MetaController(Protocol):
    def reset(self, task: dict) -> None: ...

    def observe(self, step_context: StepContext) -> AppraisalState: ...

    def decide(
        self, state: AppraisalState, step_context: StepContext
    ) -> InterventionDecision: ...
