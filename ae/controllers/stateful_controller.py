"""StatefulController: ties signal extraction -> AffectState update ->
StateSignature -> deterministic intervention selection, with
warmup/cooldown/max-intervention/patch-duration/recovery-grace bookkeeping
and per-step + episode-summary logging.

One instance is scoped to exactly one episode. Call reset() before each
new episode (the ae_full baseline wrapper does this) — nothing here persists
across episodes; there is no experience bank, no cross-episode memory, no
retrieval. This module never calls an LLM itself: interventions are fixed
template text (intervention_renderer.py), not generated.

Revision note (event detection): the first version gated intervention
firing on "the single dominant Mode changed since last step." That misses
in-place severity escalation — e.g. frustration already latched high
(dominant mode FRUSTRATED, reflect already active) and confidence then also
drops low: the dominant mode stays FRUSTRATED (priority order), so no
transition was ever detected and replan never fired even though
frustration_high AND confidence_low now both hold. This version computes a
full StateSignature every step and reacts to severity upgrades / rising
edges directly, independent of which single mode a priority collapse would
report. Mode/current_mode() is kept only for logs and visualization.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

from ae.core import InterventionType
from .affect_state import (
    AffectState, HysteresisTracker, Mode, StateSignature,
    compute_signature, initial_state, update_state,
)
from .config import AEConfig
from .intervention_renderer import render_directive
from .signals import SignalExtractor, StepSignals

# continue < verify < reflect < replan
_SEVERITY = {
    InterventionType.CONTINUE: 0,
    InterventionType.VERIFY: 1,
    InterventionType.REFLECT: 2,
    InterventionType.REPLAN: 3,
}


@dataclass
class StatefulController:
    config: AEConfig

    def __post_init__(self) -> None:
        self.signal_extractor = SignalExtractor(self.config.signals)
        self.reset()

    # ---- lifecycle ----------------------------------------------------
    def reset(self) -> None:
        """Clear all per-episode state. Must be called before each new
        episode; the controller itself never resets the environment."""
        self.state: AffectState = initial_state(self.config)
        self.hysteresis = HysteresisTracker()
        self.previous_mode: Mode = Mode.NORMAL  # logging-only, see current_mode()
        self.previous_signature: Optional[StateSignature] = None
        self.previous_invalid_action_flag: bool = False
        self.last_candidate_severity: int = 0
        self.step_index: int = 0
        self.cooldown_remaining: int = 0
        self.active_patch_type: Optional[InterventionType] = None
        self.active_patch_remaining: int = 0
        self.recovery_grace_remaining: int = 0
        self.intervention_count: int = 0
        self.intervention_counts: dict = {t.value: 0 for t in InterventionType if t != InterventionType.CONTINUE}
        self.trigger_steps: List[int] = []
        self.step_log: List[dict] = []
        self.episode_termination_reason: str = "unknown"
        # counters surfaced in the episode summary
        self.grace_consumed_count: int = 0
        self.suppressed_exhausted_count: int = 0
        self.final_exhausted_count: int = 0

    def consume_grace_step(self) -> None:
        """Called by the agent loop when a would-be exhausted_repeated
        termination is suppressed by an active recovery grace window."""
        if self.recovery_grace_remaining > 0:
            self.recovery_grace_remaining -= 1
        self.grace_consumed_count += 1
        self.suppressed_exhausted_count += 1

    def note_final_exhausted(self) -> None:
        """Called by the agent loop when exhausted_repeated actually fires
        (no grace left to suppress it)."""
        self.final_exhausted_count += 1

    # ---- per-step entry point -----------------------------------------
    def step(
        self,
        *,
        action: str,
        observation: str,
        admissible_before: List[str],
        recent_actions: List[str],
        recent_observations: List[str],
        max_steps: int,
        is_think_action: bool,
        legacy_termination_candidate: Optional[str] = None,
        termination_suppressed: bool = False,
        suppression_reason: Optional[str] = None,
        reward: Optional[float] = None,
        done: Optional[bool] = None,
        won: Optional[bool] = None,
    ) -> dict:
        """Process one (action, observation) transition. Returns the full
        log record for this step (also appended to self.step_log)."""
        self.step_index += 1

        signals = self.signal_extractor.extract(
            action=action,
            observation=observation,
            admissible_before=admissible_before,
            recent_actions=recent_actions,
            recent_observations=recent_observations,
            step_index=self.step_index,
            max_steps=max_steps,
            is_think_action=is_think_action,
        )

        state_before = self.state.copy()
        self.state = update_state(self.state, signals, self.config)
        self.hysteresis.update(self.state, self.config)
        current_mode = self.hysteresis.current_mode()  # logging only
        previous_mode = self.previous_mode

        signature = compute_signature(self.state, self.hysteresis, self.config)
        candidate, candidate_reason = self._select_intervention(signals, self.state, signature)
        candidate_severity = _SEVERITY[candidate]

        # ---- edge detection (independent of warmup/cooldown/budget) ----
        severity_upgrade = candidate_severity > self.last_candidate_severity
        invalid_flag = signals.invalid_action >= 0.5
        invalid_rising_edge = invalid_flag and not self.previous_invalid_action_flag
        if self.previous_signature is None:
            # First step: any already-active band counts as a fresh entry
            # (there's no prior step to have "exited" from).
            flag_rising_edge = any(dataclasses.astuple(signature))
        else:
            flag_rising_edge = any(
                getattr(signature, f.name) and not getattr(self.previous_signature, f.name)
                for f in dataclasses.fields(signature)
            )
        is_event = candidate != InterventionType.CONTINUE and (
            severity_upgrade or invalid_rising_edge or flag_rising_edge
        )

        intervention = InterventionType.CONTINUE
        reason = "no new event" if candidate == InterventionType.CONTINUE else "condition persists, no new event"
        triggered_new_patch = False

        in_warmup = self.step_index <= self.config.warmup_steps
        in_cooldown = self.cooldown_remaining > 0
        budget_exhausted = self.intervention_count >= self.config.max_interventions

        if is_event:
            if in_warmup:
                reason = f"event -> {candidate.value} suppressed: warmup ({self.step_index}/{self.config.warmup_steps})"
            elif in_cooldown:
                reason = f"event -> {candidate.value} suppressed: cooldown ({self.cooldown_remaining} steps left)"
            elif budget_exhausted:
                reason = f"event -> {candidate.value} suppressed: max_interventions reached ({self.config.max_interventions})"
            else:
                intervention = candidate
                reason = candidate_reason
                self.active_patch_type = intervention
                self.active_patch_remaining = self.config.patch_duration_steps
                self.cooldown_remaining = self.config.cooldown_steps
                self.recovery_grace_remaining = self.config.recovery_grace_steps
                self.intervention_count += 1
                self.intervention_counts[intervention.value] += 1
                self.trigger_steps.append(self.step_index)
                triggered_new_patch = True

        if not triggered_new_patch and self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        if not triggered_new_patch and self.active_patch_remaining > 0:
            self.active_patch_remaining -= 1
            if self.active_patch_remaining <= 0:
                self.active_patch_type = None

        self.previous_mode = current_mode
        self.previous_signature = signature
        self.previous_invalid_action_flag = invalid_flag
        self.last_candidate_severity = candidate_severity

        record = {
            "step": self.step_index,
            "action": action,
            "observation": observation,
            "signals": signals.as_dict(),
            "state_before": state_before.as_dict(),
            "state_after": self.state.as_dict(),
            "state_signature": signature.as_dict(),
            "previous_mode": previous_mode.value,
            "current_mode": current_mode.value,
            "candidate_intervention": candidate.value,
            "intervention": intervention.value,
            "intervention_reason": reason,
            "active_patch_remaining_steps": self.active_patch_remaining,
            "legacy_termination_candidate": legacy_termination_candidate,
            "termination_suppressed": termination_suppressed,
            "suppression_reason": suppression_reason,
            "recovery_grace_remaining": self.recovery_grace_remaining,
            "reward": reward,
            "done": done,
            "won": won,
        }
        self.step_log.append(record)
        return record

    def _select_intervention(self, signals: StepSignals, state: AffectState, signature: StateSignature):
        """Same rule tree as before, now driven by the full StateSignature
        instead of a single collapsed mode, so e.g. frustration_high AND
        confidence_low both being true is never hidden by mode-priority
        collapse."""
        thr = self.config.signals.repetition_trigger_threshold
        invalid = signals.invalid_action >= 0.5
        repeated = signals.repeated_action >= thr or signals.repeated_observation >= thr

        if signature.frustration_high and signature.confidence_low:
            return (
                InterventionType.REPLAN,
                f"frustration_high({state.frustration:.2f}) and confidence_low({state.confidence:.2f})",
            )
        if invalid or (signature.frustration_medium and repeated):
            if invalid:
                return InterventionType.REFLECT, "invalid_action"
            return (
                InterventionType.REFLECT,
                f"frustration_medium({state.frustration:.2f}>={self.config.frustration_medium}) "
                f"and repeated action/observation",
            )
        if signature.uncertainty_high or signature.surprise_high:
            which = "uncertainty_high" if signature.uncertainty_high else "surprise_high"
            return (
                InterventionType.VERIFY,
                f"{which} (uncertainty={state.uncertainty:.2f}, surprise={state.surprise:.2f})",
            )
        return InterventionType.CONTINUE, "no rule matched"

    # ---- prompt-facing directive ---------------------------------------
    def active_directive(self) -> str:
        """Directive text to splice into the NEXT LLM prompt, or "" if no
        patch is active. Never shown as part of the persistent trajectory —
        callers must append this only to the ephemeral prompt string for
        one LLM call, not to `history`."""
        if self.active_patch_type is None:
            return ""
        return render_directive(self.active_patch_type)

    # ---- episode summary -------------------------------------------------
    def summary(self, *, success: bool, env_steps: int, termination_reason: str) -> dict:
        return {
            "success": int(success),
            "environment_steps": env_steps,
            "normal_react_llm_calls": env_steps,
            "intervention_llm_calls": 0,  # v1: directives are fixed templates spliced into the
                                           # existing per-step ReAct call, not a separate LLM call.
            "input_output_tokens": None,  # not tracked: AnyOpenAILLM discards response.usage (see report)
            "intervention_count": self.intervention_count,
            "intervention_counts": dict(self.intervention_counts),
            "final_affect_state": self.state.as_dict(),
            "trigger_steps": list(self.trigger_steps),
            "termination_reason": termination_reason,
            "grace_consumed_count": self.grace_consumed_count,
            "suppressed_exhausted_count": self.suppressed_exhausted_count,
            "final_exhausted_count": self.final_exhausted_count,
        }
