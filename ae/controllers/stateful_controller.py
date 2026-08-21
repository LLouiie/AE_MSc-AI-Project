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

Revision note (intervention outcome tracking): the severity/rising-edge
gating above only ever fires on a NEW event -- once frustration_high AND
confidence_low both latch and a REPLAN fires, if the affect state simply
stays pinned at that same level (the underlying problem was never
actually fixed), no further rising edge or severity upgrade ever occurs,
so no second intervention can ever fire for the rest of the episode. This
adds a second, orthogonal mechanism: every fired intervention is tracked
as "pending" until its patch duration expires, at which point its outcome
is evaluated using signals.local_state_change_proxy (did the world
actually change while the directive was active?) -- "recovered" clears
the pending state and re-arms the SAME severity level to fire again on
recurrence (without waiting for a hysteresis exit/re-entry cycle);
"unresolved" schedules a severity-escalated follow-up (verify->reflect->
replan->replan) that fires as soon as cooldown clears, still bounded by
max_interventions. This is deliberately NOT a per-step retrigger -- only
patch expiry ever produces an outcome evaluation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

from ae.core import InterventionType
from .canonical_action import canonicalize_action
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

# Escalation path for an unresolved intervention outcome. REPLAN is
# already the most severe tier, so an unresolved replan just schedules
# another replan rather than a no-op.
_ESCALATION = {
    InterventionType.VERIFY: InterventionType.REFLECT,
    InterventionType.REFLECT: InterventionType.REPLAN,
    InterventionType.REPLAN: InterventionType.REPLAN,
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
        self.warmup_suppressed_count: int = 0

        # ---- intervention outcome tracking (per-episode, never leaks) ----
        self.next_intervention_id: int = 1
        self.pending_intervention: bool = False
        self.pending_intervention_type: Optional[InterventionType] = None
        self.intervention_start_step: Optional[int] = None
        self.intervention_expiry_step: Optional[int] = None
        self.meaningful_change_since_intervention: bool = False
        self.unresolved_intervention_count: int = 0
        self.steps_since_meaningful_change: int = 0
        # escalation scheduled by an unresolved outcome, waiting for cooldown
        self.escalation_pending_type: Optional[InterventionType] = None
        self.escalation_reason: Optional[str] = None
        # current/most-recent intervention's identity, surfaced every step
        self.current_intervention_id: Optional[int] = None
        self.current_intervention_type: Optional[InterventionType] = None
        self.current_intervention_outcome: Optional[str] = None  # "pending" | "recovered" | "unresolved"
        self.current_outcome_evaluation_step: Optional[int] = None
        self.current_escalated_from: Optional[str] = None

        # ---- post-intervention exact-repeat (hard failure evidence) -------
        # Set by step() when an exact repeat of the action fires during an
        # active (pending) intervention; the agent loop must check this
        # right after calling step() and terminate before the next LLM call.
        # None means "no forced termination this step".
        self.force_terminate_reason: Optional[str] = None
        # Tracks, only when config.post_intervention_repeat_reflect_double
        # is True, whether the immediately-preceding step already absorbed
        # one un-escalated consecutive exact repeat while REFLECT was
        # pending -- reset on every new/escalated intervention (see
        # _arm_intervention) and whenever the repeat streak breaks, so it
        # only ever measures two *immediately consecutive* repeats within
        # one continuous REFLECT window.
        self._reflect_repeat_pending: bool = False

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

    def note_warmup_suppressed(self) -> None:
        """Called by the agent loop when a would-be exhausted_repeated
        termination is suppressed because the controller is still (at, or
        one step past) its warmup window and could not yet have armed
        grace -- distinct from a real recovery_grace_remaining suppression,
        see consume_grace_step()."""
        self.warmup_suppressed_count += 1

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
        admissible_after: Optional[List[str]] = None,
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
            admissible_after=admissible_after,
        )

        state_before = self.state.copy()
        self.state = update_state(self.state, signals, self.config)
        self.hysteresis.update(self.state, self.config)
        current_mode = self.hysteresis.current_mode()  # logging only
        previous_mode = self.previous_mode

        signature = compute_signature(self.state, self.hysteresis, self.config)
        candidate, candidate_reason = self._select_intervention(signals, self.state, signature)
        candidate_severity = _SEVERITY[candidate]

        # ---- HIGHEST-PRIORITY OVERRIDE: post-intervention exact-repeat ----
        # Hard evidence the active intervention's directive was NOT acted
        # on: the model repeated its immediately-preceding action verbatim
        # while a fired intervention is still awaiting outcome evaluation.
        # Bypasses cooldown, recovery_grace, the normal patch_duration_steps
        # wait, and the normal event-detection/warmup/budget gating below
        # entirely -- this check (and its own budget/max_interventions
        # bound) is the ONLY thing that runs this step when it fires.
        # env_success (`won`) still outranks it: the caller only reaches
        # here after ALFWorld's own done/won check, but `not env_success` is
        # checked directly too as a second, self-contained guard.
        env_success = bool(won)
        positive_progress = (
            (reward is not None and reward > 0)
            or signals.local_state_change_proxy >= 0.5
        )
        pending_before = self.pending_intervention
        active_type_before = self.current_intervention_type
        active_id_before = self.current_intervention_id
        repeat_signal = (
            pending_before
            and active_type_before is not None
            and signals.consecutive_exact_action_repeat >= 0.5
            and not env_success
            and not positive_progress
        )

        # Exploratory ablation (post_intervention_repeat_reflect_double):
        # while REFLECT is pending, the first consecutive exact repeat is
        # absorbed (no override, no escalation -- the model gets one more
        # step to recover on its own) rather than escalating immediately.
        # Only a second, immediately-consecutive repeat escalates. VERIFY
        # and REPLAN are untouched by this flag -- they still act on the
        # first repeat exactly as before.
        reflect_absorb_first_repeat = False
        if self.config.post_intervention_repeat_enabled and self.config.post_intervention_repeat_reflect_double:
            if active_type_before == InterventionType.REFLECT:
                if repeat_signal and not self._reflect_repeat_pending:
                    reflect_absorb_first_repeat = True
                    self._reflect_repeat_pending = True
                elif not repeat_signal:
                    # Streak broken: a non-repeat step while REFLECT is
                    # still pending resets the "already saw one" flag.
                    self._reflect_repeat_pending = False
            else:
                self._reflect_repeat_pending = False

        post_intervention_exact_repeat = (
            self.config.post_intervention_repeat_enabled
            and repeat_signal
            and not reflect_absorb_first_repeat
        )

        if post_intervention_exact_repeat:
            # Resolve the currently-pending intervention right now, as
            # "unresolved" -- the repeat is itself decisive, do not wait for
            # patch_duration_steps to expire naturally.
            self.current_intervention_outcome = "unresolved"
            self.current_outcome_evaluation_step = self.step_index
            self.pending_intervention = False
            self.pending_intervention_type = None
            self.unresolved_intervention_count += 1
            self.steps_since_meaningful_change += 1

            budget_available = self.intervention_count < self.config.max_interventions
            if active_type_before == InterventionType.VERIFY:
                if budget_available:
                    self._arm_intervention(InterventionType.REFLECT, escalated_from=InterventionType.VERIFY.value)
                    intervention, transition = InterventionType.REFLECT, "VERIFY->REFLECT"
                    reason = "post-intervention exact repeat during active verify -> escalating to reflect"
                else:
                    intervention, transition = InterventionType.CONTINUE, "VERIFY->TERMINATE(budget_exhausted)"
                    self.force_terminate_reason = "post_intervention_repeat_budget_exhausted"
                    reason = ("post-intervention exact repeat during active verify, but "
                              f"max_interventions reached ({self.config.max_interventions}) -> terminating")
            elif active_type_before == InterventionType.REFLECT:
                if budget_available:
                    self._arm_intervention(InterventionType.REPLAN, escalated_from=InterventionType.REFLECT.value)
                    intervention, transition = InterventionType.REPLAN, "REFLECT->REPLAN"
                    reason = "post-intervention exact repeat during active reflect -> escalating to replan"
                else:
                    intervention, transition = InterventionType.CONTINUE, "REFLECT->TERMINATE(budget_exhausted)"
                    self.force_terminate_reason = "post_intervention_repeat_budget_exhausted"
                    reason = ("post-intervention exact repeat during active reflect, but "
                              f"max_interventions reached ({self.config.max_interventions}) -> terminating")
            else:  # REPLAN
                intervention, transition = InterventionType.CONTINUE, "REPLAN->TERMINATE"
                self.force_terminate_reason = "post_replan_exact_repeat"
                reason = "post-intervention exact repeat during active replan -> terminating episode"

            prev_action_text = recent_actions[-1] if recent_actions else ""
            post_repeat_info = {
                "active_intervention_id": active_id_before,
                "active_intervention_type": active_type_before.value,
                "previous_raw_action": prev_action_text,
                "current_raw_action": action,
                "previous_canonical_action": canonicalize_action(prev_action_text),
                "current_canonical_action": canonicalize_action(action),
                "transition": transition,
                "termination_reason": self.force_terminate_reason,
                "recovery_grace_bypassed": True,
                "cooldown_bypassed": True,
            }

            self.previous_mode = current_mode
            if self.step_index > self.config.warmup_steps:
                self.previous_signature = signature
                self.previous_invalid_action_flag = signals.invalid_action >= 0.5
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
                "intervention_id": self.current_intervention_id,
                "intervention_type": (
                    self.current_intervention_type.value if self.current_intervention_type else None
                ),
                "intervention_outcome": self.current_intervention_outcome,
                "outcome_evaluation_step": self.current_outcome_evaluation_step,
                "meaningful_change_since_intervention": self.meaningful_change_since_intervention,
                "escalated_from": self.current_escalated_from,
                "steps_since_meaningful_change": self.steps_since_meaningful_change,
                "post_intervention_exact_repeat": post_repeat_info,
                "reflect_repeat_absorbed": False,
            }
            self.step_log.append(record)
            return record

        # ---- meaningful-change bookkeeping (drives outcome evaluation) --
        meaningful_change_this_step = signals.local_state_change_proxy >= 0.5
        if meaningful_change_this_step:
            self.steps_since_meaningful_change = 0
            if self.pending_intervention:
                self.meaningful_change_since_intervention = True
        else:
            self.steps_since_meaningful_change += 1

        # ---- pending-intervention outcome evaluation (on patch expiry) --
        # Only evaluated once, exactly when a patch expires -- never on
        # every step a patch happens to be active.
        if self.pending_intervention and self.step_index >= self.intervention_expiry_step:
            if self.meaningful_change_since_intervention:
                self.current_intervention_outcome = "recovered"
                # Re-arm: allow the SAME severity level to fire again on a
                # future recurrence without requiring affect values to
                # drop first or a fresh hysteresis exit/re-entry cycle --
                # resetting last_candidate_severity makes the next
                # non-CONTINUE candidate look like a fresh severity
                # upgrade to the edge-detection logic below.
                self.last_candidate_severity = 0
            else:
                self.current_intervention_outcome = "unresolved"
                self.unresolved_intervention_count += 1
                escalated_type = _ESCALATION[self.pending_intervention_type]
                self.escalation_pending_type = escalated_type
                self.escalation_reason = (
                    f"{self.pending_intervention_type.value} intervention "
                    f"#{self.current_intervention_id} unresolved after "
                    f"{self.config.patch_duration_steps} steps -> escalating to {escalated_type.value}"
                )
            self.current_outcome_evaluation_step = self.step_index
            self.pending_intervention = False
            self.pending_intervention_type = None

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

        in_warmup = self.step_index <= self.config.warmup_steps
        in_cooldown = self.cooldown_remaining > 0
        budget_exhausted = self.intervention_count >= self.config.max_interventions
        triggered_new_patch = False

        if (
            self.escalation_pending_type is not None
            and not in_warmup and not in_cooldown and not budget_exhausted
        ):
            # A scheduled escalation from an unresolved outcome takes
            # priority over (and does not require) a fresh is_event this
            # step -- it fires as soon as cooldown clears.
            escalated_from_value = None
            if self.current_intervention_type is not None:
                escalated_from_value = self.current_intervention_type.value
            intervention = self.escalation_pending_type
            reason = self.escalation_reason
            self._arm_intervention(intervention, escalated_from=escalated_from_value)
            self.escalation_pending_type = None
            self.escalation_reason = None
            triggered_new_patch = True
        elif is_event:
            if in_warmup:
                reason = f"event -> {candidate.value} suppressed: warmup ({self.step_index}/{self.config.warmup_steps})"
            elif in_cooldown:
                reason = f"event -> {candidate.value} suppressed: cooldown ({self.cooldown_remaining} steps left)"
            elif budget_exhausted:
                reason = f"event -> {candidate.value} suppressed: max_interventions reached ({self.config.max_interventions})"
            else:
                intervention = candidate
                reason = candidate_reason
                self._arm_intervention(intervention, escalated_from=None)
                triggered_new_patch = True

        if not triggered_new_patch and self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        if not triggered_new_patch and self.active_patch_remaining > 0:
            self.active_patch_remaining -= 1
            if self.active_patch_remaining <= 0:
                self.active_patch_type = None

        self.previous_mode = current_mode
        # Edge-detection baseline (previous_signature/previous_invalid_action_flag/
        # last_candidate_severity) is frozen while in_warmup rather than rolled
        # forward: is_event above is suppressed unconditionally during warmup, so
        # updating this baseline every warmup step would silently "consume" the
        # rising edge (severity_upgrade/invalid_rising_edge/flag_rising_edge all
        # compare against the immediately-preceding step) -- by the time warmup
        # ends, a condition that had already been sitting active since step 1
        # would look like "no new event" forever, even though it was never once
        # actually evaluated for a real intervention. Freezing the baseline at
        # its pre-warmup value guarantees the first non-warmup call compares
        # against a genuinely stale baseline, so any band/severity still active
        # once warmup lifts is detected as a fresh edge. self.state/hysteresis
        # above are NOT frozen -- only this edge-detection bookkeeping is.
        if not in_warmup:
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
            "intervention_id": self.current_intervention_id,
            "intervention_type": (
                self.current_intervention_type.value if self.current_intervention_type else None
            ),
            "intervention_outcome": self.current_intervention_outcome,
            "outcome_evaluation_step": self.current_outcome_evaluation_step,
            "meaningful_change_since_intervention": self.meaningful_change_since_intervention,
            "escalated_from": self.current_escalated_from,
            "steps_since_meaningful_change": self.steps_since_meaningful_change,
            "post_intervention_exact_repeat": None,
            "reflect_repeat_absorbed": reflect_absorb_first_repeat,
        }
        self.step_log.append(record)
        return record

    def _arm_intervention(self, intervention: InterventionType, *, escalated_from: Optional[str]) -> None:
        """Shared bookkeeping for firing a new intervention, whether from a
        fresh is_event or a scheduled unresolved-outcome escalation."""
        self.active_patch_type = intervention
        self.active_patch_remaining = self.config.patch_duration_steps
        self.cooldown_remaining = self.config.cooldown_steps
        self.recovery_grace_remaining = self.config.recovery_grace_steps
        self.intervention_count += 1
        self.intervention_counts[intervention.value] += 1
        self.trigger_steps.append(self.step_index)

        self.pending_intervention = True
        self.pending_intervention_type = intervention
        self.intervention_start_step = self.step_index
        self.intervention_expiry_step = self.step_index + self.config.patch_duration_steps
        self.meaningful_change_since_intervention = False

        self.current_intervention_id = self.next_intervention_id
        self.next_intervention_id += 1
        self.current_intervention_type = intervention
        self.current_intervention_outcome = "pending"
        self.current_outcome_evaluation_step = None
        self.current_escalated_from = escalated_from
        self._reflect_repeat_pending = False

    def _select_intervention(self, signals: StepSignals, state: AffectState, signature: StateSignature):
        """Same rule tree as before, now driven by the full StateSignature
        instead of a single collapsed mode, so e.g. frustration_high AND
        confidence_low both being true is never hidden by mode-priority
        collapse."""
        invalid = signals.invalid_action >= 0.5
        # Strict adjacent-step exact repeat only -- deliberately NOT
        # repeated_action (repetition_window-max similarity) or
        # repeated_observation (Jaccard on observation text, which
        # over-triggers on same-template-different-location exploration,
        # e.g. two different empty shelves both producing "you see
        # nothing."). See signals.py::consecutive_exact_action_repeat().
        repeated = signals.consecutive_exact_action_repeat >= 0.5

        if signature.frustration_high and signature.confidence_low:
            # REPLAN requires sustained evidence beyond the same-step
            # frustration_high+confidence_low spike: repeated_action shares
            # weight on both frustration (+) and confidence (-), so a
            # single exact repeat alone can push both bands simultaneously
            # within a few steps (see audit_reports/ae_reflect_replan_
            # formula_redesign*.md) -- that is local-fault evidence for
            # REFLECT, not on its own evidence of a plan-level failure.
            # steps_since_meaningful_change (already tracked, reused here
            # unmodified) crossing patch_duration_steps is the same bar the
            # controller already uses elsewhere to judge "an intervention
            # had a fair chance and nothing changed" -- reused here as the
            # "sustained no-progress" gate for REPLAN specifically.
            if self.steps_since_meaningful_change >= self.config.patch_duration_steps:
                return (
                    InterventionType.REPLAN,
                    f"frustration_high({state.frustration:.2f}) and confidence_low({state.confidence:.2f}) "
                    f"sustained(steps_since_meaningful_change={self.steps_since_meaningful_change}"
                    f">=patch_duration_steps={self.config.patch_duration_steps})",
                )
            # Not yet sustained -- fall through; may still resolve to
            # REFLECT below (invalid_action or frustration_medium+repeated).
        if invalid or (signature.frustration_medium and repeated):
            if invalid:
                return InterventionType.REFLECT, "invalid_action"
            return (
                InterventionType.REFLECT,
                f"frustration_medium({state.frustration:.2f}>={self.config.frustration_medium}) "
                f"and consecutive_exact_action_repeat",
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
            "unresolved_intervention_count": self.unresolved_intervention_count,
            "warmup_suppressed_count": self.warmup_suppressed_count,
        }
