"""AE controller configuration.

Every numeric constant that shapes signal->state->intervention behavior
lives here, loaded from configs/controllers/*.yaml. Nothing in signals.py,
affect_state.py, or stateful_controller.py should hardcode a threshold,
weight, or decay — if a formula needs a number, it comes from this dataclass
so the whole controller can be re-tuned (on the practice set only, per
standing project constraint) without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class SignalConfig:
    # Number of recent actions/observations kept for repetition/novelty
    # comparison (Jaccard token overlap over this window).
    repetition_window: int = 3
    # A valid action whose resulting observation is >= this similar to the
    # immediately preceding observation counts as "unexpected_outcome" (a
    # nominally legal action that produced no discernible effect). This is
    # a heuristic proxy, not a semantic outcome model — see module docstring
    # in signals.py.
    unexpected_similarity_threshold: float = 0.8
    # Below this token-overlap value, an observation counts as "novel"
    # enough to count toward progress_signal.
    novelty_progress_threshold: float = 0.5
    # repeated_action / repeated_observation (continuous Jaccard values, used
    # directly as state-update inputs) are binarized against this threshold
    # only for the controller's rule-tree check ("repeated_action or
    # repeated_observation" in the reflect branch) — the raw floats still
    # feed the AffectState update regardless of this cutoff.
    repetition_trigger_threshold: float = 0.6


@dataclass
class StateWeights:
    """Named per-term weights for one state variable's update delta.
    Each is multiplied by the named signal (or, for *_relief terms, by
    progress_signal) and summed into that state's raw delta before decay
    and clipping. These are first-pass engineering constants tuned by hand
    for a first working version, not derived from any theory."""

    invalid_action: float = 0.0
    repeated_action: float = 0.0
    repeated_observation: float = 0.0
    observation_novelty: float = 0.0
    unexpected_outcome: float = 0.0
    progress_signal: float = 0.0
    budget_ratio: float = 0.0
    # Multiplied by progress_signal; meant to be negative (relieves
    # uncertainty/frustration) or positive (boosts confidence) on progress.
    progress_relief: float = 0.0


@dataclass
class HysteresisBand:
    enter: float
    exit: float


@dataclass
class AEConfig:
    # ---- initial values ----
    initial_uncertainty: float = 0.30
    initial_frustration: float = 0.00
    initial_surprise: float = 0.00
    initial_confidence: float = 0.60

    # ---- decay (applied to previous_state each step, before adding delta) ----
    decay_uncertainty: float = 0.75
    decay_frustration: float = 0.85
    decay_surprise: float = 0.35
    decay_confidence: float = 0.75

    # ---- per-state signal weights ----
    uncertainty_weights: StateWeights = field(default_factory=lambda: StateWeights(
        repeated_observation=0.25, budget_ratio=0.10, progress_relief=-0.30,
    ))
    frustration_weights: StateWeights = field(default_factory=lambda: StateWeights(
        invalid_action=0.35, repeated_action=0.30, repeated_observation=0.20,
        progress_relief=-0.35,
    ))
    surprise_weights: StateWeights = field(default_factory=lambda: StateWeights(
        unexpected_outcome=0.60, observation_novelty=0.25,
    ))
    confidence_weights: StateWeights = field(default_factory=lambda: StateWeights(
        progress_signal=0.35, observation_novelty=0.10,
        invalid_action=-0.25, repeated_action=-0.15,
    ))

    # ---- hysteresis bands ----
    uncertainty_hysteresis: HysteresisBand = field(default_factory=lambda: HysteresisBand(0.70, 0.30))
    frustration_hysteresis: HysteresisBand = field(default_factory=lambda: HysteresisBand(0.70, 0.30))
    surprise_hysteresis: HysteresisBand = field(default_factory=lambda: HysteresisBand(0.70, 0.30))
    # Confidence hysteresis is inverted (LOW is the abnormal band).
    confidence_low_hysteresis: HysteresisBand = field(default_factory=lambda: HysteresisBand(0.30, 0.70))

    # ---- controller thresholds ----
    frustration_medium: float = 0.45

    # ---- budgets ----
    warmup_steps: int = 3
    cooldown_steps: int = 3
    max_interventions: int = 3
    patch_duration_steps: int = 3
    # After an intervention fires, suppress the agent-level repeated-action
    # ("exhausted") early termination for up to this many subsequent steps,
    # giving the injected directive an actual chance to change behavior
    # before the old stuck-loop heuristic ends the episode. Never applies
    # in react mode (controller=None). Never suppresses a real environment
    # terminal (`done`) or a genuine win.
    recovery_grace_steps: int = 3

    # ---- post-intervention exact-repeat override (see stateful_controller
    #      .py's HIGHEST-PRIORITY OVERRIDE block) ----
    # Master switch: False fully disables the override (the step() method
    # never checks consecutive_exact_action_repeat while an intervention is
    # pending), reverting to the pre-override behavior in every other
    # respect (put/move canonicalization, warmup fix, frustration-gate fix,
    # grace/cooldown/budget/legacy early stop all remain exactly as-is).
    post_intervention_repeat_enabled: bool = True
    # False (default): REFLECT escalates to REPLAN on the FIRST consecutive
    # exact repeat while REFLECT is pending (original behavior).
    # True: REFLECT does NOT escalate on the first consecutive exact
    # repeat -- that occurrence is absorbed (logged, no override fires,
    # the step falls through to normal processing) to give the model one
    # more chance to recover. Only a SECOND, immediately-consecutive exact
    # repeat while REFLECT is still pending escalates to REPLAN. VERIFY
    # (always escalates to REFLECT on first repeat) and REPLAN (always
    # terminates on first repeat) are unaffected by this flag.
    post_intervention_repeat_reflect_double: bool = False

    # ---- repeated-action first-encounter REFLECT entry (see
    #      stateful_controller.py::_select_intervention()) ----
    # False (default): unchanged original routing -- frustration_high AND
    # confidence_low always selects REPLAN, regardless of what drove it.
    # True: when frustration_high AND confidence_low both hold AND a
    # genuine exact-repeat is present (consecutive_exact_action_repeat)
    # AND no intervention is currently pending (this is the FIRST time
    # this specific stuck sequence has been seen, not a recurrence while
    # something is already active), select REFLECT instead of REPLAN this
    # once. Reuses the existing patch_duration_steps + meaningful_change_
    # since_intervention + _ESCALATION[REFLECT]=REPLAN machinery
    # unchanged for the recover/escalate judgment -- no new mechanism.
    # Does not affect invalid_action-only routing (no repeat present),
    # does not affect frustration_high+confidence_low without a repeat,
    # does not affect VERIFY, and does not touch cooldown/warmup/budget
    # gating (still applied downstream exactly as before, on whichever
    # candidate is returned).
    reflect_first_encounter_repeat_enabled: bool = False

    # ---- signal extraction ----
    signals: SignalConfig = field(default_factory=SignalConfig)

    # ---- run mode ----
    # react            -> AE fully disabled, behavior identical to pre-AE code
    # ae_full          -> full state accumulation + hysteresis + intervention
    # fixed_interval   -> reserved, not implemented in this pass
    # stateless_trigger-> reserved, not implemented in this pass
    # ae_no_hysteresis -> reserved, not implemented in this pass (ablation)
    mode: str = "ae_full"


def load_config(path: str | None) -> AEConfig:
    if not path or not os.path.exists(path):
        return AEConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    def _weights(d: dict | None) -> StateWeights:
        return StateWeights(**(d or {}))

    def _band(d: dict | None, default: HysteresisBand) -> HysteresisBand:
        if not d:
            return default
        return HysteresisBand(enter=d.get("enter", default.enter), exit=d.get("exit", default.exit))

    cfg = AEConfig()
    initial = raw.get("initial", {})
    decay = raw.get("decay", {})
    weights = raw.get("weights", {})
    hysteresis = raw.get("hysteresis", {})
    controller = raw.get("controller", {})
    signals = raw.get("signals", {})

    return AEConfig(
        initial_uncertainty=initial.get("uncertainty", cfg.initial_uncertainty),
        initial_frustration=initial.get("frustration", cfg.initial_frustration),
        initial_surprise=initial.get("surprise", cfg.initial_surprise),
        initial_confidence=initial.get("confidence", cfg.initial_confidence),
        decay_uncertainty=decay.get("uncertainty", cfg.decay_uncertainty),
        decay_frustration=decay.get("frustration", cfg.decay_frustration),
        decay_surprise=decay.get("surprise", cfg.decay_surprise),
        decay_confidence=decay.get("confidence", cfg.decay_confidence),
        uncertainty_weights=_weights(weights.get("uncertainty")) if weights.get("uncertainty") else cfg.uncertainty_weights,
        frustration_weights=_weights(weights.get("frustration")) if weights.get("frustration") else cfg.frustration_weights,
        surprise_weights=_weights(weights.get("surprise")) if weights.get("surprise") else cfg.surprise_weights,
        confidence_weights=_weights(weights.get("confidence")) if weights.get("confidence") else cfg.confidence_weights,
        uncertainty_hysteresis=_band(hysteresis.get("uncertainty"), cfg.uncertainty_hysteresis),
        frustration_hysteresis=_band(hysteresis.get("frustration"), cfg.frustration_hysteresis),
        surprise_hysteresis=_band(hysteresis.get("surprise"), cfg.surprise_hysteresis),
        confidence_low_hysteresis=_band(hysteresis.get("confidence_low"), cfg.confidence_low_hysteresis),
        frustration_medium=controller.get("frustration_medium", cfg.frustration_medium),
        warmup_steps=controller.get("warmup_steps", cfg.warmup_steps),
        cooldown_steps=controller.get("cooldown_steps", cfg.cooldown_steps),
        max_interventions=controller.get("max_interventions", cfg.max_interventions),
        patch_duration_steps=controller.get("patch_duration_steps", cfg.patch_duration_steps),
        recovery_grace_steps=controller.get("recovery_grace_steps", cfg.recovery_grace_steps),
        post_intervention_repeat_enabled=controller.get(
            "post_intervention_repeat_enabled", cfg.post_intervention_repeat_enabled),
        post_intervention_repeat_reflect_double=controller.get(
            "post_intervention_repeat_reflect_double", cfg.post_intervention_repeat_reflect_double),
        reflect_first_encounter_repeat_enabled=controller.get(
            "reflect_first_encounter_repeat_enabled", cfg.reflect_first_encounter_repeat_enabled),
        signals=SignalConfig(
            repetition_window=signals.get("repetition_window", cfg.signals.repetition_window),
            unexpected_similarity_threshold=signals.get(
                "unexpected_similarity_threshold", cfg.signals.unexpected_similarity_threshold),
            novelty_progress_threshold=signals.get(
                "novelty_progress_threshold", cfg.signals.novelty_progress_threshold),
            repetition_trigger_threshold=signals.get(
                "repetition_trigger_threshold", cfg.signals.repetition_trigger_threshold),
        ),
        mode=raw.get("mode", cfg.mode),
    )
