"""AffectState: the four intra-episode control variables and their update
rule, plus hysteresis-gated mode tracking.

These are affect-inspired control variables, not a claim about real
emotion — the names (uncertainty/frustration/surprise/confidence) label
*what observable pattern each one tracks*, nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import AEConfig
from .signals import StepSignals


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class AffectState:
    uncertainty: float
    frustration: float
    surprise: float
    confidence: float

    def as_dict(self) -> dict:
        return {
            "uncertainty": round(self.uncertainty, 4),
            "frustration": round(self.frustration, 4),
            "surprise": round(self.surprise, 4),
            "confidence": round(self.confidence, 4),
        }

    def copy(self) -> "AffectState":
        return AffectState(self.uncertainty, self.frustration, self.surprise, self.confidence)


def initial_state(config: AEConfig) -> AffectState:
    return AffectState(
        uncertainty=config.initial_uncertainty,
        frustration=config.initial_frustration,
        surprise=config.initial_surprise,
        confidence=config.initial_confidence,
    )


def _weighted_delta(weights, signals: StepSignals) -> float:
    """new_state = clip(decay * previous_state + signal_delta, 0, 1);
    signal_delta is this function's return value: a sum of
    (named_weight * named_signal) terms plus progress_relief * progress_signal."""
    return (
        weights.invalid_action * signals.invalid_action
        + weights.repeated_action * signals.repeated_action
        + weights.repeated_observation * signals.repeated_observation
        + weights.observation_novelty * signals.observation_novelty
        + weights.unexpected_outcome * signals.unexpected_outcome
        + weights.progress_signal * signals.progress_signal
        + weights.budget_ratio * signals.budget_ratio
        + weights.progress_relief * signals.progress_signal
    )


def update_state(prev: AffectState, signals: StepSignals, config: AEConfig) -> AffectState:
    """new_state = clip(decay * previous_state + signal_delta, 0.0, 1.0)
    applied independently per variable, per the spec. progress_signal is
    the one signal that appears in every state's formula (as a positive
    term for confidence, as a negative `progress_relief` term for
    uncertainty/frustration) — this is the concrete implementation of
    "progress should lower uncertainty and frustration" / "valid progress
    ... should raise confidence"."""
    uncertainty = clip01(
        config.decay_uncertainty * prev.uncertainty
        + _weighted_delta(config.uncertainty_weights, signals)
    )
    frustration = clip01(
        config.decay_frustration * prev.frustration
        + _weighted_delta(config.frustration_weights, signals)
    )
    surprise = clip01(
        config.decay_surprise * prev.surprise
        + _weighted_delta(config.surprise_weights, signals)
    )
    confidence = clip01(
        config.decay_confidence * prev.confidence
        + _weighted_delta(config.confidence_weights, signals)
    )
    return AffectState(uncertainty, frustration, surprise, confidence)


class Mode(Enum):
    NORMAL = "NORMAL"
    UNCERTAIN = "UNCERTAIN"
    FRUSTRATED = "FRUSTRATED"
    SURPRISED = "SURPRISED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


# Priority when multiple bands are simultaneously abnormal: frustration is
# the strongest signal that the *current strategy* is failing (most
# actionable — replan), low confidence next (the agent doesn't trust its
# own read of the situation), then uncertainty (evidence is thin, worth
# verifying), then surprise (a one-off mismatch, least urgent of the four).
# This fixed order — not dict/insertion order — is what "使用明确的优先级
# 规则" in the spec asks for.
_MODE_PRIORITY = [Mode.FRUSTRATED, Mode.LOW_CONFIDENCE, Mode.UNCERTAIN, Mode.SURPRISED]


@dataclass
class HysteresisTracker:
    """One boolean latch per abnormal band (uncertainty/frustration/
    surprise high, confidence low). A band flips to "active" only when the
    state crosses `enter`, and back to "inactive" only when it crosses
    `exit` — never re-evaluated against a single shared threshold, which is
    what prevents flicker for values oscillating near one cutoff."""

    uncertainty_active: bool = False
    frustration_active: bool = False
    surprise_active: bool = False
    confidence_low_active: bool = False

    def update(self, state: AffectState, config: AEConfig) -> None:
        u = config.uncertainty_hysteresis
        f = config.frustration_hysteresis
        s = config.surprise_hysteresis
        c = config.confidence_low_hysteresis

        if not self.uncertainty_active and state.uncertainty >= u.enter:
            self.uncertainty_active = True
        elif self.uncertainty_active and state.uncertainty <= u.exit:
            self.uncertainty_active = False

        if not self.frustration_active and state.frustration >= f.enter:
            self.frustration_active = True
        elif self.frustration_active and state.frustration <= f.exit:
            self.frustration_active = False

        if not self.surprise_active and state.surprise >= s.enter:
            self.surprise_active = True
        elif self.surprise_active and state.surprise <= s.exit:
            self.surprise_active = False

        # Confidence's abnormal band is LOW, so the direction is inverted:
        # "enter" the low-confidence band from above (falling below
        # enter), "exit" it from below (rising above exit).
        if not self.confidence_low_active and state.confidence <= c.enter:
            self.confidence_low_active = True
        elif self.confidence_low_active and state.confidence >= c.exit:
            self.confidence_low_active = False

    def current_mode(self) -> Mode:
        """Single dominant mode, kept ONLY for logging/visualization —
        see StateSignature below for what event detection actually uses.
        A priority-collapsed single mode necessarily hides simultaneously-
        active bands (e.g. frustration_high AND confidence_low both true
        collapses to just "FRUSTRATED" here), which is exactly why using
        this alone to gate intervention events missed the replan upgrade
        case this revision fixes."""
        active = []
        if self.frustration_active:
            active.append(Mode.FRUSTRATED)
        if self.confidence_low_active:
            active.append(Mode.LOW_CONFIDENCE)
        if self.uncertainty_active:
            active.append(Mode.UNCERTAIN)
        if self.surprise_active:
            active.append(Mode.SURPRISED)
        if not active:
            return Mode.NORMAL
        for m in _MODE_PRIORITY:
            if m in active:
                return m
        return Mode.NORMAL  # unreachable, defensive


@dataclass(frozen=True)
class StateSignature:
    """The full discrete abnormality signature for one step — every band
    that's currently active, not collapsed into a single dominant mode.
    This (plus the raw signals) is what event detection is computed from;
    Mode/current_mode() above remains a logging-only projection of this."""

    uncertainty_high: bool
    frustration_medium: bool
    frustration_high: bool
    surprise_high: bool
    confidence_low: bool

    def as_dict(self) -> dict:
        return {
            "uncertainty_high": self.uncertainty_high,
            "frustration_medium": self.frustration_medium,
            "frustration_high": self.frustration_high,
            "surprise_high": self.surprise_high,
            "confidence_low": self.confidence_low,
        }


def compute_signature(state: AffectState, hysteresis: HysteresisTracker, config: AEConfig) -> StateSignature:
    return StateSignature(
        uncertainty_high=hysteresis.uncertainty_active,
        frustration_medium=state.frustration >= config.frustration_medium,
        frustration_high=hysteresis.frustration_active,
        surprise_high=hysteresis.surprise_active,
        confidence_low=hysteresis.confidence_low_active,
    )
