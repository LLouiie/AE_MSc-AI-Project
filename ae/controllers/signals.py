"""Deterministic per-step signal extraction for the AE controller.

No embeddings, no extra LLM evaluator calls — every signal here is computed
from string normalization + token-set (Jaccard) overlap + the ALFWorld
environment's own returned fields (admissible_commands, won), confirmed
empirically (2026-07-25) to be present in info: `info.keys() ==
['extra.gamefile', 'admissible_commands', 'won']`.

Honesty note (per explicit instruction not to fabricate precise semantics):
ALFWorld exposes no per-step partial-credit / progress signal — `won` is
only ever True on the terminal successful step. `progress_signal` below is
kept as the field name (renaming would touch StepSignals, every
StateWeights.progress_signal/progress_relief config key, the log schema,
and the test suite for no behavioral gain), but it is NOT a task-progress
estimate — it is purely an `observation_novelty_proxy`: "did the observation
change enough, on a legal action, to look unstuck," computed from Jaccard
token overlap against recent history. Nothing in this repo should describe
`progress_signal` as ground-truth task progress; every reference to it here
and in stateful_controller.py/config.py is written as "novelty proxy," not
"progress." `unexpected_outcome` is the same kind of heuristic proxy, built
from admissibility + observation similarity, not a learned outcome model.

Cheaper, more direct progress-adjacent signals ALFWorld could plausibly
support (not implemented this round — see the AE revision report for the
per-signal cost/feasibility note):
  - admissible_commands set change (new commands appearing/disappearing
    between steps) — already have both sides of this in the agent loop's
    admissible_before/after, just not diffed into a signal yet; cheapest
    of the five to add.
  - location change (parsing "You arrive at X" before process_ob() strips
    it) — cheap, text-pattern based like current signals.
  - receptacle state change ("is open"/"is closed" keyword deltas) — cheap,
    same caveat as unexpected_outcome (heuristic, not semantic).
  - target object pickup/place event (action starts with take/put AND
    observation confirms it) — cheap, but ties the signal to ALFWorld's
    specific verb vocabulary.
  - inventory change — NOT cheap: ALFWorld has no free inventory readout
    in `info`; would need an extra `env.step(['inventory'])` per check,
    i.e. a real extra environment/LLM turn, not a free signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .config import SignalConfig

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Fixed template fragments that add no discriminative information to a
# Jaccard comparison; stripped before tokenizing. ALFWorld observations are
# already run through alfworld_runs_ae/environment.py::process_ob() before
# reaching here (strips the "You arrive at loc X. " navigation prefix), so
# this list only needs the remaining boilerplate.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^-= welcome to textworld,? alfred! =-\s*", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    """lowercase, strip, collapse whitespace, drop fixed boilerplate."""
    if not text:
        return ""
    t = text.lower().strip()
    for pat in _BOILERPLATE_PATTERNS:
        t = pat.sub("", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


def token_set(text: str) -> set:
    return set(_TOKEN_RE.findall(normalize_text(text)))


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity in [0, 1]. Two empty strings are
    treated as maximally similar (both "nothing"); one empty vs one
    non-empty is maximally dissimilar."""
    ta, tb = token_set(a), token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class StepSignals:
    invalid_action: float = 0.0
    repeated_action: float = 0.0
    repeated_observation: float = 0.0
    observation_novelty: float = 0.0
    unexpected_outcome: float = 0.0
    progress_signal: float = 0.0
    budget_ratio: float = 0.0

    def as_dict(self) -> dict:
        return {
            "invalid_action": self.invalid_action,
            "repeated_action": self.repeated_action,
            "repeated_observation": self.repeated_observation,
            "observation_novelty": self.observation_novelty,
            "unexpected_outcome": self.unexpected_outcome,
            "progress_signal": self.progress_signal,
            "budget_ratio": self.budget_ratio,
        }


@dataclass
class SignalExtractor:
    config: SignalConfig = field(default_factory=SignalConfig)

    def extract(
        self,
        *,
        action: str,
        observation: str,
        admissible_before: List[str],
        recent_actions: List[str],
        recent_observations: List[str],
        step_index: int,
        max_steps: int,
        is_think_action: bool,
    ) -> StepSignals:
        """Compute all 7 signals for one (action, observation) transition.

        `admissible_before` is the admissible_commands list captured BEFORE
        this action was taken (from the previous step's info, or the
        initial reset) — ALFWorld/TextWorld's admissible_commands describes
        what's legal in the state the action was chosen from, not the
        resulting state, so validity must be checked against the prior
        list. `recent_actions`/`recent_observations` are the window (most
        recent last) preceding this step, already excluding the current
        one.
        """
        window = self.config.repetition_window

        # -- invalid_action --
        # 'think:' actions are intentionally not real environment commands
        # (the agent's own scratchpad convention), so they're never scored
        # as invalid even though they'll never appear in admissible_commands.
        if is_think_action:
            invalid_action = 0.0
        else:
            invalid_action = 0.0 if action.strip() in admissible_before else 1.0

        # -- repeated_action --
        # Max similarity of this action to any of the last `window` actions.
        recent_a = recent_actions[-window:] if window > 0 else []
        repeated_action = max((jaccard(action, a) for a in recent_a), default=0.0)

        # -- repeated_observation --
        recent_o = recent_observations[-window:] if window > 0 else []
        repeated_observation = max((jaccard(observation, o) for o in recent_o), default=0.0)

        # -- observation_novelty --
        # Complement of repeated_observation: how different this
        # observation is from the recent window. Kept as its own named
        # signal (rather than inlining 1 - repeated_observation at call
        # sites) since it gets its own weight in state updates.
        observation_novelty = 1.0 - repeated_observation

        # -- unexpected_outcome (heuristic proxy, see module docstring) --
        # A nominally legal action (or a think: action, which always
        # "succeeds" as a no-op) that produced an observation near-identical
        # to the immediately preceding one, or the literal ALFWorld
        # no-op string "Nothing happens.", counts as unexpected: the agent
        # had no reason to expect no effect.
        immediately_prior = recent_o[-1] if recent_o else ""
        no_effect = (
            normalize_text(observation) == "nothing happens."
            or jaccard(observation, immediately_prior) >= self.config.unexpected_similarity_threshold
        )
        unexpected_outcome = 1.0 if (invalid_action == 0.0 and no_effect and not is_think_action) else 0.0

        # -- progress_signal := observation_novelty_proxy (NOT task progress; see module docstring) --
        # Valid, non-think action whose observation is novel enough
        # relative to recent history. This rewards "the world visibly
        # changed in a new way," which is the closest observable proxy to
        # progress ALFWorld's API surface allows without a learned model —
        # it is explicitly NOT a claim that the task got closer to done.
        if is_think_action or invalid_action == 1.0:
            progress_signal = 0.0
        else:
            progress_signal = (
                observation_novelty
                if repeated_observation < self.config.novelty_progress_threshold
                else 0.0
            )

        # -- budget_ratio --
        budget_ratio = min(1.0, step_index / max_steps) if max_steps > 0 else 0.0

        return StepSignals(
            invalid_action=invalid_action,
            repeated_action=repeated_action,
            repeated_observation=repeated_observation,
            observation_novelty=observation_novelty,
            unexpected_outcome=unexpected_outcome,
            progress_signal=progress_signal,
            budget_ratio=budget_ratio,
        )
