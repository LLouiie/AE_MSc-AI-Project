"""Deterministic per-step signal extraction for the AE controller.

No embeddings, no extra LLM evaluator calls — every signal here is computed
from string normalization + token-set (Jaccard) overlap + canonical action
parsing (canonical_action.py) + the ALFWorld environment's own returned
fields (admissible_commands, won), reconfirmed empirically (2026-07-25) to
be present in info: `info.keys() == ['extra.gamefile', 'won',
'admissible_commands']`. No `inventory`/`facts`/structured-state field
exists anywhere in `info` — never assume one is available; every signal
below is built only from what's actually there.

Honesty note (per explicit instruction not to fabricate precise semantics):
ALFWorld exposes no per-step partial-credit / progress signal — `won` is
only ever True on the terminal successful step. `progress_signal` below is
kept as the field name (renaming would touch StepSignals, every
StateWeights.progress_signal/progress_relief config key, the log schema,
and the test suite for no behavioral gain), but it is NOT a task-progress
estimate — it is purely an `observation_novelty_proxy`: "did the observation
change enough, on a legal action, to look unstuck," computed from Jaccard
token overlap against recent history. Nothing in this repo should describe
`progress_signal` as ground-truth task progress.

`unexpected_outcome`, `information_gain_proxy`, and `local_state_change_proxy`
are three DISTINCT heuristic proxies (previously `unexpected_outcome` alone
tried to stand in for all three, via observation-similarity regardless of
action family — this over-triggered on completely routine exploratory
outcomes like "go to shelf 2" -> "you see nothing", flagged in the previous
pilot's report as a plausible mis-trigger):
  - `local_state_change_proxy`: did the world observably change at all
    (admissible_commands set diff, now implemented — the cheapest of the
    candidate signals identified last round — plus ALFWorld's own
    "Nothing happens." no-op string). Family-agnostic.
  - `unexpected_outcome`: restricted to STATE-CHANGING action families
    (open/close/take/put/heat/cool/clean/toggle) — legal but no local
    state change. EXPLORATORY families (go/look/examine/inspect/inventory)
    never raise this just for not finding a target; that's routine search,
    not a surprising outcome. Actions that don't parse into a known family
    fall back to the old observation-similarity heuristic.
  - `information_gain_proxy`: did this step teach the agent something new
    (novel observation or a changed admissible set) — most meaningful for
    exploratory actions, computed uniformly.

Still not implemented this round (see cost/feasibility notes from the
previous pilot's report): location-change text parsing, receptacle
open/closed keyword deltas, and true inventory tracking (ALFWorld has no
free inventory readout in `info`; would need an extra `env.step(['inventory'])`
call per check, i.e. a real extra environment/LLM turn, not a free signal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .canonical_action import canonicalize_action, clean_action_text
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


def action_similarity(action: str, other: str) -> float:
    """Repeated-action similarity used for the repeated_action signal.
    Prefers the canonical (family, object[, target]) signature -- a
    preposition/case/leading-numbering paraphrase of the same command
    scores a full 1.0 match this way, which plain Jaccard misses (e.g.
    "put lettuce 1 on countertop 1" vs "3. put lettuce 1 at countertop 1"
    share only 3 of 5 tokens). Falls back to Jaccard only when either side
    doesn't parse into a known action family (see canonical_action.py)."""
    ca, cb = canonicalize_action(action), canonicalize_action(other)
    if ca is not None and cb is not None:
        return 1.0 if ca == cb else 0.0
    return jaccard(action, other)


def consecutive_exact_action_repeat(action: str, prev_action: Optional[str]) -> float:
    """Strict adjacent-step exact-repeat check: compares ONLY the current
    action against the single immediately-preceding one (never a
    repetition_window), and requires a true exact match, not a similarity
    score. Canonical (family, object[, target]) tuple compared when both
    actions parse into a known family -- so "go to shelf 5" -> "go to
    shelf 6" is 0.0 (different object) and "1. use desklamp 1" -> "use
    desklamp 1" is 1.0 (same family+object, leading enumeration is format
    noise). Falls back to a format-cleaned (not object-stripped) raw
    string comparison only when either side doesn't parse into a known
    family. Deliberately distinct from repeated_action (window-max
    similarity) and repeated_observation (Jaccard on observation TEXT,
    which over-triggers on same-template-different-location exploration
    like "you see nothing" at two different shelves) -- this signal alone
    gates the frustration_medium intervention branch."""
    if prev_action is None:
        return 0.0
    ca, cp = canonicalize_action(action), canonicalize_action(prev_action)
    if ca is not None and cp is not None:
        return 1.0 if ca == cp else 0.0
    return 1.0 if clean_action_text(action) == clean_action_text(prev_action) else 0.0


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
    # Strict adjacent-step exact-repeat, deliberately independent of
    # repeated_action's repetition_window(=3)-max and of repeated_observation
    # entirely (see action_similarity()/consecutive_exact_action_repeat()
    # below for why repeated_observation's plain-Jaccard-on-observation-text
    # over-triggers on same-template-different-location exploration, e.g.
    # "You arrive at shelf 5...you see nothing." vs "...shelf 6...nothing.").
    # Only this signal, not repeated_action/repeated_observation, gates the
    # frustration_medium intervention branch in
    # stateful_controller.py::_select_intervention().
    consecutive_exact_action_repeat: float = 0.0
    observation_novelty: float = 0.0
    unexpected_outcome: float = 0.0
    information_gain_proxy: float = 0.0
    local_state_change_proxy: float = 0.0
    progress_signal: float = 0.0
    budget_ratio: float = 0.0

    def as_dict(self) -> dict:
        return {
            "invalid_action": self.invalid_action,
            "repeated_action": self.repeated_action,
            "repeated_observation": self.repeated_observation,
            "consecutive_exact_action_repeat": self.consecutive_exact_action_repeat,
            "observation_novelty": self.observation_novelty,
            "unexpected_outcome": self.unexpected_outcome,
            "information_gain_proxy": self.information_gain_proxy,
            "local_state_change_proxy": self.local_state_change_proxy,
            "progress_signal": self.progress_signal,
            "budget_ratio": self.budget_ratio,
        }


_STATE_CHANGING_FAMILIES = {"open", "close", "take", "put", "heat", "cool", "clean", "toggle"}
_EXPLORATORY_FAMILIES = {"go", "look", "examine", "inspect", "inventory"}


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
        admissible_after: List[str] = None,
    ) -> StepSignals:
        """Compute all 9 signals for one (action, observation) transition.

        `admissible_before` is the admissible_commands list captured BEFORE
        this action was taken (from the previous step's info, or the
        initial reset) — ALFWorld/TextWorld's admissible_commands describes
        what's legal in the state the action was chosen from, not the
        resulting state, so validity must be checked against the prior
        list. `admissible_after` (optional, defaults to `admissible_before`
        when the caller doesn't have it -- e.g. older call sites/tests) is
        the admissible_commands list returned AFTER this action, used only
        to detect whether the action's legal option set changed at all.
        `recent_actions`/`recent_observations` are the window (most recent
        last) preceding this step, already excluding the current one.
        """
        if admissible_after is None:
            admissible_after = admissible_before
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
        # Max similarity of this action to any of the last `window` actions
        # (canonical-signature match when parseable, else Jaccard fallback
        # -- see action_similarity()).
        recent_a = recent_actions[-window:] if window > 0 else []
        repeated_action = max((action_similarity(action, a) for a in recent_a), default=0.0)

        # -- consecutive_exact_action_repeat --
        # Deliberately NOT windowed (ignores `window`/repetition_window) --
        # compares only against the single immediately-preceding action.
        # See consecutive_exact_action_repeat()'s own docstring for why this
        # is a separate signal from repeated_action/repeated_observation.
        prev_action = recent_actions[-1] if recent_actions else None
        exact_repeat = consecutive_exact_action_repeat(action, prev_action)

        # -- repeated_observation --
        recent_o = recent_observations[-window:] if window > 0 else []
        repeated_observation = max((jaccard(observation, o) for o in recent_o), default=0.0)

        # -- observation_novelty --
        # Complement of repeated_observation: how different this
        # observation is from the recent window. Kept as its own named
        # signal (rather than inlining 1 - repeated_observation at call
        # sites) since it gets its own weight in state updates.
        observation_novelty = 1.0 - repeated_observation

        # -- local_state_change_proxy (heuristic, see module docstring) --
        # Did the world observably change at all? ALFWorld's own no-op
        # string "Nothing happens." is the environment's own signal that a
        # legal-looking action had no effect; admissible_commands changing
        # at all (new options appearing/disappearing, e.g. "close X" only
        # becomes available after "open X" succeeds) is a second,
        # independent free signal for the same thing, since some real
        # state changes (e.g. cleaning/heating/cooling an object) don't
        # always reword the observation as plainly as "Nothing happens."
        # would if they failed.
        state_unchanged_text = normalize_text(observation) == "nothing happens."
        admissible_set_changed = set(admissible_before) != set(admissible_after)
        local_state_change_proxy = (
            0.0 if (state_unchanged_text and not admissible_set_changed and not is_think_action)
            else (0.0 if is_think_action else 1.0)
        )

        # -- unexpected_outcome (heuristic proxy, see module docstring) --
        # Revised this round: only STATE-CHANGING actions (open/close/take/
        # put/heat/cool/clean/toggle) that are nominally legal but produce
        # no local state change count as unexpected. EXPLORATORY actions
        # (go/look/examine/inspect/inventory) not finding a target, or
        # landing on an empty receptacle, is completely routine ALFWorld
        # search behavior and must NOT raise surprise just because the
        # resulting observation resembles a recent one -- this was the
        # previous round's identified over-trigger (a plain "go to shelf 2"
        # -> "you see nothing" was flagged surprise_high purely from
        # observation-similarity, with no family distinction at all).
        # Actions that don't parse into a known family fall back to the
        # old family-agnostic heuristic rather than silently suppressing
        # unexpected_outcome for text the parser has no opinion about.
        if is_think_action:
            family = None
        else:
            canonical = canonicalize_action(action)
            family = canonical[0] if canonical is not None else None
        if is_think_action or invalid_action == 1.0:
            unexpected_outcome = 0.0
        elif family in _EXPLORATORY_FAMILIES:
            unexpected_outcome = 0.0
        elif family in _STATE_CHANGING_FAMILIES:
            unexpected_outcome = 1.0 if local_state_change_proxy == 0.0 else 0.0
        else:
            immediately_prior = recent_o[-1] if recent_o else ""
            no_effect = (
                state_unchanged_text
                or jaccard(observation, immediately_prior) >= self.config.unexpected_similarity_threshold
            )
            unexpected_outcome = 1.0 if no_effect else 0.0

        # -- information_gain_proxy (heuristic, see module docstring) --
        # Did this step teach the agent something new, distinct from
        # local_state_change_proxy (which is about the ACTION changing the
        # world) -- an observation novel relative to recent history, or a
        # newly different admissible-command set (e.g. arriving somewhere
        # that reveals new objects/actions), both count. Most relevant for
        # exploratory actions but computed uniformly.
        information_gain_proxy = (
            1.0 if (repeated_observation < self.config.novelty_progress_threshold or admissible_set_changed)
            else 0.0
        )

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
            consecutive_exact_action_repeat=exact_repeat,
            observation_novelty=observation_novelty,
            unexpected_outcome=unexpected_outcome,
            information_gain_proxy=information_gain_proxy,
            local_state_change_proxy=local_state_change_proxy,
            progress_signal=progress_signal,
            budget_ratio=budget_ratio,
        )
