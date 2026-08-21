"""Canonical action signatures for repetition/stuck detection.

Parses common ALFWorld action-family verbs into a normalized structured
tuple (family, object[, target]) so that surface-level paraphrases of the
SAME action ("put lettuce on countertop 1" / "put lettuce in countertop 1"
/ "3. put lettuce at countertop 1") are recognized as repeats even though
they are not character-identical. This fixes a paraphrase-evasion gap
found in the previous pilot (pick_cool_then_place_in_recep-Lettuce-
CounterTop-10, steps 15-50: the model cycled through on/in/at/into
prepositions and 1./2./3.../33. leading numbering for 35 straight steps
without the plain exact-match repeated-action check ever firing again).

Canonicalization is used ONLY for the repeated-action signal, stuck
detection, and logging/analysis -- never to auto-correct or substitute
the action actually executed against the environment (same constraint as
alfworld_runs_ae/output_parser.py's handling of raw generations). The
actual put->move execution-time rewrite lives in the independent
alfworld_runs_ae/alfworld_action_normalize.py compatibility adapter, not
here.

If an action can't be reliably parsed into one of the known families,
canonicalize_action() returns None and callers fall back to the existing
Jaccard token-overlap similarity (see signals.py::_action_similarity).

"put"/"move" equivalence (added alongside the put->move execution fix):
the model's prompt teaches it to say "put {obj} in/on {recep}", but the
ALFWorld data actually in use only accepts "move {obj} to {recep}" --
the environment wrapper rewrites the executed string, but the model can
still spontaneously say "move ..." on its own (observed in existing run
logs) or keep repeating "put ..." paraphrases. Both verbs describe the
same placement action, so "move X to Y" canonicalizes to the exact same
("put", obj, target) signature as "put X in/on Y" -- this is purely a
repetition/stuck-detection identity, it does not change what gets sent to
env.step() (see canonical_action.py's own callers: signals.py only).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_LEADING_ENUM_RE = re.compile(r"^\s*\d+[\.\)]\s*")
_WHITESPACE_RE = re.compile(r"\s+")
_PREP = r"(?:in|on|at|into|inside|onto)(?:/(?:in|on|at|into|inside|onto))*"

# family -> regex with named groups obj/target, tried in order (most
# specific multi-argument families first so e.g. "put" never falls through
# to a looser single-argument pattern).
_PATTERNS = [
    ("take", re.compile(rf"^take\s+(?P<obj>.+?)\s+from\s+(?P<target>.+)$")),
    ("put", re.compile(rf"^put\s+(?P<obj>.+?)\s+{_PREP}\s+(?P<target>.+)$")),
    # "move X to Y" is the same placement action as "put X in/on Y" (see
    # module docstring) -- mapped to the "put" family label below (not a
    # separate "move" family) so existing callers/tests that key off the
    # literal family string "put" keep working unchanged.
    ("move", re.compile(r"^move\s+(?P<obj>.+?)\s+to\s+(?P<target>.+)$")),
    ("heat", re.compile(r"^heat\s+(?P<obj>.+?)\s+with\s+(?P<target>.+)$")),
    ("cool", re.compile(r"^cool\s+(?P<obj>.+?)\s+with\s+(?P<target>.+)$")),
    ("clean", re.compile(r"^clean\s+(?P<obj>.+?)\s+with\s+(?P<target>.+)$")),
    ("go", re.compile(r"^go\s+to\s+(?P<obj>.+)$")),
    ("open", re.compile(r"^open\s+(?P<obj>.+)$")),
    ("close", re.compile(r"^close\s+(?P<obj>.+)$")),
    # ALFWorld's real admissible command for this family is "use X"
    # (e.g. "use desklamp 1"); "toggle X" is accepted as a synonym since
    # that's the family name the spec uses and some agents say it that way.
    ("toggle", re.compile(r"^(?:toggle|use)\s+(?P<obj>.+)$")),
    ("look", re.compile(r"^look(?:\s+.*)?$")),
    ("inventory", re.compile(r"^inventory$")),
    ("think", re.compile(r"^think:\s*(?P<obj>.*)$")),
]


def _normalize_object(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.lower().strip().rstrip(".")
    return _WHITESPACE_RE.sub(" ", text)


def clean_action_text(action: str) -> str:
    """Format-noise-only cleanup for callers that need a raw-string
    fallback comparison when an action doesn't parse into a known family
    (e.g. signals.py's consecutive_exact_action_repeat): lowercases,
    strips a leading enumeration artifact ("1. ", "2)") and a trailing
    period, and collapses whitespace. Purely additive -- does not change
    canonicalize_action()'s own (already-working) text handling below.
    Never touches object/container indices -- "shelf 5" and "shelf 6" must
    stay distinguishable after cleanup."""
    if not action:
        return ""
    text = _LEADING_ENUM_RE.sub("", action.strip())
    text = _WHITESPACE_RE.sub(" ", text.lower().strip())
    return text.rstrip(".")


def canonicalize_action(action: str) -> Optional[Tuple[str, ...]]:
    """Returns e.g. ("put", "lettuce 1", "countertop 1") for any of "put
    lettuce 1 on/in/at/into countertop 1", or None if the action doesn't
    match a known family. Object indices ("countertop 1" vs "countertop
    2") are kept -- they're a real semantic distinction, not formatting
    noise -- only the leading enumeration artifact ("1. ", "2)") and
    preposition/case/whitespace variation are normalized away."""
    if not action:
        return None
    text = _LEADING_ENUM_RE.sub("", action.strip())
    text = _WHITESPACE_RE.sub(" ", text.lower().strip())
    if not text:
        return None

    for family, pattern in _PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        groups = m.groupdict()
        if family in ("look", "inventory"):
            return (family,)
        if family == "think":
            return ("think", _normalize_object(groups.get("obj")) or "")
        if family == "move":
            family = "put"  # put/move are the same canonical placement family
        obj = _normalize_object(groups.get("obj"))
        target = _normalize_object(groups.get("target")) if "target" in groups else None
        return (family, obj, target) if target is not None else (family, obj)
    return None
