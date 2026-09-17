"""Shared Thought/Action output parser.

Used identically by every baseline that goes through
alfworld_runs_ae/agents.py::ALFWorldAgent (react, reflexion, ae_full) --
there is exactly one parser, not an AE-specific one. The environment
only ever receives `parsed_action`; raw_generation is never handed to
env.step() directly.

Three accepted forms, in priority order:
  1. Labeled two-line "Thought: ...\\nAction: ..." (the format the prompt
     now explicitly asks for).
  2. Labeled "Action: ..." only (no Thought: line).
  3. Unlabeled bare line -- the pre-existing alfworld_3prompts.json
     few-shot style ("> think: ..." / "> go to shelf 1"), kept for
     backward compatibility since the few-shot examples themselves were
     not rewritten this round.

No auto-correction of the action's SEMANTICS: an unparseable generation
is reported as a parse failure with parsed_action=None, never silently
replaced by a guessed or nearest-admissible command. Stripping a leading
legacy "> " cursor cue / leading enumeration ("1.", "2)") / surrounding
whitespace off an already-labeled or already-validated action string is
NOT a semantic correction -- it is the SAME action, just with formatting
noise the model copied from the few-shot examples removed (bug found
2026-07-25: "Action: > open fridge 1" was being kept verbatim, with the
stray "> " making a valid action register as not-admissible downstream).

Bug #2 fixed the same day: the unlabeled bare-line fallback (form 3)
was accepting ANY first non-empty line as a literal action, including a
full sentence of free-form reasoning with no relation to the ALFWorld
action grammar (e.g. "Okay, let me try to figure this out..."). Form 3
now only succeeds if the cleaned line either (a) matches one of the
known ALFWorld action families (canonical_action.canonicalize_action),
or (b) equals one of the environment's current admissible_commands
(passed in by the caller, normalized case/whitespace-insensitively).
Anything else is now a parse failure (parse_failure_reason=
"unrecognised_bare_text"), not a silently-executed hallucinated action.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ae.controllers.canonical_action import canonicalize_action, clean_action_text as canonical_clean_action_text  # noqa: E402

_THOUGHT_RE = re.compile(r"^\s*thought\s*:\s*(.*)$", re.IGNORECASE)
_ACTION_RE = re.compile(r"^\s*action\s*:\s*(.*)$", re.IGNORECASE)

# Legacy few-shot cursor cue ("> action") and leading enumeration
# ("1. action" / "2) action") -- pure formatting noise the model copies
# from the alfworld_3prompts.json examples, stripped in either order and
# repeatedly (e.g. "1. > open fridge 1") until nothing more matches.
_LEGACY_CARET_RE = re.compile(r"^\s*>+\s*")
_LEADING_ENUM_RE = re.compile(r"^\s*\d+[\.\)]\s*")


def _clean_action_text(text: str) -> str:
    text = text.strip()
    prev = None
    while prev != text:
        prev = text
        text = _LEGACY_CARET_RE.sub("", text)
        text = _LEADING_ENUM_RE.sub("", text)
        text = text.strip()
    return canonical_clean_action_text(text)


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _bare_action_is_recognized(cleaned: str, admissible_commands: Optional[List[str]]) -> bool:
    if not cleaned:
        return False
    if canonicalize_action(cleaned) is not None:
        return True
    if admissible_commands:
        target = _normalize_for_match(cleaned)
        return any(_normalize_for_match(a) == target for a in admissible_commands)
    return False


@dataclass
class ParsedOutput:
    raw_generation: str
    parsed_thought: Optional[str]
    parsed_action: Optional[str]
    parse_success: bool
    parse_failure_reason: Optional[str]

    def as_dict(self) -> dict:
        return {
            "raw_generation": self.raw_generation,
            "parsed_thought": self.parsed_thought,
            "parsed_action": self.parsed_action,
            "parse_success": self.parse_success,
            "parse_failure_reason": self.parse_failure_reason,
        }


def parse_agent_output(
    raw_generation: str, admissible_commands: Optional[List[str]] = None
) -> ParsedOutput:
    text = raw_generation.strip()
    if not text:
        return ParsedOutput(raw_generation, None, None, False, "empty_generation")

    lines = [l for l in text.splitlines() if l.strip()]

    thought = None
    action = None
    for line in lines:
        m = _THOUGHT_RE.match(line)
        if m and thought is None:
            thought = m.group(1).strip()
            continue
        m = _ACTION_RE.match(line)
        if m and action is None:
            action = m.group(1).strip()

    if action:
        # Explicitly labeled: trusted as an action (no grammar/admissible
        # gate), but stripped of legacy formatting noise -- see module
        # docstring bug #1.
        cleaned = _clean_action_text(action)
        if cleaned:
            return ParsedOutput(raw_generation, thought, cleaned, True, None)

    if thought is not None:
        # A Thought: line exists (often because generation was cut off by
        # the stop sequence / max_tokens before an Action: line appeared)
        # but no Action: line was ever found. Do not guess.
        return ParsedOutput(raw_generation, thought, None, False, "truncated_no_action")

    # No "Thought:"/"Action:" labels anywhere: fall back to the unlabeled
    # bare-line form -- but only accept it if it's recognizably an
    # ALFWorld action (see module docstring bug #2), never a raw sentence
    # of free-form reasoning.
    first_line = _clean_action_text(lines[0])
    if first_line and _bare_action_is_recognized(first_line, admissible_commands):
        return ParsedOutput(raw_generation, None, first_line, True, None)

    return ParsedOutput(raw_generation, None, None, False, "unrecognised_bare_text")
