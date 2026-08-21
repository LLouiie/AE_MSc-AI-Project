"""ALFWorld action-string compatibility adapter: put -> move.

Ported from the independently-verified fix in the sibling ReflAct
reproduction project (~/projects/react-reflact-audit/common/
alfworld_action_normalize.py, 27 passing tests, validated against a live
gate + full-134 rerun there) -- copied into THIS repo as a self-contained
module (no cross-repo import, no sys.path reach-out) per the instruction
not to create a runtime dependency on another project's checkout.

Root cause (identical in both projects, confirmed against this repo's own
audit_reports/AE_CONTROLLER_AUDIT_2026-07-31.md and a fresh read-only log
scan): alfworld_3prompts.json's few-shot examples teach the model to say
"put {obj} in/on {recep}", but the ALFWorld data release actually in use
here ($ALFWORLD_DATA/json_2.1.1) compiles every game file's PutObject
admissible-command template as "move {o} to {r}" instead. A literal
"put ... in/on ..." string is therefore never recognized by env.step() and
silently returns "Nothing happens.", regardless of whether the object/
receptacle choice was otherwise correct (confirmed empirically: 1612/1612
real "put ..." actions across this repo's existing full react/reflexion
run logs returned "Nothing happens.", while the rare spontaneous "move ...
to ..." actions the model occasionally emits on its own succeed when the
receptacle index is right).

This module is the ONLY new logic introduced by the put->move fix. It is a
pure string-rewrite of the action immediately before it is handed to
env.step(). It does not touch prompts, ICL examples, the parser upstream
of this point (alfworld_runs_ae/output_parser.py), history/conversation
formatting, or any generation parameter -- callers apply this only to the
single already-parsed action string about to be executed, never to raw LLM
output, Thought text, Observation text, or conversation history, all of
which must keep showing the model's literal original wording.
"""
import re

_PUT_RE = re.compile(
    r"put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)",
    flags=re.IGNORECASE,
)


def normalize_alfworld_action(action):
    """Rewrite a complete 'put X in/on|in|on Y' action to 'move X to Y'.

    Returns (executed_action, normalization) where `normalization` is the
    string "put_to_move" if a rewrite happened, else None and
    `executed_action == action.strip()` unchanged. `action=None` passes
    through as `(None, None)` so callers that only normalize when a parsed
    action actually exists (parse failures yield parsed_action=None) don't
    need their own None-guard duplicated at every call site.
    """
    if action is None:
        return None, None

    raw_action = action.strip()

    match = _PUT_RE.fullmatch(raw_action)
    if not match:
        return raw_action, None

    obj = match.group(1).strip()
    receptacle = match.group(2).strip()

    return f"move {obj} to {receptacle}", "put_to_move"
