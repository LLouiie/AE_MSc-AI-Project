"""MPO official ReAct prompt/ICL loader + action parser for react_reflact_anchor.

Vendored assets live under external/mpo_reflact/ (see SOURCE.md there for the
pinned commit SHA and exact upstream paths). This module loads the vendored
instruction text + ICL examples verbatim and calls MPO's own `prompt_with_icl`
(also vendored, unmodified) to build the flat first-turn prompt -- it does
not reimplement or rewrite MPO's prompt formatting.

Only the *prompt/ICL/template* and the *action parsing* behavior are mirrored
from MPO. Environment creation, task->env mapping, and success determination
continue to use this repo's own alfworld_runs_ae/environment.py (see
external/mpo_reflact/SOURCE.md for the one deliberate divergence: success
here comes from info["won"], not from MPO's own done-only check).
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_MPO_DIR = os.path.join(_REPO_ROOT, "external", "mpo_reflact")

# templates.py has no package __init__.py in the vendored copy (we took the
# single file we need, not MPO's whole `prompt` package); imported as a bare
# module by inserting its containing directory onto sys.path, same pattern
# alfworld_runs_ae/agents.py already uses for its own sys.path.append calls.
_MPO_PROMPT_DIR = os.path.join(_MPO_DIR, "prompt")
if _MPO_PROMPT_DIR not in sys.path:
    sys.path.insert(0, _MPO_PROMPT_DIR)
from templates import prompt_with_icl  # noqa: E402

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from alfworld_runs_ae.environment import PREFIXES  # noqa: E402 -- reuse, do not duplicate

MPO_SOURCE_REPO = "https://github.com/WeiminXiong/MPO"
MPO_SOURCE_COMMIT = "5529eca70ab352eb7e56533ca44cb60fbf34bef1"

_INSTRUCTION_PATH = os.path.join(_MPO_DIR, "prompt", "instructions", "alfworld_inst.txt")
_ICL_PATH = os.path.join(_MPO_DIR, "prompt", "icl_examples", "alfworld_icl.json")

with open(_INSTRUCTION_PATH) as f:
    MPO_INSTRUCTION = f.read()

with open(_ICL_PATH) as f:
    _RAW_ICL = json.load(f)

MPO_CATEGORIES = tuple(PREFIXES.keys())
if set(MPO_CATEGORIES) != set(_RAW_ICL.keys()):
    raise RuntimeError(
        "alfworld_runs_ae/environment.py::PREFIXES categories no longer "
        "match the vendored MPO ICL file's categories -- do not silently "
        f"proceed. PREFIXES keys={set(MPO_CATEGORIES)!r} "
        f"icl keys={set(_RAW_ICL.keys())!r}"
    )


def get_mpo_category(env_name: str) -> str:
    """Same prefix-matching rule as environment.py::get_task_type, but
    returns the raw MPO/ALFWorld category key (e.g. "pick_and_place")
    instead of the react_*/reflect_* prompt-file short name (e.g. "put")."""
    for prefix in PREFIXES:
        if env_name.startswith(prefix):
            return prefix
    raise ValueError(f"env_name {env_name!r} does not match any known MPO category")


def build_first_turn_prompt(category: str, task_observation: str) -> Tuple[str, List[dict]]:
    """Returns (flat_prompt_text, chat_messages) for icl_num=1 -- MPO's
    react_reflact_anchor config uses exactly one demo, the FIRST entry in
    that category's example list (raw_icl[category][0], since icl_num=1
    only ever consumes index 0 of the range it loops over).

    `chat_messages` is the multi-turn role list MPO itself builds for the
    alternative icl_format='conversation'; the published-anchor config uses
    icl_format='first' (the whole ICL block folded into ONE user turn), so
    `flat_prompt_text` is what actually gets sent as the first user message
    -- `chat_messages` is returned too so the road not taken stays
    inspectable rather than silently discarded.
    """
    demo_list_for_category = _RAW_ICL[category]
    flat_prompt, chat_messages = prompt_with_icl(
        instruction=MPO_INSTRUCTION,
        raw_icl=demo_list_for_category,
        cur_task=task_observation,
        icl_num=1,
        workflow=None,
    )
    return flat_prompt, chat_messages


# Mirrors external/mpo_reflact/envs/alfworld_env_reference.py::parse_action
# exactly (same regex, same put-normalization). MPO's own version calls
# `re.findall(...)[0]` (raises IndexError on no match) and later `assert
# action is not None` -- we return None instead so the caller records a
# parse failure per this repo's convention (never crash the episode loop or
# fabricate an action), matching alfworld_runs_ae/output_parser.py's stance.
_ACTION_RE = re.compile(r"Action:\s?(.*)", re.DOTALL)
_PUT_RE = re.compile(r"put\s+(.*)\s+[io]n\s+(.*)")


def parse_mpo_action(llm_output: str) -> Optional[str]:
    text = llm_output.strip()
    matches = _ACTION_RE.findall(text)
    if not matches:
        return None
    action = matches[0].strip()
    if not action:
        return None
    put_match = _PUT_RE.findall(action)
    if put_match:
        action = f"put {put_match[0][0]} in/on {put_match[0][1]}"
    return action
