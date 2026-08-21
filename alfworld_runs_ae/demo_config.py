"""Configurable ALFWorld ICL demonstration (few-shot example) selection.

Replaces the old hardcoded two-shot base prompt
(`_PROMPTS[f'react_{task_type}_1'] + _PROMPTS[f'react_{task_type}_0']`,
"Here are two examples.") with an explicit, file-driven DemoConfig that
`alfworld_runs_ae/agents.py::_build_base_prompt()` reads instead of
hardcoding indices. ReAct, Reflexion, and AE (ae_full) all share this same
config object and code path (all three route through `ALFWorldAgent`/
`ALFWorldReflectAgent` in agents.py) -- there is no separate demo-selection
logic per baseline. The controller's REFLECT/REPLAN/VERIFY intervention
directive (ae/controllers/intervention_renderer.py) is spliced into the
prompt after `_build_base_prompt()` returns and never touches this module
or counts as a demonstration.

Default (no config path given) reproduces the original hardcoded two-shot
behavior byte-for-byte: `DemoConfig()` == num_demos=2, demo_indices=(1, 0)
-- so every existing run/script that never passes `--demo-config` continues
to build the exact same prompt it always has, and existing results/logs
remain reproducible without modification.

One-shot demo selection (2026-08-07): the user chose `demo_indices=[0]`
uniformly across all six ALFWorld task types (configs/demos/one_shot_v1.yaml).
Evidentiary basis, confirmed by verbatim comparison against ReflAct
(arXiv:2505.15182v2) Appendix K, Figures 15-18, extracted from the paper's
own PDF (ReflAct has no official code release; the PDF text is the only
primary source):
  - "put" (pick_and_place) and "clean" (pick_clean_then_place): CONFIRMED --
    the paper's one-shot ICL example text (room description, goal line,
    action/observation sequence) is the same underlying task/trajectory as
    `react_put_0` / `react_clean_0` in alfworld_3prompts.json, verbatim
    except for the old "> think:"/"OK." cursor style vs. the paper's typeset
    "Thought:"/"Action:" style. `react_put_1` / `react_clean_1` are a
    different task entirely (apple/sidetable) and do NOT match.
  - "heat", "cool", "examine", "puttwo": NOT independently confirmed -- the
    paper's Appendix K only publishes full one-shot examples for the two
    types above; the other four are not printed anywhere in the 33-page
    PDF (Section L's "examine" case study is a live-run trajectory excerpt,
    not the fixed ICL demo text, and heat/cool/puttwo have no excerpt at
    all). Index 0 for these four is a PRE-REGISTERED EXTRAPOLATION from the
    two confirmed types' consistent pattern, explicitly authorized by the
    user with this evidentiary distinction on record -- it is not itself
    verified against ReflAct's actual implementation for those four types.
This selection was fixed BEFORE any GPU experiment and must not be revised
by comparing per-index scores after the fact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import yaml

_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


@dataclass(frozen=True)
class DemoConfig:
    """Fixed, pre-registered demonstration selection for the ALFWorld base
    prompt. `demo_indices` are looked up as `react_{task_type}_{idx}` in
    alfworld_3prompts.json, in the given order (index 0 of demo_indices is
    concatenated first)."""

    num_demos: int = 2
    demo_indices: Tuple[int, ...] = (1, 0)

    def __post_init__(self):
        if self.num_demos < 1:
            raise ValueError(f"num_demos must be >= 1, got {self.num_demos}")
        if len(self.demo_indices) != self.num_demos:
            raise ValueError(
                f"demo_indices has {len(self.demo_indices)} entries but "
                f"num_demos={self.num_demos} -- they must match exactly"
            )
        if len(set(self.demo_indices)) != len(self.demo_indices):
            raise ValueError(f"demo_indices must not contain duplicates: {self.demo_indices}")
        for idx in self.demo_indices:
            if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
                raise ValueError(
                    f"demo_indices entries must be non-negative ints, got {idx!r}"
                )


DEFAULT_DEMO_CONFIG = DemoConfig()  # byte-identical to the original hardcoded two-shot prompt


def demo_count_phrase(num_demos: int) -> str:
    """"Here is one example." for num_demos==1, "Here are two examples."
    (byte-identical to the original hardcoded string) for num_demos==2,
    and the natural generalization for any other count."""
    word = _NUM_WORDS.get(num_demos, str(num_demos))
    if num_demos == 1:
        return f"Here is {word} example."
    return f"Here are {word} examples."


def load_demo_config(path: str | None) -> DemoConfig:
    """Load a DemoConfig from a YAML file (see configs/demos/*.yaml for
    examples). `path=None` (the default everywhere this is called) returns
    DEFAULT_DEMO_CONFIG unchanged -- no existing caller's behavior changes
    unless it explicitly opts in with a path. Unlike
    ae/controllers/config.py::load_config, a path that IS given but does
    not exist is a hard error, not a silent fallback to the default: which
    demo set ran is research-integrity-sensitive, so a typo'd or missing
    path must never silently substitute a different prompt than the one
    the caller asked for."""
    if not path:
        return DEFAULT_DEMO_CONFIG
    if not os.path.exists(path):
        raise FileNotFoundError(f"demo config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    num_demos = raw.get("num_demos", DEFAULT_DEMO_CONFIG.num_demos)
    demo_indices = raw.get("demo_indices", list(DEFAULT_DEMO_CONFIG.demo_indices))
    return DemoConfig(num_demos=num_demos, demo_indices=tuple(demo_indices))
