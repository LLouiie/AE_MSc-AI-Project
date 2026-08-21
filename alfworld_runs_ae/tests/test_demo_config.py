"""Tests for the configurable ALFWorld ICL demonstration selection
(demo_config.py + agents.py::_build_base_prompt's demo_config param), added
2026-08-07 to replace the old hardcoded two-shot base prompt
(`_PROMPTS[f'react_{task_type}_1'] + _PROMPTS[f'react_{task_type}_0']`,
"Here are two examples.") with an explicit, file-driven DemoConfig.

Covers: one-shot shows only the configured example and none of the other
(unconfigured) ones; singular wording for one-shot; all six ALFWorld task
types resolve the correct example; the legacy two-shot default is
byte-identical to the pre-change hardcoded prompt; illegal configs raise
immediately; ReAct and AE (ae_full) build the identical initial base prompt
for the same task under the same demo_config (they share one code path via
ALFWorldAgent, so this is really a "no baseline-specific special-casing
exists" check).

Plain assert-based, same convention as the other test files in this repo.
Run directly:
    python3 alfworld_runs_ae/tests/test_demo_config.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "hotpotqa_runs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # alfworld_runs_ae

import json  # noqa: E402
import tempfile  # noqa: E402

import agents  # noqa: E402
from agents import _build_base_prompt, ALFWorldAgent  # noqa: E402
from demo_config import DemoConfig, DEFAULT_DEMO_CONFIG, load_demo_config, demo_count_phrase  # noqa: E402

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


MODEL = "Qwen/Qwen3-8B"
GOAL = "put a mug in the sink"

# task_type -> {index: unique "Your task is to: ..." fingerprint substring}
# (see alfworld_runs/prompts/alfworld_3prompts.json; extracted once via a
# read-only script, not hardcoded from memory of the file's prose).
_FINGERPRINTS = {
    "put":     {0: "put some spraybottle on toilet",     1: "find some apple and put it in sidetable",  2: "put a soapbottle in garbagecan"},
    "clean":   {0: "put a clean lettuce in diningtable",  1: "clean some apple and put it in sidetable", 2: "clean some soapbar and put it in toilet"},
    "heat":    {0: "heat some egg and put it in diningtable", 1: "put a hot apple in fridge",             2: "heat some bread and put it in countertop"},
    "cool":    {0: "cool some pan and put it in stoveburner", 1: "put a cool mug in shelf",                2: "cool some potato and put it in diningtable"},
    "examine": {0: "look at bowl under the desklamp",     1: "examine the pen with the desklamp",         2: "look at statue under the desklamp"},
    "puttwo":  {0: "put two creditcard in dresser",       1: "put two cellphone in sofa",                 2: "put two saltshaker in drawer"},
}
ALL_SIX_TYPES = list(_FINGERPRINTS.keys())
ONE_SHOT = DemoConfig(num_demos=1, demo_indices=(0,))
TWO_SHOT_LEGACY = DemoConfig(num_demos=2, demo_indices=(1, 0))


# ── 1. one-shot: exactly the configured example appears, no other ──────────
def test_one_shot_contains_only_the_configured_example():
    for t in ALL_SIX_TYPES:
        prompt = _build_base_prompt(t, GOAL, [], "", ONE_SHOT)
        check(f"1a. [{t}] index-0 fingerprint present",
              _FINGERPRINTS[t][0] in prompt, prompt[:200])
        for other in (1, 2):
            check(f"1b. [{t}] index-{other} fingerprint absent",
                  _FINGERPRINTS[t][other] not in prompt)


# ── 2. one-shot uses singular wording, not "Here are two examples." ────────
def test_one_shot_uses_singular_wording():
    for t in ALL_SIX_TYPES:
        prompt = _build_base_prompt(t, GOAL, [], "", ONE_SHOT)
        check(f"2a. [{t}] singular phrase present", "Here is one example." in prompt)
        check(f"2b. [{t}] plural phrase absent", "Here are two examples." not in prompt)
        check(f"2c. [{t}] no leftover 'examples' (plural) anywhere",
              "examples" not in prompt.split("\n", 1)[0])


# ── 3. all six task types resolve the correct (index 0) example ────────────
def test_all_six_task_types_select_correct_example():
    for t in ALL_SIX_TYPES:
        prompt = _build_base_prompt(t, GOAL, [], "", ONE_SHOT)
        check(f"3. [{t}] resolves react_{t}_0's own task, not another type's",
              _FINGERPRINTS[t][0] in prompt)
        # cross-type isolation: none of the OTHER five types' index-0 goal
        # lines leak into this one's prompt.
        for other_t in ALL_SIX_TYPES:
            if other_t == t:
                continue
            check(f"3b. [{t}] does not contain [{other_t}]'s index-0 fingerprint",
                  _FINGERPRINTS[other_t][0] not in prompt)


# ── 4. legacy two-shot mode is byte-identical to the pre-change hardcoded
#       prompt (both via explicit TWO_SHOT_LEGACY and via the default when
#       demo_config is omitted entirely) ───────────────────────────────────
def test_two_shot_legacy_byte_identical_to_original_hardcoded_prompt():
    for t in ALL_SIX_TYPES:
        with open(agents.PROMPTS_FILE) as f:
            raw_prompts = json.load(f)
        original_hardcoded = (
            f"Interact with a household to solve a task. Here are two examples.\n"
            f"{raw_prompts[f'react_{t}_1'] + raw_prompts[f'react_{t}_0']}"
            f"\nHere is the task:\n{GOAL}\n{agents._FORMAT_INSTRUCTION}"
        )
        via_explicit_config = _build_base_prompt(t, GOAL, [], "", TWO_SHOT_LEGACY)
        via_omitted_config = _build_base_prompt(t, GOAL, [], "")  # demo_config=None default
        check(f"4a. [{t}] explicit two-shot config byte-identical to original hardcoded prompt",
              via_explicit_config == original_hardcoded)
        check(f"4b. [{t}] omitted demo_config (default) byte-identical to original hardcoded prompt",
              via_omitted_config == original_hardcoded)
        check(f"4c. [{t}] explicit two-shot config == omitted-config default",
              via_explicit_config == via_omitted_config)
    check("4d. DEFAULT_DEMO_CONFIG equals the explicit legacy two-shot config",
          DEFAULT_DEMO_CONFIG == TWO_SHOT_LEGACY, DEFAULT_DEMO_CONFIG)
    check("4e. demo_count_phrase(2) is exactly the original hardcoded phrase",
          demo_count_phrase(2) == "Here are two examples.", demo_count_phrase(2))


# ── 5. illegal configs raise immediately ────────────────────────────────────
def test_illegal_configs_raise_immediately():
    def _raises(fn):
        try:
            fn()
            return False
        except (ValueError, KeyError, FileNotFoundError):
            return True

    check("5a. num_demos=0 raises", _raises(lambda: DemoConfig(num_demos=0, demo_indices=())))
    check("5b. num_demos negative raises", _raises(lambda: DemoConfig(num_demos=-1, demo_indices=())))
    check("5c. demo_indices length mismatch (too few) raises",
          _raises(lambda: DemoConfig(num_demos=2, demo_indices=(0,))))
    check("5d. demo_indices length mismatch (too many) raises",
          _raises(lambda: DemoConfig(num_demos=1, demo_indices=(0, 1))))
    check("5e. duplicate demo_indices raises",
          _raises(lambda: DemoConfig(num_demos=2, demo_indices=(0, 0))))
    check("5f. negative demo index raises",
          _raises(lambda: DemoConfig(num_demos=1, demo_indices=(-1,))))
    check("5g. out-of-range demo index (no react_put_9) raises when building a prompt",
          _raises(lambda: _build_base_prompt("put", GOAL, [], "", DemoConfig(num_demos=1, demo_indices=(9,)))))
    check("5h. load_demo_config with a nonexistent path raises FileNotFoundError (no silent fallback)",
          _raises(lambda: load_demo_config("/nonexistent/path/does_not_exist.yaml")))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("num_demos: 2\ndemo_indices: [0]\n")  # length mismatch inside the YAML itself
        bad_path = f.name
    try:
        check("5i. load_demo_config with an internally-inconsistent yaml raises",
              _raises(lambda: load_demo_config(bad_path)))
    finally:
        os.remove(bad_path)


# ── 6. ReAct and AE build the byte-identical initial base prompt ───────────
class FakeActLLM:
    def __init__(self):
        self.model = MODEL
        self.last_meta = None
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        self.last_meta = {"finish_reason": "stop", "was_truncated": False, "usage": "unavailable"}
        return "Thought: t\nAction: go to shelf 1"


class FakeEnvOneStepThenDone:
    def reset(self):
        ob = "-= Welcome =-\n\nYou are in a room.\n\nYour task is to: put a mug in the sink."
        return [ob], {"admissible_commands": [["go to shelf 1"]], "won": [False]}

    def step(self, actions):
        return (["You see nothing."], [0], [True],
                {"admissible_commands": [["go to shelf 1"]], "won": [True]})

    def close(self):
        pass


def test_react_and_ae_build_identical_initial_base_prompt():
    from ae.controllers.stateful_controller import StatefulController
    from ae.controllers.config import AEConfig

    for demo_cfg in (None, ONE_SHOT, TWO_SHOT_LEGACY):
        react_agent = ALFWorldAgent(FakeActLLM(), termination_policy="fixed_horizon", demo_config=demo_cfg)
        ae_agent = ALFWorldAgent(FakeActLLM(), controller=StatefulController(AEConfig()),
                                  termination_policy="fixed_horizon", demo_config=demo_cfg)

        react_agent.run(FakeEnvOneStepThenDone(), GOAL, "put", to_print=False, task_id="x")
        ae_agent.run(FakeEnvOneStepThenDone(), GOAL, "put", to_print=False, task_id="x")

        react_base = _build_base_prompt("put", GOAL, [], "", react_agent.demo_config)
        ae_base = _build_base_prompt("put", GOAL, [], "", ae_agent.demo_config)
        label = "default" if demo_cfg is None else ("one-shot" if demo_cfg.num_demos == 1 else "two-shot")
        check(f"6. [{label}] ReAct and AE initial base prompt byte-identical",
              react_base == ae_base, (react_base[:100], ae_base[:100]))


if __name__ == "__main__":
    test_one_shot_contains_only_the_configured_example()
    test_one_shot_uses_singular_wording()
    test_all_six_task_types_select_correct_example()
    test_two_shot_legacy_byte_identical_to_original_hardcoded_prompt()
    test_illegal_configs_raise_immediately()
    test_react_and_ae_build_identical_initial_base_prompt()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
