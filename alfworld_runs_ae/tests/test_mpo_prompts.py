"""Tests for the vendored-MPO prompt/ICL loader and action parser
(alfworld_runs_ae/mpo_prompts.py), used only by react_reflact_anchor.

Plain assert-based, same convention as test_output_parser.py.
Run directly:
    python3 alfworld_runs_ae/tests/test_mpo_prompts.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # alfworld_runs_ae

from environment import PREFIXES, get_all_tasks  # noqa: E402
from mpo_prompts import (  # noqa: E402
    MPO_CATEGORIES, build_first_turn_prompt, get_mpo_category, parse_mpo_action,
)

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def test_categories_match_prefixes():
    check("1. MPO_CATEGORIES is exactly PREFIXES' keys",
          set(MPO_CATEGORIES) == set(PREFIXES.keys()),
          f"{sorted(MPO_CATEGORIES)} vs {sorted(PREFIXES.keys())}")


def test_get_mpo_category_matches_env_name_prefix():
    check("2. pick_and_place prefix resolves",
          get_mpo_category("pick_and_place_simple-SaltShaker-None-Drawer-10") == "pick_and_place")
    check("3. pick_cool_then_place prefix resolves",
          get_mpo_category("pick_cool_then_place_in_recep-Pan-None-CounterTop-10") == "pick_cool_then_place")
    check("4. look_at_obj prefix resolves",
          get_mpo_category("look_at_obj_in_light-Pencil-None-DeskLamp-308") == "look_at_obj")

    raised = False
    try:
        get_mpo_category("totally_unknown_prefix-Foo-1")
    except ValueError:
        raised = True
    check("5. unknown prefix raises ValueError, does not guess", raised)


def test_all_134_tasks_map_to_a_known_category():
    tasks = get_all_tasks()
    check("6. get_all_tasks returns 134 tasks", len(tasks) == 134, f"got {len(tasks)}")
    failures = []
    for t in tasks:
        env_name = t.get("env_name") or t["gamefile"].split("/")[-3]
        try:
            get_mpo_category(env_name)
        except ValueError:
            failures.append(env_name)
    check("7. every one of the 134 tasks maps to a known MPO category",
          not failures, f"unmapped: {failures[:5]}")


def test_exactly_one_demo_loaded_icl_num_1():
    flat_prompt, chat_messages = build_first_turn_prompt(
        "pick_and_place", "Your task is to: put a spraybottle in toilet.",
    )
    check("8. flat prompt contains the instruction",
          "intelligent agent in a household environment" in flat_prompt)
    check("9. flat prompt contains exactly one example (icl_num=1 wording)",
          "Here is an example." in flat_prompt and "Here are" not in flat_prompt)
    check("10. flat prompt does NOT contain a second-example marker",
          "Example task 2" not in flat_prompt)
    check("11. flat prompt ends with the current task text",
          flat_prompt.rstrip().endswith("put a spraybottle in toilet."))
    check("12. chat_messages alternates starting with user, ending with the task as a user turn",
          chat_messages[0]["role"] == "user" and chat_messages[-1]["role"] == "user")


def test_parse_mpo_action_basic():
    p = parse_mpo_action("Thought: I should look.\nAction: go to cabinet 1")
    check("13. extracts action after Action: label", p == "go to cabinet 1")


def test_parse_mpo_action_put_normalization():
    p_on = parse_mpo_action("Action: put mug 1 on countertop 1")
    p_in = parse_mpo_action("Action: put mug 1 in countertop 1")
    check("14. 'put X on Y' normalized to 'put X in/on Y'", p_on == "put mug 1 in/on countertop 1")
    check("15. 'put X in Y' normalized to 'put X in/on Y'", p_in == "put mug 1 in/on countertop 1")


def test_parse_mpo_action_no_label_returns_none():
    p = parse_mpo_action("I am thinking about what to do next, no label at all.")
    check("16. no 'Action:' label anywhere -> None (never fabricates an action)", p is None)


def test_parse_mpo_action_dotall_takes_everything_after_first_label():
    # Mirrors MPO's own re.findall(r"Action:\s?(.*)", text, re.DOTALL)[0]
    # behavior exactly: greedy + DOTALL means it swallows to end of string,
    # even across newlines, even if the text looks like it has "more" after.
    p = parse_mpo_action("Action: go to cabinet 1\nsome trailing hallucinated text")
    check("17. DOTALL greedy match swallows trailing text (matches MPO exactly, not our own parser's stricter behavior)",
          p == "go to cabinet 1\nsome trailing hallucinated text", repr(p))


if __name__ == "__main__":
    test_categories_match_prefixes()
    test_get_mpo_category_matches_env_name_prefix()
    test_all_134_tasks_map_to_a_known_category()
    test_exactly_one_demo_loaded_icl_num_1()
    test_parse_mpo_action_basic()
    test_parse_mpo_action_put_normalization()
    test_parse_mpo_action_no_label_returns_none()
    test_parse_mpo_action_dotall_takes_everything_after_first_label()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
