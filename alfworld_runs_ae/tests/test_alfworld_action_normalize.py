"""Tests for alfworld_action_normalize.normalize_alfworld_action.

Ported from the independently-verified 27-case suite in
~/projects/react-reflact-audit/common/test_alfworld_action_normalize.py
(copied into local assertions, not imported cross-repo), plus additional
coverage for None-input safety and the object/receptacle-index-preserving
guarantee this repo's put->move fix depends on.

Plain assert-based, same convention as ae/controllers/tests/test_ae_controller.py.
Run directly:
    python3 alfworld_runs_ae/tests/test_alfworld_action_normalize.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # alfworld_runs_ae

from alfworld_action_normalize import normalize_alfworld_action  # noqa: E402

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


# ── 1/2/3/4. put-in / put-on / numbered object+receptacle / case-insensitive
CASES_SHOULD_CONVERT = [
    ("put mug 1 in fridge 1", "move mug 1 to fridge 1"),
    ("put mug 1 on countertop 1", "move mug 1 to countertop 1"),
    ("put mug 1 in/on coffeemachine 1", "move mug 1 to coffeemachine 1"),
    ("PUT mug 1 IN fridge 1", "move mug 1 to fridge 1"),
]

# ── 4. different ALFWorld object/receptacle names ────────────────────────
EXTRA_CONVERT = [
    ("put vase 1 in/on safe 1", "move vase 1 to safe 1"),
    ("put pan 2 in/on stoveburner 1", "move pan 2 to stoveburner 1"),
    ("put onion 1 on countertop 1", "move onion 1 to countertop 1"),
    ("put creditcard 1 in/on dresser 1", "move creditcard 1 to dresser 1"),
    ("put soapbar 2 in/on garbagecan 10", "move soapbar 2 to garbagecan 10"),
]

# ── 5. move already-phrased, and other action families pass through unchanged
CASES_SHOULD_PASS_THROUGH = [
    "go to fridge 1",
    "take mug 1 from countertop 1",
    "move mug 1 to fridge 1",
]

EXTRA_PASS_THROUGH = [
    "open fridge 1",
    "close fridge 1",
    "clean mug 1 with sinkbasin 1",
    "heat mug 1 with microwave 1",
    "cool mug 1 with fridge 1",
    "toggle desklamp 1",
    "use desklamp 1",
    "examine mug 1",
    "look at mug 1",
    "look",
    "inventory",
    # malformed / incomplete put utterances must NOT be force-converted
    "put",
    "put mug 1",
    "Nothing can be done. The task cannot be completed as no soapbar was found.",
    # ── 7. contains the characters "put" but is not a put action ─────────
    "pickup mug 1",
    "putty 1 in drawer 1",
    "computer 1 on desk 1",
    "disputed item in drawer 1",
]

# ── 8. incomplete/malformed put commands must not be broadly rewritten ───
MALFORMED_NOT_CONVERTED = [
    "put mug 1",
    "put in fridge 1",
    "put",
    "put mug 1 fridge 1",  # missing preposition entirely
]

# ── normal-language/Thought/Observation text with "put" inside must not be
#    touched by this function -- callers only ever pass it the single
#    already-parsed action string, never free text, but the function itself
#    must still be safe if misused this way.
THOUGHT_LIKE_TEXT = [
    "Thought: I need to put the mug in the fridge, then take it out.",
    "Observation: You put the mug 1 in/on the fridge 1.",
]


def test_should_convert():
    for raw, expected in CASES_SHOULD_CONVERT + EXTRA_CONVERT:
        executed, norm = normalize_alfworld_action(raw)
        check(f"convert: {raw!r} -> {expected!r}",
              executed == expected and norm == "put_to_move",
              (executed, norm))


def test_should_pass_through():
    for raw in CASES_SHOULD_PASS_THROUGH + EXTRA_PASS_THROUGH:
        executed, norm = normalize_alfworld_action(raw)
        check(f"passthrough: {raw!r}",
              executed == raw.strip() and norm is None,
              (executed, norm))


def test_malformed_put_not_converted():
    for raw in MALFORMED_NOT_CONVERTED:
        executed, norm = normalize_alfworld_action(raw)
        check(f"malformed put not force-converted: {raw!r}",
              executed == raw.strip() and norm is None,
              (executed, norm))


def test_thought_and_observation_text_untouched():
    for raw in THOUGHT_LIKE_TEXT:
        executed, norm = normalize_alfworld_action(raw)
        check(f"Thought/Observation-like text unchanged: {raw!r}",
              executed == raw.strip() and norm is None,
              (executed, norm))


def test_object_and_receptacle_numbering_preserved():
    executed, norm = normalize_alfworld_action("put soapbar 2 in/on garbagecan 10")
    check("numbering preserved through conversion",
          executed == "move soapbar 2 to garbagecan 10", executed)

    executed2, _ = normalize_alfworld_action("put mug 12 in fridge 3")
    check("multi-digit object and receptacle indices both preserved",
          executed2 == "move mug 12 to fridge 3", executed2)


def test_different_object_or_receptacle_index_not_collapsed():
    a, _ = normalize_alfworld_action("put mug 1 in fridge 1")
    b, _ = normalize_alfworld_action("put mug 2 in fridge 1")
    c, _ = normalize_alfworld_action("put mug 1 in fridge 2")
    check("different object index yields a different executed_action", a != b, (a, b))
    check("different receptacle index yields a different executed_action", a != c, (a, c))


def test_none_and_empty_input_safe():
    executed, norm = normalize_alfworld_action(None)
    check("None input returns (None, None), no crash", (executed, norm) == (None, None))

    executed, norm = normalize_alfworld_action("")
    check("empty string passes through unchanged", (executed, norm) == ("", None))


if __name__ == "__main__":
    test_should_convert()
    test_should_pass_through()
    test_malformed_put_not_converted()
    test_thought_and_observation_text_untouched()
    test_object_and_receptacle_numbering_preserved()
    test_different_object_or_receptacle_index_not_collapsed()
    test_none_and_empty_input_safe()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
