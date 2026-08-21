"""Tests for canonical_action.py and its wiring into signals.py's
repeated_action signal. Plain assert-based, same convention as
test_ae_controller.py. Run directly:
    python3 ae/controllers/tests/test_canonical_action.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root

from ae.controllers.canonical_action import canonicalize_action  # noqa: E402
from ae.controllers.signals import SignalExtractor, action_similarity  # noqa: E402

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


# ── the exact put-preposition-cycle from the 20-task pilot report ────────
def test_put_preposition_cycle_all_equal():
    """pick_cool_then_place_in_recep-Lettuce-CounterTop-10, steps 15-50:
    the model cycled through on/in/at/into and an incrementing leading
    number for 35 steps straight ("put lettuce 1 on countertop 1", "1. put
    lettuce 1 on countertop 1", "2. put lettuce 1 in countertop 1", ...),
    none of which were exact-string matches of each other, so the old
    plain equality check never re-fired after the first hit. All of these
    must canonicalize to the same signature."""
    variants = [
        "put lettuce 1 on countertop 1",
        "put lettuce 1 in countertop 1",
        "put lettuce 1 at countertop 1",
        "put lettuce 1 into countertop 1",
        "1. put lettuce 1 on countertop 1",
        "2. put lettuce 1 in countertop 1",
        "17. put lettuce 1 at countertop 1",
        "33. put lettuce 1 on countertop 1",
    ]
    sigs = {canonicalize_action(v) for v in variants}
    check("1. all put-preposition/leading-numbering variants canonicalize identically",
          len(sigs) == 1, sigs)
    check("1b. the shared signature is exactly ('put', 'lettuce 1', 'countertop 1')",
          sigs == {("put", "lettuce 1", "countertop 1")}, sigs)


def test_put_different_target_not_equal():
    a = canonicalize_action("put lettuce 1 on countertop 1")
    b = canonicalize_action("put lettuce 1 on countertop 2")
    check("2. different receptacle index is NOT collapsed away "
          "(object indices are a real distinction, not formatting noise)",
          a != b, (a, b))


def test_take_and_go_families():
    check("3. take X from Y", canonicalize_action("take mug 1 from fridge 1")
          == ("take", "mug 1", "fridge 1"))
    check("3b. go to X", canonicalize_action("go to shelf 2") == ("go", "shelf 2"))
    check("3c. open X", canonicalize_action("open cabinet 3") == ("open", "cabinet 3"))
    check("3d. heat X with Y", canonicalize_action("heat egg 1 with microwave 1")
          == ("heat", "egg 1", "microwave 1"))
    check("3e. cool X with Y", canonicalize_action("cool potato 1 with fridge 1")
          == ("cool", "potato 1", "fridge 1"))
    check("3f. clean X with Y", canonicalize_action("clean soapbar 1 with sinkbasin 1")
          == ("clean", "soapbar 1", "sinkbasin 1"))
    check("3g. toggle/use synonym", canonicalize_action("use desklamp 1")
          == canonicalize_action("toggle desklamp 1"))
    check("3h. look", canonicalize_action("look") == ("look",))
    check("3i. inventory", canonicalize_action("inventory") == ("inventory",))
    check("3j. think pseudo-action", canonicalize_action("think: I should check the fridge")
          == ("think", "i should check the fridge"))


def test_unparseable_returns_none():
    check("4. free-form garbage does not falsely match a family",
          canonicalize_action("1. Check cabinets. Open cabinet 1, 2, 3.") is None)
    check("4b. empty string returns None", canonicalize_action("") is None)


def test_action_similarity_uses_canonical_when_parseable():
    a = "put lettuce 1 on countertop 1"
    b = "3. put lettuce 1 at countertop 1"
    check("5. canonically-equal paraphrases score a full 1.0 "
          "(plain Jaccard on these two strings would score well under 1.0)",
          action_similarity(a, b) == 1.0)
    c = "put lettuce 1 on countertop 2"
    check("5b. canonically-different targets score 0.0, not a partial Jaccard credit",
          action_similarity(a, c) == 0.0)


def test_action_similarity_falls_back_to_jaccard_when_unparseable():
    a = "1. Check cabinets. Open cabinet 1, 2, 3."
    b = "1. Check shelves. Open shelf 1, 2, 3."
    sim = action_similarity(a, b)
    check("6. unparseable actions fall back to graded Jaccard similarity "
          "(neither 0 nor 1, since these overlap partially in tokens)",
          0.0 < sim < 1.0, sim)


# ── put/move canonical equivalence (added alongside the put->move
#    execution fix in alfworld_runs_ae/alfworld_action_normalize.py) ──────
def test_put_and_move_canonicalize_identically():
    """The model's prompt teaches 'put X in/on Y'; the real ALFWorld data
    only executes 'move X to Y'. Regardless of which verb the model says
    (or which one actually reached env.step()), repetition/stuck detection
    must see them as the same action -- this is what canonical_action.py
    now guarantees, independent of the separate execution-time rewrite."""
    a = canonicalize_action("put mug 1 in fridge 1")
    b = canonicalize_action("move mug 1 to fridge 1")
    check("9. 'put mug 1 in fridge 1' and 'move mug 1 to fridge 1' canonicalize identically",
          a == b, (a, b))
    check("9b. the shared signature is exactly ('put', 'mug 1', 'fridge 1')",
          a == ("put", "mug 1", "fridge 1"), a)

    c = canonicalize_action("put bread 1 on countertop 2")
    d = canonicalize_action("move bread 1 to countertop 2")
    check("9c. 'put bread 1 on countertop 2' and 'move bread 1 to countertop 2' canonicalize identically",
          c == d, (c, d))


def test_put_and_move_still_distinguish_object_and_receptacle():
    """Unifying the put/move family label must not collapse away real
    object/receptacle distinctions -- a different object or receptacle
    index between a 'put' and a 'move' phrasing must still be a different
    canonical signature."""
    a = canonicalize_action("put mug 1 in fridge 1")
    b = canonicalize_action("move mug 2 to fridge 1")
    check("10. different object index (put vs move) is NOT collapsed", a != b, (a, b))

    c = canonicalize_action("put mug 1 in fridge 1")
    d = canonicalize_action("move mug 1 to fridge 2")
    check("10b. different receptacle index (put vs move) is NOT collapsed", c != d, (c, d))


def test_action_similarity_treats_put_and_move_as_equal():
    """End-to-end through action_similarity (what signals.py's
    repeated_action actually calls): a 'move'-phrased action must score a
    full 1.0 match against an equivalent 'put'-phrased recent action, the
    same way two 'put' paraphrases already do."""
    a = "put mug 1 in fridge 1"
    b = "move mug 1 to fridge 1"
    check("11. action_similarity('put ...', 'move ...') for the same object/receptacle is 1.0",
          action_similarity(a, b) == 1.0)

    c = "move mug 1 to countertop 1"
    check("11b. action_similarity('put ...', 'move ...') for a DIFFERENT receptacle is 0.0",
          action_similarity(a, c) == 0.0)


def test_signal_extractor_repeated_action_uses_canonical():
    """End-to-end: SignalExtractor.extract()'s repeated_action must reflect
    the canonical match, not just exact-string comparison, given the exact
    sequence from the pilot report."""
    extractor = SignalExtractor()
    recent_actions = [
        "put lettuce 1 on countertop 1",
        "1. put lettuce 1 in countertop 1",
        "2. put lettuce 1 at countertop 1",
    ]
    signals = extractor.extract(
        action="3. put lettuce 1 into countertop 1",
        observation="Nothing happens.",
        admissible_before=["put lettuce 1 in countertop 1"],
        recent_actions=recent_actions,
        recent_observations=["Nothing happens."] * 3,
        step_index=18,
        max_steps=50,
        is_think_action=False,
    )
    check("7. repeated_action signal is 1.0 for the preposition/numbering "
          "cycle (previously invisible to exact-match repetition checks)",
          signals.repeated_action == 1.0, signals.repeated_action)


if __name__ == "__main__":
    test_put_preposition_cycle_all_equal()
    test_put_different_target_not_equal()
    test_take_and_go_families()
    test_unparseable_returns_none()
    test_action_similarity_uses_canonical_when_parseable()
    test_action_similarity_falls_back_to_jaccard_when_unparseable()
    test_put_and_move_canonicalize_identically()
    test_put_and_move_still_distinguish_object_and_receptacle()
    test_action_similarity_treats_put_and_move_as_equal()
    test_signal_extractor_repeated_action_uses_canonical()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
