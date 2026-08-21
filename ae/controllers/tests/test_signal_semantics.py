"""Tests for the action-family-aware unexpected_outcome/surprise fix (Part
E of the revision spec) and the new information_gain_proxy /
local_state_change_proxy signals. Plain assert-based, same convention as
the other controller test files. Run directly:
    python3 ae/controllers/tests/test_signal_semantics.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root

from ae.controllers.signals import SignalExtractor  # noqa: E402

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def _extract(**kwargs):
    base = dict(
        admissible_before=["go to shelf 1", "go to shelf 2", "go to shelf 3"],
        admissible_after=["go to shelf 1", "go to shelf 2", "go to shelf 3"],
        recent_actions=[], recent_observations=[],
        step_index=4, max_steps=50, is_think_action=False,
    )
    base.update(kwargs)
    return SignalExtractor().extract(**base)


# ── the exact previous-round over-trigger case ───────────────────────────
def test_routine_empty_receptacle_exploration_not_unexpected():
    """pick_and_place_simple-Vase-None-Safe-219 step 4 / pick_clean_then_
    place_in_recep-SoapBar-None-Cabinet-424 step 4: a plain "go to X"
    landing on an empty receptacle previously scored unexpected_outcome=1.0
    purely because the observation resembled a recent one. This is
    completely routine ALFWorld search behavior, not a surprising outcome,
    and must now score 0."""
    s = _extract(
        action="go to shelf 2",
        observation="You arrive at shelf 2. On the shelf 2, you see nothing.",
        recent_observations=["You arrive at shelf 1. On the shelf 1, you see nothing."],
    )
    check("1. routine 'go' to an empty receptacle scores unexpected_outcome=0",
          s.unexpected_outcome == 0.0, s.unexpected_outcome)


def test_look_and_examine_also_exempt():
    s_look = _extract(action="look", observation="You see nothing special.",
                       recent_observations=["You see nothing special."])
    s_examine = _extract(action="examine shelf 2", observation="On the shelf 2, you see nothing.",
                          recent_observations=["On the shelf 1, you see nothing."])
    check("2. 'look' is exempt from unexpected_outcome", s_look.unexpected_outcome == 0.0)
    check("2b. 'examine X' is exempt from unexpected_outcome", s_examine.unexpected_outcome == 0.0)


# ── state-changing families still get a real unexpected_outcome check ────
def test_state_changing_action_with_no_effect_is_unexpected():
    s = _extract(
        action="open cabinet 1",
        observation="Nothing happens.",
        admissible_before=["open cabinet 1", "go to cabinet 2"],
        admissible_after=["open cabinet 1", "go to cabinet 2"],  # unchanged -> no effect
    )
    check("3. a legal state-changing action producing literally no effect "
          "still scores unexpected_outcome=1.0", s.unexpected_outcome == 1.0)


def test_state_changing_action_with_real_effect_is_not_unexpected():
    s = _extract(
        action="open cabinet 1",
        observation="You open the cabinet 1. The cabinet 1 is open. In it, you see a mug 1.",
        admissible_before=["open cabinet 1", "go to cabinet 2"],
        admissible_after=["close cabinet 1", "take mug 1 from cabinet 1", "go to cabinet 2"],
    )
    check("4. a state-changing action whose admissible set actually changed "
          "is not flagged unexpected, even though open cabinets often read "
          "as textually similar to each other", s.unexpected_outcome == 0.0)


def test_state_changing_action_no_admissible_change_but_real_text_effect():
    """clean/heat/cool often don't change admissible_commands at all (no
    new interactable objects appear) even though the action genuinely
    succeeded -- local_state_change_proxy must not rely on admissible-diff
    alone, since the "Nothing happens." text check independently confirms
    a real effect happened here."""
    s = _extract(
        action="clean soapbar 1 with sinkbasin 1",
        observation="You clean the soapbar 1 using the sinkbasin 1.",
        admissible_before=["clean soapbar 1 with sinkbasin 1"],
        admissible_after=["clean soapbar 1 with sinkbasin 1"],  # unchanged, but text confirms success
    )
    check("5. a successful clean/heat/cool with an unchanged admissible "
          "set but non-'Nothing happens.' text is NOT flagged unexpected",
          s.unexpected_outcome == 0.0)
    check("5b. local_state_change_proxy reflects the real (textual) change",
          s.local_state_change_proxy == 1.0)


def test_invalid_action_never_unexpected():
    s = _extract(action="go to nowhere", observation="Nothing happens.",
                  admissible_before=["go to shelf 1"], admissible_after=["go to shelf 1"])
    check("6. an invalid action is never itself flagged unexpected_outcome "
          "(it's wrong, not surprising)", s.unexpected_outcome == 0.0)


def test_unknown_family_falls_back_to_old_heuristic():
    # In real ALFWorld play, an action that doesn't parse into a known
    # family is essentially always also invalid (not in admissible_before,
    # since every real admissible command IS covered by the known
    # families) -- and invalid actions never get unexpected_outcome=1.0
    # regardless of family. To exercise the "legal but unparseable" branch
    # in isolation, this test synthetically puts the garbage string itself
    # into admissible_before/after.
    garbage = "1. Check cabinets. Open cabinet 1, 2, 3."
    s = _extract(
        action=garbage,
        observation="Nothing happens.",
        admissible_before=["go to shelf 1", garbage], admissible_after=["go to shelf 1", garbage],
    )
    check("7. a legal-but-unparseable action still uses the old "
          "family-agnostic no-effect heuristic (not silently exempted)",
          s.unexpected_outcome == 1.0)


# ── information_gain_proxy / local_state_change_proxy exist and behave ──
def test_information_gain_proxy_on_novel_admissible_set():
    s = _extract(
        action="go to shelf 4",
        observation="You arrive at shelf 4. On the shelf 4, you see a bowl 1.",
        admissible_before=["go to shelf 1", "go to shelf 2"],
        admissible_after=["go to shelf 1", "go to shelf 2", "take bowl 1 from shelf 4"],
        recent_observations=["You arrive at shelf 3. On the shelf 3, you see nothing."],
    )
    check("8. a newly-changed admissible set counts as information gain",
          s.information_gain_proxy == 1.0)


def test_local_state_change_proxy_zero_when_truly_nothing_changed():
    s = _extract(
        action="open cabinet 1", observation="Nothing happens.",
        admissible_before=["open cabinet 1"], admissible_after=["open cabinet 1"],
    )
    check("9. local_state_change_proxy is 0 when neither text nor "
          "admissible set changed", s.local_state_change_proxy == 0.0)


# ── consecutive_exact_action_repeat (frustration_medium repeat-gate fix) ──
# Real A40 audit case: "go to shelf 5" -> "go to shelf 6" scored
# repeated_observation=0.8 (Jaccard over near-identical "You arrive at
# shelf N. On the shelf N, you see nothing." template text) even though
# these are two DIFFERENT, never-before-visited locations -- this new
# signal must score 0 for that case, using ONLY the immediately-preceding
# action (not repeated_action's repetition_window-max) and a strict
# canonical-tuple-or-cleaned-string exact match (not similarity).
def test_exact_repeat_different_shelf_numbers_not_repeated():
    s = _extract(action="go to shelf 6", recent_actions=["go to shelf 5"],
                 observation="You arrive at shelf 6. On the shelf 6, you see nothing.",
                 recent_observations=["You arrive at shelf 5. On the shelf 5, you see nothing."])
    check("10. 'go to shelf 5' -> 'go to shelf 6' is NOT an exact repeat "
          "(different object number, real semantic distinction)",
          s.consecutive_exact_action_repeat == 0.0, s.consecutive_exact_action_repeat)


def test_exact_repeat_go_then_open_different_family_not_repeated():
    s = _extract(action="open drawer 3", recent_actions=["go to drawer 3"],
                 observation="You open the drawer 3. The drawer 3 is open. In it, you see a saltshaker 1.",
                 recent_observations=["You arrive at drawer 3. The drawer 3 is closed."])
    check("11. 'go to drawer 3' -> 'open drawer 3' is NOT an exact repeat "
          "(different action family -- a real state transition, not stuck)",
          s.consecutive_exact_action_repeat == 0.0, s.consecutive_exact_action_repeat)


def test_exact_repeat_same_action_twice_is_repeated():
    s = _extract(action="use desklamp 1", recent_actions=["use desklamp 1"],
                 observation="Nothing happens.", recent_observations=["Nothing happens."])
    check("12. 'use desklamp 1' -> 'use desklamp 1' (identical) IS an exact repeat",
          s.consecutive_exact_action_repeat == 1.0, s.consecutive_exact_action_repeat)


def test_exact_repeat_with_leading_enumeration_prefix_still_detected():
    s = _extract(action="use desklamp 1", recent_actions=["1. use desklamp 1"],
                 observation="Nothing happens.", recent_observations=["Nothing happens."])
    check("13. leading enumeration prefix ('1. use desklamp 1' vs 'use desklamp "
          "1') is format noise, not a real difference -- still detected as repeat",
          s.consecutive_exact_action_repeat == 1.0, s.consecutive_exact_action_repeat)


def test_exact_repeat_ignores_window_only_checks_immediately_preceding():
    # The action 3 steps back matches, but the IMMEDIATELY preceding one
    # (last in recent_actions) does not -- must NOT count as a repeat,
    # unlike repeated_action which would take the window max.
    s = _extract(action="use desklamp 1",
                 recent_actions=["use desklamp 1", "go to shelf 2", "go to shelf 3"],
                 observation="You arrive at shelf 4.", recent_observations=[])
    check("14. only the SINGLE immediately-preceding action is compared, "
          "not a repetition_window max (unlike repeated_action)",
          s.consecutive_exact_action_repeat == 0.0, s.consecutive_exact_action_repeat)


def test_exact_repeat_no_prior_action_is_zero():
    s = _extract(action="go to shelf 1", recent_actions=[], observation="You arrive at shelf 1.")
    check("15. no prior action (first step of episode) scores 0, not an error",
          s.consecutive_exact_action_repeat == 0.0, s.consecutive_exact_action_repeat)


def test_exact_repeat_unparseable_action_falls_back_to_cleaned_string_match():
    garbage = "1. Check cabinets. Open cabinet 1, 2, 3."
    s = _extract(action=garbage, recent_actions=["check cabinets. open cabinet 1, 2, 3."],
                 observation="Nothing happens.")
    check("16. an unparseable action falls back to format-cleaned (not "
          "object-stripped) exact string match", s.consecutive_exact_action_repeat == 1.0)


if __name__ == "__main__":
    test_routine_empty_receptacle_exploration_not_unexpected()
    test_look_and_examine_also_exempt()
    test_state_changing_action_with_no_effect_is_unexpected()
    test_state_changing_action_with_real_effect_is_not_unexpected()
    test_state_changing_action_no_admissible_change_but_real_text_effect()
    test_invalid_action_never_unexpected()
    test_unknown_family_falls_back_to_old_heuristic()
    test_information_gain_proxy_on_novel_admissible_set()
    test_local_state_change_proxy_zero_when_truly_nothing_changed()
    test_exact_repeat_different_shelf_numbers_not_repeated()
    test_exact_repeat_go_then_open_different_family_not_repeated()
    test_exact_repeat_same_action_twice_is_repeated()
    test_exact_repeat_with_leading_enumeration_prefix_still_detected()
    test_exact_repeat_ignores_window_only_checks_immediately_preceding()
    test_exact_repeat_no_prior_action_is_zero()
    test_exact_repeat_unparseable_action_falls_back_to_cleaned_string_match()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
