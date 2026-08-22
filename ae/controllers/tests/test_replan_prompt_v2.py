"""Tests for REPLAN prompt-v2 "B+" (see
audit_reports/ae_replan_prompt_v2_last_action_semantics_audit.md and
ae_replan_prompt_v2_b_plus_report.md). Drives a real StatefulController
through real step() calls to exercise the real signal computation ->
progress-gated bookkeeping -> active_directive()/render_directive() ->
prompt-splice path -- no hand-typed directive strings, no mocking of the
controller/renderer/signal extractor.

Run directly:
    python3 ae/controllers/tests/test_replan_prompt_v2.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "alfworld_runs_ae"))

from ae.core import InterventionType  # noqa: E402
from ae.controllers.config import AEConfig  # noqa: E402
from ae.controllers.stateful_controller import StatefulController  # noqa: E402
from ae.controllers.intervention_renderer import render_directive  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def _step_kwargs(action, observation, admissible=("x",), display_action_text=None,
                  display_observation_text=None, recent_actions=(), recent_observations=()):
    return dict(
        action=action, observation=observation,
        admissible_before=list(admissible), admissible_after=list(admissible),
        recent_actions=list(recent_actions), recent_observations=list(recent_observations),
        max_steps=50, is_think_action=False,
        display_action_text=display_action_text, display_observation_text=display_observation_text,
    )


def _base_cfg(**overrides):
    kwargs = dict(warmup_steps=3, cooldown_steps=0, max_interventions=20, patch_duration_steps=3,
                   post_intervention_repeat_enabled=False)
    kwargs.update(overrides)
    return AEConfig(**kwargs)


GOAL = "Your task is to: heat some egg and put it in garbagecan.___14"


def _drive(controller, steps, admissible=("go to shelf 1", "go to shelf 2", "take mug 1 from shelf 1")):
    """steps: list of (action, observation) with local_state_change_proxy
    controlled purely by whether observation normalizes to "nothing
    happens." (proxy=0, non-progress) or not (proxy=1, progress) -- the
    same established mechanic used throughout this session's tests."""
    recent_a, recent_o = [], []
    r = None
    for act, obs in steps:
        r = controller.step(**_step_kwargs(act, obs, admissible=admissible,
                                            recent_actions=recent_a, recent_observations=recent_o))
        recent_a.append(act)
        recent_o.append(obs)
    return r


# ---- 1/5: non-progress action+observation recorded together, atomically ----
def test_nonprogress_pair_recorded_together():
    cfg = _base_cfg(warmup_steps=0)
    controller = StatefulController(cfg)
    _drive(controller, [("go to shelf 1", "Nothing happens.")])
    check("1a. last_nonprogress_action_text == this step's action",
          controller.last_nonprogress_action_text == "go to shelf 1")
    check("1b. last_nonprogress_observation_text == this step's observation "
          "(same step as the action -- atomic pairing, not a mismatched pair)",
          controller.last_nonprogress_observation_text == "Nothing happens.")


# ---- 2/3/4: progress does not overwrite; a LATER non-progress step does ----
def test_progress_does_not_overwrite_then_later_nonprogress_updates():
    cfg = _base_cfg(warmup_steps=0)
    controller = StatefulController(cfg)
    # step 1: non-progress -> records X
    _drive(controller, [("go to shelf 1", "Nothing happens.")])
    check("2a. baseline recorded before any progress step",
          controller.last_nonprogress_action_text == "go to shelf 1")

    # step 2: genuine progress (obs != "nothing happens.") -> must NOT overwrite
    r = _drive(controller, [("take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1.")])
    check("2b. progress step's own signal really did register as progress "
          "(local_state_change_proxy>=0.5) -- otherwise this test proves nothing",
          r["signals"]["local_state_change_proxy"] >= 0.5, r["signals"])
    check("2c. after a progress step, last_nonprogress_action_text is UNCHANGED "
          "(still the earlier non-progress action, not the progressing one)",
          controller.last_nonprogress_action_text == "go to shelf 1",
          controller.last_nonprogress_action_text)
    check("2d. after a progress step, last_nonprogress_observation_text is UNCHANGED",
          controller.last_nonprogress_observation_text == "Nothing happens.",
          controller.last_nonprogress_observation_text)

    # test 3: render the directive right now and confirm the progressing
    # action never appears labeled as "no observed progress"
    directive = render_directive(
        InterventionType.REPLAN, goal=GOAL,
        steps_since_meaningful_change=controller.steps_since_meaningful_change,
        last_nonprogress_action=controller.last_nonprogress_action_text,
        last_nonprogress_observation=controller.last_nonprogress_observation_text,
        current_observation=controller.current_observation_text,
    )
    check("3. the progressing action ('take mug 1 from shelf 1') never appears as "
          "'Most recent action with no observed progress' in the rendered directive",
          "Most recent action with no observed progress: take mug 1 from shelf 1" not in directive,
          directive)

    # step 3: a NEW non-progress action -> record must update to it (test 4)
    _drive(controller, [("go to shelf 2", "Nothing happens.")])
    check("4a. a later non-progress step DOES update the record to the new action",
          controller.last_nonprogress_action_text == "go to shelf 2",
          controller.last_nonprogress_action_text)
    check("4b. ...and its paired observation updates together with it",
          controller.last_nonprogress_observation_text == "Nothing happens.")


# ---- 6: current_observation_text always reflects the latest step ----
def test_current_observation_always_latest_regardless_of_progress():
    cfg = _base_cfg(warmup_steps=0)
    controller = StatefulController(cfg)
    _drive(controller, [("go to shelf 1", "Nothing happens.")])
    check("6a. current_observation_text reflects a non-progress step's observation",
          controller.current_observation_text == "Nothing happens.")
    _drive(controller, [("take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1.")])
    check("6b. current_observation_text updates to a PROGRESS step's observation too "
          "(unconditional, unlike last_nonprogress_*)",
          controller.current_observation_text == "You pick up the mug 1 from the shelf 1.",
          controller.current_observation_text)
    _drive(controller, [("go to shelf 2", "Nothing happens.")])
    check("6c. current_observation_text updates again on a subsequent non-progress step",
          controller.current_observation_text == "Nothing happens.")


def _arm_replan(goal=GOAL):
    """Controller-agnostic REPLAN trigger: repeats the exact same admissible
    action ("go to shelf 1") past warmup until REPLAN actually arms,
    whichever step count that takes -- old-controller (REPLAN-unconditional
    on frustration_high+confidence_low) arms as soon as that pair of bands
    crosses; new-controller (sustained-only) additionally needs
    steps_since_meaningful_change>=patch_duration_steps, which the same
    repeated "Nothing happens." action also satisfies, just a few steps
    later. This intentionally does NOT hardcode a fixed step count (an
    earlier version did, tuned only for the sustained controller, and
    crashed when reused against the old controller -- see
    audit_reports/ae_replan_prompt_v2_dual_controller_experiment_report.md).
    display_action_text/display_observation_text differ from the internal
    action from the first non-warmup repeat onward (a synthetic
    "pre-parse" wording, not a real ALFWorld normalization pair) purely to
    prove the directive renders the display text, not the internal one,
    on whichever step actually arms REPLAN."""
    cfg = _base_cfg()
    controller = StatefulController(cfg)
    recent_a, recent_o = [], []
    r = None
    # warmup_steps=3: two invalid actions + one first occurrence, all
    # suppressed regardless of candidate severity.
    for act, obs in [("do invalid thing", "You can't do that."),
                      ("do another invalid", "That doesn't work here."),
                      ("go to shelf 1", "Nothing happens.")]:
        r = controller.step(**_step_kwargs(act, obs, admissible=("go to shelf 1",),
                                            recent_actions=recent_a, recent_observations=recent_o))
        recent_a.append(act)
        recent_o.append(obs)
    max_iters = 15
    for _ in range(max_iters):
        r = controller.step(**_step_kwargs(
            "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
            display_action_text="walk over to shelf 1", display_observation_text="Nothing happens.",
            recent_actions=recent_a, recent_observations=recent_o,
        ))
        recent_a.append("go to shelf 1")
        recent_o.append("Nothing happens.")
        if r["intervention"] == InterventionType.REPLAN.value:
            break
    assert r["intervention"] == InterventionType.REPLAN.value, r
    directive = controller.active_directive(goal=goal)
    return controller, directive


def test_replan_directive_contains_full_original_task():
    _, directive = _arm_replan()
    check("full-task-1. directive contains the FULL original task text verbatim",
          GOAL in directive, directive)
    check("full-task-1b. does not just say a vague 'reread the task' placeholder",
          "reread" not in directive.lower() and "think again" not in directive.lower())


def test_replan_directive_contains_nonprogress_action_and_env_response():
    _, directive = _arm_replan()
    check("pair-1. contains the model-visible non-progress action text under the new label",
          "Most recent action with no observed progress: walk over to shelf 1" in directive, directive)
    check("pair-2. contains the environment response to that action under the new label",
          "Environment response to that action: Nothing happens." in directive, directive)


def test_replan_directive_contains_current_observation():
    _, directive = _arm_replan()
    check("current-obs-1. contains a 'Current observation:' line with real (non-placeholder) content",
          "Current observation: Nothing happens." in directive, directive)


def test_replan_directive_explicitly_forbids_repeating_the_nonprogress_action():
    _, directive = _arm_replan()
    check("forbid-1. explicit prohibition on repeating the most recent no-observed-progress "
          "action is present (worded without 'failed', which local_state_change_proxy<0.5 "
          "alone cannot establish -- see the progress-wording precision fix)",
          "do not repeat the most recent action with no observed progress" in directive.lower(), directive)


def test_replan_directive_requires_one_line_plan_and_one_action():
    _, directive = _arm_replan()
    check("format-1. asks for a single concise Thought line (not a multi-line plan)",
          "one concise thought line" in directive.lower(), directive)
    check("format-2. asks for the plan to be expressed as a compact A -> B -> C route",
          "->" in directive, directive)
    check("format-3. asks for exactly one executable Action line",
          "exactly one executable action line" in directive.lower(), directive)
    check("format-4. still forbids assuming unobserved state",
          "do not assume any state that has not been observed" in directive.lower(), directive)


# ---- 7: within a REPLAN window, a progressing action is not mislabeled on the NEXT directive ----
def test_replan_window_progress_not_mislabeled_on_next_directive():
    controller, directive_t1 = _arm_replan()
    pre_progress_nonprogress_action = controller.last_nonprogress_action_text
    check("window-7a. sanity: directive shown at t+1 correctly names the pre-arm non-progress action",
          "Most recent action with no observed progress: walk over to shelf 1" in directive_t1, directive_t1)

    # model's t+1 response: this time it makes REAL progress
    r = controller.step(**_step_kwargs(
        "take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1.",
        admissible=("go to shelf 1", "go to shelf 2", "take mug 1 from shelf 1"),
        display_action_text="take mug 1 from shelf 1",
        display_observation_text="You pick up the mug 1 from the shelf 1.",
        recent_actions=["do invalid thing", "do another invalid", "go to shelf 1", "go to shelf 2",
                         "go to shelf 1", "go to shelf 1"],
        recent_observations=["You can't do that.", "That doesn't work here.", "Nothing happens.",
                              "Nothing happens.", "Nothing happens.", "Nothing happens."],
    ))
    check("window-7b. t+1's own action really did register progress",
          r["signals"]["local_state_change_proxy"] >= 0.5, r["signals"])

    directive_t2 = controller.active_directive(goal=GOAL)
    check("window-7c. t+2's directive does NOT relabel the just-successful t+1 action "
          "as 'no observed progress'",
          "no observed progress: take mug 1 from shelf 1" not in directive_t2.lower(), directive_t2)
    check("window-7d. t+2's directive still names the ORIGINAL pre-arm non-progress action "
          "(unchanged from t+1, since nothing non-progressing happened since)",
          f"Most recent action with no observed progress: {pre_progress_nonprogress_action}" in directive_t2,
          directive_t2)
    check("window-7e. t+2's Current observation DOES update to the real, latest observation "
          "(unlike the frozen non-progress pair)",
          "Current observation: You pick up the mug 1 from the shelf 1." in directive_t2, directive_t2)


def test_missing_optional_state_degrades_safely_no_none_no_blank_no_fabrication():
    directive = render_directive(InterventionType.REPLAN, goal=None,
                                  steps_since_meaningful_change=None,
                                  last_nonprogress_action=None, last_nonprogress_observation=None,
                                  current_observation=None)
    check("safe-1. no literal 'None' anywhere in the rendered directive",
          "None" not in directive, directive)
    check("safe-2. no bare empty placeholder for the task line",
          "Original task: \n" not in directive, directive)
    check("safe-3. explicit, honest placeholder for the missing non-progress action, not a blank",
          "(no non-progress action recorded this episode)" in directive, directive)
    check("safe-4. explicit, honest placeholder for the missing non-progress observation, not a blank",
          "(no corresponding observation recorded this episode)" in directive, directive)
    check("safe-5. explicit, honest placeholder for the missing current observation, not a blank",
          "(no observation recorded this episode)" in directive, directive)
    check("safe-6. explicit, honest placeholder for the missing task text, not a blank",
          "(task text unavailable)" in directive, directive)
    check("safe-7. missing steps_since_meaningful_change safely defaults to 0 -- no crash/None, "
          "and (per the progress-aware wording fix) 0 correctly selects the progress-observed "
          "branch rather than fabricating a literal 'for 0 steps' non-progress claim",
          "The latest step produced an observed state change." in directive
          and "No meaningful progress" not in directive, directive)


# ---- 9: reset() clears all three fields ----
def test_reset_clears_all_three_fields():
    cfg = _base_cfg(warmup_steps=0)
    controller = StatefulController(cfg)
    _drive(controller, [("go to shelf 1", "Nothing happens.")])
    check("reset-pre. fields are populated before reset (sanity)",
          controller.last_nonprogress_action_text is not None
          and controller.last_nonprogress_observation_text is not None
          and controller.current_observation_text is not None)
    controller.reset()
    check("reset-9a. last_nonprogress_action_text cleared to None after reset()",
          controller.last_nonprogress_action_text is None)
    check("reset-9b. last_nonprogress_observation_text cleared to None after reset()",
          controller.last_nonprogress_observation_text is None)
    check("reset-9c. current_observation_text cleared to None after reset()",
          controller.current_observation_text is None)


def test_reflect_and_verify_directives_byte_identical_to_v1():
    reflect = render_directive(InterventionType.REFLECT)
    verify = render_directive(InterventionType.VERIFY)
    expected_reflect = (
        "[ACTIVE CONTROL DIRECTIVE]\n"
        "Diagnose why the recent actions did not change the environment.\n"
        "Do not repeat the same failed action.\n"
        "Choose a corrected action from the current environment state."
    )
    expected_verify = (
        "[ACTIVE CONTROL DIRECTIVE]\n"
        "Re-check the current goal and latest observation.\n"
        "Identify what information is still missing.\n"
        "Prefer one information-gathering action before changing the whole strategy."
    )
    check("v1-10a. REFLECT directive byte-identical to v1", reflect == expected_reflect, reflect)
    check("v1-10b. VERIFY directive byte-identical to v1", verify == expected_verify, verify)
    check("v1-10c. REFLECT/VERIFY directives ignore all REPLAN-only kwargs entirely",
          render_directive(InterventionType.REFLECT, goal="anything", last_nonprogress_action="x",
                            last_nonprogress_observation="y", current_observation="z",
                            steps_since_meaningful_change=99) == expected_reflect)


def test_real_prompt_construction_path_end_to_end():
    """Exercises the actual agents.py splice logic (prompt_for_llm +=
    f"\\n\\n{directive}\\n") against a real controller/directive, not a
    reimplementation."""
    controller, _ = _arm_replan()
    base_prompt = "Interact with a household to solve a task...\nHere is the task:\n" + GOAL + "\n"
    prompt_for_llm = base_prompt
    directive = controller.active_directive(goal=GOAL)
    if directive:
        prompt_for_llm += f"\n\n{directive}\n"
    check("splice-1. assembled prompt contains the task text once from base_prompt "
          "and the directive's own restated copy (both present)",
          prompt_for_llm.count(GOAL) == 2, prompt_for_llm.count(GOAL))
    check("splice-2. directive appears strictly after the base prompt",
          prompt_for_llm.index("[ACTIVE CONTROL DIRECTIVE: REPLAN]") > prompt_for_llm.index("Here is the task:"))


if __name__ == "__main__":
    test_nonprogress_pair_recorded_together()
    test_progress_does_not_overwrite_then_later_nonprogress_updates()
    test_current_observation_always_latest_regardless_of_progress()
    test_replan_directive_contains_full_original_task()
    test_replan_directive_contains_nonprogress_action_and_env_response()
    test_replan_directive_contains_current_observation()
    test_replan_directive_explicitly_forbids_repeating_the_nonprogress_action()
    test_replan_directive_requires_one_line_plan_and_one_action()
    test_replan_window_progress_not_mislabeled_on_next_directive()
    test_missing_optional_state_degrades_safely_no_none_no_blank_no_fabrication()
    test_reset_clears_all_three_fields()
    test_reflect_and_verify_directives_byte_identical_to_v1()
    test_real_prompt_construction_path_end_to_end()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
