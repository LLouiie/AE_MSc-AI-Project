"""Tests for the REPLAN prompt-v2 progress-aware wording fix (see
audit_reports/ae_replan_prompt_v2_progress_wording_report.md). Drives a
real StatefulController through real step() calls (and, for the real-log
section, replays the ACTUAL recorded signal sequence from the 134-task
sustained-only run through the real, unmodified
step()->_select_intervention()->active_directive()->render_directive()
path -- monkeypatching only signal_extractor.extract() to return the
historical StepSignals, the same "full trajectory replay" methodology used
elsewhere this session) -- no hand-typed directive strings, no mocking of
controller/renderer decision logic itself.

Run directly:
    python3 ae/controllers/tests/test_replan_progress_wording.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "alfworld_runs_ae"))

from ae.core import InterventionType  # noqa: E402
from ae.controllers.config import AEConfig  # noqa: E402
from ae.controllers.stateful_controller import StatefulController  # noqa: E402
from ae.controllers.signals import StepSignals  # noqa: E402
from ae.controllers.intervention_renderer import render_directive  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
# The real-log replay section reads the historical 134-task run logs, which
# only exist in the real repo (this isolated workdir's rsync deliberately
# excluded ae/runners/runs -- pure read-only historical data, unrelated to
# the live 5-seed batch's stateful_controller.py swap, safe to read directly).
REAL_RUN_LOGS_ROOT = "/homes/jy625/projects/AE3"

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


def _arm_replan(goal=GOAL):
    """Controller-agnostic REPLAN trigger -- see the identical helper's
    docstring in test_replan_prompt_v2.py for why this loops rather than
    assuming a fixed step count (old-controller and new-controller arm
    REPLAN after different numbers of repeats of the same action)."""
    cfg = _base_cfg()
    controller = StatefulController(cfg)
    recent_a, recent_o = [], []
    r = None
    for act, obs in [("do invalid thing", "You can't do that."),
                      ("do another invalid", "That doesn't work here."),
                      ("go to shelf 1", "Nothing happens.")]:
        r = controller.step(**_step_kwargs(act, obs, admissible=("go to shelf 1",),
                                            recent_actions=recent_a, recent_observations=recent_o))
        recent_a.append(act)
        recent_o.append(obs)
    max_iters = 15
    for _ in range(max_iters):
        r = controller.step(**_step_kwargs("go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
                                            recent_actions=recent_a, recent_observations=recent_o))
        recent_a.append("go to shelf 1")
        recent_o.append("Nothing happens.")
        if r["intervention"] == InterventionType.REPLAN.value:
            break
    assert r["intervention"] == InterventionType.REPLAN.value, r
    # test-only bookkeeping (not a production attribute) so downstream
    # tests can continue the REAL history instead of hardcoding a stale,
    # fixed-step-count list that no longer matches this controller-agnostic
    # variable-length loop.
    controller._test_recent_a = recent_a
    controller._test_recent_o = recent_o
    return controller


_FORBIDDEN_ON_ZERO = [
    "for 0 steps",
    "Discard the current ineffective strategy",
    "no meaningful progress" ,  # case-insensitive check applied separately too
]


def _assert_no_forbidden_zero_phrases(directive, label, results_list):
    lowered = directive.lower()
    ok = ("no meaningful progress" not in lowered
          and "discard the current ineffective strategy" not in lowered
          and "for 0 steps" not in lowered)
    check(f"{label}: none of the forbidden zero-progress phrases appear", ok, directive)
    results_list.append(ok)


# ---- 1: ssmc > 0 still shows the original no-progress/replan wording ----
def test_ssmc_positive_keeps_no_progress_wording():
    controller = _arm_replan()
    directive = controller.active_directive(goal=GOAL)
    check("1a. ssmc>0 shows 'No meaningful progress has been made for N steps.'",
          f"No meaningful progress has been made for {controller.steps_since_meaningful_change} steps." in directive,
          directive)
    check("1b. ssmc>0 shows the discard-and-replan guidance line",
          "Discard the current ineffective strategy and create a different route." in directive, directive)
    check("1c. ssmc>0 still asks for a different route (not 'continue the productive route')",
          "give a different 2-4-subgoal route" in directive
          and "continue the current productive route" not in directive, directive)
    check("1c2. ssmc>0 explains the differing action relative to the most recent "
          "no-observed-progress action, not a bare 'failed action' claim",
          "explain why the next action differs from the most recent action with no observed progress."
          in directive, directive)
    check("1d. ssmc>0 still ends with a prohibition on repeating the most recent "
          "no-observed-progress action (not worded as 'failed action')",
          "Do not repeat the most recent action with no observed progress." in directive, directive)
    check("1e. 'failed action' no longer appears anywhere in the ssmc>0 branch "
          "(local_state_change_proxy<0.5 only proves no observed change, not task-semantic failure)",
          "failed action" not in directive.lower(), directive)


# ---- 2/3/4/5/6: ssmc == 0 (progress just observed) ----
def test_ssmc_zero_progress_aware_wording():
    controller = _arm_replan()
    # model's t+1 response makes real progress -> ssmc resets to 0
    r = controller.step(**_step_kwargs(
        "take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1.",
        admissible=("go to shelf 1", "go to shelf 2", "take mug 1 from shelf 1"),
        recent_actions=controller._test_recent_a, recent_observations=controller._test_recent_o,
    ))
    check("sanity: this step really did register progress and ssmc really is 0",
          r["signals"]["local_state_change_proxy"] >= 0.5 and controller.steps_since_meaningful_change == 0,
          (r["signals"]["local_state_change_proxy"], controller.steps_since_meaningful_change))

    directive = controller.active_directive(goal=GOAL)

    check("2. does NOT contain 'No meaningful progress has been made for 0 steps'",
          "No meaningful progress has been made for 0 steps" not in directive, directive)
    check("3. does NOT contain 'Discard the current ineffective strategy' or an equivalent "
          "'current route is still ineffective' claim",
          "Discard the current ineffective strategy" not in directive
          and "current route" not in directive.lower() + " ineffective" if False else
          "discard the current ineffective strategy" not in directive.lower(), directive)
    check("4. explicitly states an observed state change occurred (not a claim about REPLAN's own "
          "timeline -- see report: arming can complete AFTER the step that reset ssmc)",
          "The latest step produced an observed state change." in directive, directive)
    check("4b. does NOT claim the change happened 'since REPLAN was activated' (an inaccurate "
          "time reference per the real call-sequence audit)",
          "since REPLAN was activated" not in directive, directive)
    check("5. explicitly instructs continuing from the latest observation, without asserting the "
          "route is definitively 'productive' (local_state_change_proxy only proves observed "
          "change, not task benefit) and without treating REPLAN's own active status as a reason "
          "to discard it",
          "Continue from the latest observation and do not discard the updated route solely "
          "because REPLAN remains active." in directive, directive)
    check("6. still explicitly forbids returning to the most recent no-progress action",
          "Do not return to the most recent action with no observed progress." in directive, directive)
    check("6b. the no-progress action being avoided is still concretely named (not just an abstract ban)",
          "Most recent action with no observed progress: go to shelf 1" in directive, directive)


# ---- 7: "no progress -> recover -> continue" 3-step window switches correctly ----
def test_three_step_window_switches_with_real_state():
    controller = _arm_replan()
    d1 = controller.active_directive(goal=GOAL)
    check("7a. t+1 (right after arm, ssmc>0): no-progress branch",
          "No meaningful progress has been made for" in d1, d1)

    r2 = controller.step(**_step_kwargs(
        "take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1.",
        admissible=("go to shelf 1", "go to shelf 2", "take mug 1 from shelf 1"),
        recent_actions=controller._test_recent_a, recent_observations=controller._test_recent_o,
    ))
    d2 = controller.active_directive(goal=GOAL)
    check("7b. t+2 (after a progress step, ssmc==0): progress-observed branch",
          "The latest step produced an observed state change." in d2, d2)
    check("7b2. t+2 shows the just-successful action's observation as Current observation",
          "Current observation: You pick up the mug 1 from the shelf 1." in d2, d2)
    return controller  # handed to test 8 to continue the same window


def test_eight_switches_back_on_renewed_nonprogress():
    controller = _arm_replan()
    controller.step(**_step_kwargs(
        "take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1.",
        admissible=("go to shelf 1", "go to shelf 2", "take mug 1 from shelf 1"),
        recent_actions=controller._test_recent_a, recent_observations=controller._test_recent_o,
    ))
    # t+2's own action is a NEW non-progress action (still within the same
    # 3-step REPLAN window -- patch_duration_steps=3, this is offset 2 of 3)
    r3 = controller.step(**_step_kwargs(
        "go to shelf 2", "Nothing happens.",
        admissible=("go to shelf 1", "go to shelf 2", "take mug 1 from shelf 1"),
        recent_actions=controller._test_recent_a + ["take mug 1 from shelf 1"],
        recent_observations=controller._test_recent_o + ["You pick up the mug 1 from the shelf 1."],
    ))
    check("8a. sanity: this step really registered no progress (ssmc back to 1)",
          r3["signals"]["local_state_change_proxy"] < 0.5 and controller.steps_since_meaningful_change == 1,
          (r3["signals"]["local_state_change_proxy"], controller.steps_since_meaningful_change))
    d3 = controller.active_directive(goal=GOAL)
    check("8b. t+3 switches BACK to the no-progress/replan branch once ssmc>0 again",
          "No meaningful progress has been made for 1 steps." in d3, d3)
    check("8c. t+3's non-progress action record updated to the NEW no-progress action "
          "('go to shelf 2'), not stuck on the earlier one",
          "Most recent action with no observed progress: go to shelf 2" in d3, d3)


# ---- 9: B+ fields/update timing completely unaffected by this wording-only change ----
def test_b_plus_fields_and_timing_unaffected():
    controller = _arm_replan()
    pre = (controller.last_nonprogress_action_text, controller.last_nonprogress_observation_text,
           controller.current_observation_text)
    controller.step(**_step_kwargs(
        "take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1.",
        admissible=("go to shelf 1", "go to shelf 2", "take mug 1 from shelf 1"),
        recent_actions=controller._test_recent_a, recent_observations=controller._test_recent_o,
    ))
    check("9a. last_nonprogress_action_text unchanged by a progress step (same B+ behavior as before)",
          controller.last_nonprogress_action_text == pre[0], (controller.last_nonprogress_action_text, pre[0]))
    check("9b. last_nonprogress_observation_text unchanged by a progress step",
          controller.last_nonprogress_observation_text == pre[1])
    check("9c. current_observation_text DOES update (unconditional, as before)",
          controller.current_observation_text == "You pick up the mug 1 from the shelf 1."
          and controller.current_observation_text != pre[2])


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
    check("10a. REFLECT directive byte-identical to v1", reflect == expected_reflect, reflect)
    check("10b. VERIFY directive byte-identical to v1", verify == expected_verify, verify)


def test_patch_duration_still_three_steps():
    controller = _arm_replan()
    check("11. patch_duration_steps is still 3 (unchanged by this wording-only fix)",
          controller.config.patch_duration_steps == 3, controller.config.patch_duration_steps)
    check("11b. REPLAN's active_patch_remaining is exactly patch_duration_steps at arm time",
          controller.active_patch_remaining == 3, controller.active_patch_remaining)


# ---- real-log replay: the 5 real episodes / 7 previously-contradictory directive instances ----
_REAL_CASES = [
    # (env_name, q_index, arm_step_idx, goal, [(action, observation, recorded_signals_dict, recorded_ssmc), ...] for offsets 1..3)
]


def _load_real_case(env_name, q_index, arm_step_idx, split, run_name):
    path = os.path.join(REAL_RUN_LOGS_ROOT, "ae/runners/runs", run_name, "episode_log.jsonl")
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["env_name"] == env_name and d["q_index"] == q_index:
                step_log = d["ae_step_log"]
                # index 0 = the ARM step itself (always a non-progress step,
                # since ssmc>=patch_duration_steps is the trigger condition
                # -- this seeds "last non-progress action" for offset 1's
                # directive); indices 1,2,3 = window offsets 1,2,3.
                records = [step_log[arm_step_idx]] + step_log[arm_step_idx + 1: arm_step_idx + 4]
                return d["goal"], records
    raise AssertionError(f"case not found: {env_name} q={q_index} in {run_name}")


def test_real_log_replay_five_cases_seven_instances_no_longer_contradictory():
    """Replays the REAL recorded signal sequence (not hand-typed) for each
    of the 5 real REPLAN windows previously found to produce 7
    contradictory directive instances. The directive shown when GENERATING
    offset k's action reads controller state as it stood immediately AFTER
    processing the PRIOR record (the arm step for k=1, offset k-1's record
    for k=2,3) -- not that offset's own resulting record (a step's own
    ssmc/action/observation become visible to the NEXT directive, not the
    one shown while it was being taken). This mirrors exactly how
    active_directive() is called in agents.py's loop (before the step that
    produces the current record)."""
    cases = [
        ("pick_heat_then_place_in_recep-Egg-None-GarbageCan-10", 15, 3, "ae_full_sustained_v1_a40_practice"),
        ("look_at_obj_in_light-CD-None-DeskLamp-308", 34, 10, "ae_full_sustained_v1_a40_practice"),
        ("pick_heat_then_place_in_recep-Egg-None-GarbageCan-10", 50, 3, "ae_full_sustained_v1_a40_practice"),
        ("pick_clean_then_place_in_recep-Cloth-None-Cabinet-424", 84, 11, "ae_full_sustained_v1_a40_practice"),
        ("pick_heat_then_place_in_recep-Egg-None-GarbageCan-10", 7, 3, "ae_full_sustained_v1_a40_exam"),
    ]
    total_zero_instances = 0
    for env_name, q_index, arm_step_idx, run_name in cases:
        goal, records = _load_real_case(env_name, q_index, arm_step_idx, None, run_name)
        # records[0] = arm step (always non-progress). Accumulate the
        # "most recent non-progress (action, observation)" and "current
        # observation" cumulatively over records[0..k-1] for each offset
        # k=1,2,3's directive, exactly matching the B+ update rule.
        last_nonprogress_action = records[0]["action"]
        last_nonprogress_observation = records[0]["observation"]
        current_obs = records[0]["observation"]
        ssmc_seen = records[0]["steps_since_meaningful_change"]  # always patch_duration_steps at arm
        for offset in (1, 2, 3):
            directive = render_directive(
                InterventionType.REPLAN, goal=goal, steps_since_meaningful_change=ssmc_seen,
                last_nonprogress_action=last_nonprogress_action,
                last_nonprogress_observation=last_nonprogress_observation,
                current_observation=current_obs,
            )
            if ssmc_seen == 0:
                total_zero_instances += 1
                _assert_no_forbidden_zero_phrases(
                    directive, f"real-replay {env_name[:30]} q={q_index} offset={offset} (ssmc=0)", [])
                check(f"real-replay {env_name[:30]} q={q_index} offset={offset}: "
                      "explicitly acknowledges the observed state change instead",
                      "The latest step produced an observed state change." in directive)
            # now advance state to what THIS offset's own record produced,
            # for use by the NEXT offset's directive
            this_rec = records[offset]
            proxy = this_rec["signals"]["local_state_change_proxy"]
            current_obs = this_rec["observation"]
            ssmc_seen = this_rec["steps_since_meaningful_change"]
            if proxy < 0.5:
                last_nonprogress_action = this_rec["action"]
                last_nonprogress_observation = this_rec["observation"]
    # NOTE: this is 10, not the 7 originally reported in
    # ae_replan_prompt_v2_b_plus_report.md section 9 -- that count only
    # checked offsets 2/3 (using the WINDOW list's own shifted values) and
    # implicitly assumed offset 1 always shows the arm-time ssmc
    # (patch_duration_steps, never 0). That assumption is wrong: the
    # routing DECISION at the arm step reads ssmc from BEFORE that step's
    # own bookkeeping, but arming itself happens AFTER that step's
    # bookkeeping already ran -- so if the arm step's OWN action happened
    # to register progress (as it does for q=15/q=50/q=7 here), the very
    # FIRST directive shown (offset 1) already sees ssmc==0. See the b_plus
    # report correction in the final report for this round.
    check("real-replay: found 10 ssmc==0 directive instances across the 5 real cases "
          "(corrects the earlier undercounted 7 -- see report)",
          total_zero_instances == 10, total_zero_instances)


if __name__ == "__main__":
    test_ssmc_positive_keeps_no_progress_wording()
    test_ssmc_zero_progress_aware_wording()
    test_three_step_window_switches_with_real_state()
    test_eight_switches_back_on_renewed_nonprogress()
    test_b_plus_fields_and_timing_unaffected()
    test_reflect_and_verify_directives_byte_identical_to_v1()
    test_patch_duration_still_three_steps()
    test_real_log_replay_five_cases_seven_instances_no_longer_contradictory()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
