"""Tests for the two new PIR ablation config toggles added on top of the
post-intervention exact-repeat rule (see test_post_intervention_exact_repeat.py
for the original rule's own tests, all of which must keep passing unchanged
since both new toggles default to preserving that exact original behavior):

    post_intervention_repeat_enabled (default True)
        False -> the entire override is disabled; VERIFY/REFLECT/REPLAN
        active-intervention exact repeats fall through to normal
        event-detection/warmup/cooldown/budget processing, exactly as if
        the rule had never been added ("v2.5" frozen baseline).

    post_intervention_repeat_reflect_double (default False)
        True -> ONLY changes REFLECT's branch: the first consecutive exact
        repeat while REFLECT is pending is absorbed (no escalation, one
        more recovery chance); only a second, immediately-consecutive
        exact repeat escalates to REPLAN. VERIFY and REPLAN branches are
        completely unaffected by this flag ("exploratory ablation C").

This file does not touch frustration/affect weights/decay/warmup/budget
values, and does not modify the original rule's behavior under the
defaults. Run directly:
    python3 ae/controllers/tests/test_pir_ablation_modes.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "alfworld_runs_ae"))

from ae.core import InterventionType  # noqa: E402
from ae.controllers.config import AEConfig, HysteresisBand, load_config  # noqa: E402
from ae.controllers.stateful_controller import StatefulController  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
NOPIR_YAML = os.path.join(REPO_ROOT, "configs", "controllers", "ae_full_final_nopir_v2_5.yaml")
C_YAML = os.path.join(REPO_ROOT, "configs", "controllers", "ae_full_reflect2repeat_v2_6_c.yaml")

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def _step_kwargs(action, observation, admissible=("x",), recent_actions=(), recent_observations=(),
                  reward=0, done=False, won=False):
    return dict(
        action=action, observation=observation,
        admissible_before=list(admissible), admissible_after=list(admissible),
        recent_actions=list(recent_actions), recent_observations=list(recent_observations),
        max_steps=50, is_think_action=False, reward=reward, done=done, won=won,
    )


# NOTE: frustration_medium=1.1 (unreachable, since frustration is clipped to
# [0,1]) is used throughout this file's isolated single/two-step scenarios
# to keep the NORMAL event-detection/edge-detection path below the PIR
# override from independently re-arming or escalating an intervention on
# its own (a real, pre-existing controller property: on the very first
# real step() call after a controller-level intervention is armed directly
# via _arm_intervention() in a test, previous_signature is still None, so
# ANY already-active band flag -- including frustration_medium AND
# consecutive_exact_action_repeat, the same gate the frustration-fix uses
# -- counts as a fresh rising edge and can independently fire a NEW
# intervention through the ordinary is_event path, separate from and in
# addition to whatever PIR itself decides). This is not something this
# ablation-mode change introduces; the original test_post_intervention_
# exact_repeat.py tests never hit it because PIR firing there always
# returns immediately, before this code path is ever reached -- but tests
# in THIS file exercise the "no PIR override fires" case on purpose (PIR
# disabled, or a REFLECT repeat absorbed rather than escalated), so the
# normal path underneath genuinely does run and must be neutralized to
# isolate what THIS file is actually testing.


# ── 0: yaml files parse into the expected flag combinations ────────────
def test_yaml_configs_parse_expected_toggle_values():
    nopir_cfg = load_config(NOPIR_YAML)
    c_cfg = load_config(C_YAML)
    check("0a. nopir yaml: post_intervention_repeat_enabled is False",
          nopir_cfg.post_intervention_repeat_enabled is False)
    check("0b. nopir yaml: post_intervention_repeat_reflect_double is False (irrelevant, but default)",
          nopir_cfg.post_intervention_repeat_reflect_double is False)
    check("0c. C yaml: post_intervention_repeat_enabled is True",
          c_cfg.post_intervention_repeat_enabled is True)
    check("0d. C yaml: post_intervention_repeat_reflect_double is True",
          c_cfg.post_intervention_repeat_reflect_double is True)
    # every other field must be identical between the two configs
    import dataclasses
    nopir_d = dataclasses.asdict(nopir_cfg)
    c_d = dataclasses.asdict(c_cfg)
    del nopir_d["post_intervention_repeat_enabled"], nopir_d["post_intervention_repeat_reflect_double"]
    del c_d["post_intervention_repeat_enabled"], c_d["post_intervention_repeat_reflect_double"]
    check("0e. every non-PIR field is byte-identical between the two configs",
          nopir_d == c_d)
    check("0f. default AEConfig() also has PIR enabled, reflect_double False (v3 back-compat)",
          AEConfig().post_intervention_repeat_enabled is True
          and AEConfig().post_intervention_repeat_reflect_double is False)


# ── 1: PIR fully disabled -- VERIFY/REFLECT/REPLAN repeats never override ──
def test_disabled_verify_repeat_falls_through():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5, frustration_medium=1.1,
                    post_intervention_repeat_enabled=False)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.VERIFY, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("1. PIR disabled: active VERIFY + exact repeat does NOT escalate to REFLECT",
          r["intervention"] != InterventionType.REFLECT.value or r["post_intervention_exact_repeat"] is None,
          r["intervention"])
    check("1b. PIR disabled: no post_intervention_exact_repeat log entry at all",
          r["post_intervention_exact_repeat"] is None)
    check("1c. PIR disabled: force_terminate_reason never set", controller.force_terminate_reason is None)
    check("1d. PIR disabled: VERIFY is still the active/pending type (no forced change)",
          controller.current_intervention_type == InterventionType.VERIFY)


def test_disabled_reflect_repeat_falls_through():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5, frustration_medium=1.1,
                    post_intervention_repeat_enabled=False)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("2. PIR disabled: active REFLECT + exact repeat does NOT escalate to REPLAN",
          controller.current_intervention_type == InterventionType.REFLECT)
    check("2b. PIR disabled: no post_intervention_exact_repeat log entry at all",
          r["post_intervention_exact_repeat"] is None)
    check("2c. PIR disabled: force_terminate_reason never set", controller.force_terminate_reason is None)


def test_disabled_replan_repeat_does_not_terminate():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5, frustration_medium=1.1,
                    post_intervention_repeat_enabled=False)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REPLAN, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("3. PIR disabled: active REPLAN + exact repeat does NOT force-terminate",
          controller.force_terminate_reason is None, controller.force_terminate_reason)
    check("3b. PIR disabled: no post_intervention_exact_repeat log entry at all",
          r["post_intervention_exact_repeat"] is None)
    check("3c. PIR disabled: episode/controller keeps running normally (still pending REPLAN, "
          "unless the OLD patch-expiry/budget path independently changed it)",
          controller.step_index == 1)


# ── 2: reflect_double=True -- VERIFY and REPLAN branches unaffected ────
def test_c_mode_verify_still_escalates_on_first_repeat():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5, frustration_medium=1.1,
                    post_intervention_repeat_enabled=True, post_intervention_repeat_reflect_double=True)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.VERIFY, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("4. C mode: active VERIFY + first exact repeat still escalates to REFLECT immediately "
          "(unaffected by reflect_double)",
          r["intervention"] == InterventionType.REFLECT.value, r["intervention"])
    check("4b. C mode: transition logged as VERIFY->REFLECT (same as original rule)",
          r["post_intervention_exact_repeat"]["transition"] == "VERIFY->REFLECT")


def test_c_mode_replan_still_terminates_on_first_repeat():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5, frustration_medium=1.1,
                    post_intervention_repeat_enabled=True, post_intervention_repeat_reflect_double=True)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REPLAN, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("5. C mode: active REPLAN + first exact repeat still terminates immediately "
          "(unaffected by reflect_double)",
          controller.force_terminate_reason == "post_replan_exact_repeat", controller.force_terminate_reason)
    check("5b. C mode: transition logged as REPLAN->TERMINATE (same as original rule)",
          r["post_intervention_exact_repeat"]["transition"] == "REPLAN->TERMINATE")


# ── 3: reflect_double=True core behavior -- absorb first, escalate second ──
def test_c_mode_reflect_first_repeat_absorbed_not_escalated():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5, frustration_medium=1.1,
                    post_intervention_repeat_enabled=True, post_intervention_repeat_reflect_double=True)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("6. C mode: first consecutive exact repeat while REFLECT pending is ABSORBED, "
          "not escalated to REPLAN",
          controller.current_intervention_type == InterventionType.REFLECT, controller.current_intervention_type)
    check("6b. C mode: no forced termination on the absorbed first repeat",
          controller.force_terminate_reason is None)
    check("6c. C mode: post_intervention_exact_repeat log entry is None (override did not fire)",
          r["post_intervention_exact_repeat"] is None)
    check("6d. C mode: reflect_repeat_absorbed is True on this step's record",
          r["reflect_repeat_absorbed"] is True)
    check("6e. C mode: REFLECT is still pending (intervention_count never incremented past the original 1)",
          controller.intervention_count == 1, controller.intervention_count)


def test_c_mode_reflect_second_consecutive_repeat_escalates():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5, frustration_medium=1.1,
                    post_intervention_repeat_enabled=True, post_intervention_repeat_reflect_double=True)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)
    r1 = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("7a. first repeat (step 1) absorbed", r1["post_intervention_exact_repeat"] is None
          and r1["reflect_repeat_absorbed"] is True)
    r2 = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1", "go to shelf 1"],
        recent_observations=["Nothing happens.", "Nothing happens."],
    ))
    check("7. C mode: SECOND immediately-consecutive exact repeat while REFLECT still "
          "pending escalates to REPLAN",
          r2["intervention"] == InterventionType.REPLAN.value, r2["intervention"])
    check("7b. C mode: transition logged as REFLECT->REPLAN on the second repeat",
          r2["post_intervention_exact_repeat"]["transition"] == "REFLECT->REPLAN")
    check("7c. C mode: exactly one escalation happened (intervention_count now 2: "
          "original REFLECT + escalated REPLAN)",
          controller.intervention_count == 2, controller.intervention_count)
    check("7d. C mode: the second step's own record shows reflect_repeat_absorbed False "
          "(it escalated, it wasn't absorbed)",
          r2["reflect_repeat_absorbed"] is False)


def test_c_mode_reflect_streak_broken_by_different_action_resets_absorb_flag():
    """a1==a2 (absorbed) -> a3 != a2 (streak broken, no escalation, normal
    processing) -> a4==a3 (repeat again, but the streak restarted, so this
    must be treated as a FRESH first repeat -- absorbed again, not
    escalated) -- confirms _reflect_repeat_pending correctly resets on a
    non-repeat step rather than staying latched for the rest of the
    episode."""
    # frustration_hysteresis also neutralized here (in addition to
    # frustration_medium): this test's 3-step "Nothing happens." sequence
    # is long enough for the pre-existing, PIR-unrelated repeated_action/
    # repeated_observation frustration accumulation to legitimately cross
    # the real frustration_high hysteresis band on its own (confirmed
    # empirically: frustration_high AND confidence_low -> REPLAN fires via
    # the ordinary edge-detection path by step 3), which would confound
    # this test's specific claim about the REFLECT-repeat streak flag.
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=20, frustration_medium=1.1,
                    frustration_hysteresis=HysteresisBand(enter=99.0, exit=99.0),
                    uncertainty_hysteresis=HysteresisBand(enter=99.0, exit=99.0),
                    surprise_hysteresis=HysteresisBand(enter=99.0, exit=99.0),
                    post_intervention_repeat_enabled=True, post_intervention_repeat_reflect_double=True)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)

    r1 = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1", "go to shelf 2"),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("8a. step1 (a1==a0 repeat) absorbed", r1["reflect_repeat_absorbed"] is True)

    r2 = controller.step(**_step_kwargs(
        "go to shelf 2", "Nothing happens.", admissible=("go to shelf 1", "go to shelf 2"),
        recent_actions=["go to shelf 1", "go to shelf 1"],
        recent_observations=["Nothing happens.", "Nothing happens."],
    ))
    check("8b. step2 (different action, streak broken) is NOT absorbed and does NOT escalate",
          r2["reflect_repeat_absorbed"] is False and r2["post_intervention_exact_repeat"] is None,
          (r2["reflect_repeat_absorbed"], r2["post_intervention_exact_repeat"]))
    check("8c. REFLECT still pending after the streak-breaking step",
          controller.current_intervention_type == InterventionType.REFLECT)

    r3 = controller.step(**_step_kwargs(
        "go to shelf 2", "Nothing happens.", admissible=("go to shelf 1", "go to shelf 2"),
        recent_actions=["go to shelf 1", "go to shelf 1", "go to shelf 2"],
        recent_observations=["Nothing happens.", "Nothing happens.", "Nothing happens."],
    ))
    check("9. step3 repeats step2's action -- since the streak had been broken, this is "
          "treated as a FRESH first repeat: absorbed again, NOT escalated",
          r3["reflect_repeat_absorbed"] is True and r3["post_intervention_exact_repeat"] is None,
          (r3["reflect_repeat_absorbed"], r3["post_intervention_exact_repeat"]))
    check("9b. intervention_count still 1 -- no escalation ever happened in this whole sequence",
          controller.intervention_count == 1, controller.intervention_count)


def test_c_mode_budget_exhausted_still_terminates_on_escalation_attempt():
    """With reflect_double on, budget exhaustion must still be respected on
    the SECOND (escalating) repeat -- exactly like the original rule's own
    budget-exhaustion test, just reached one repeat later."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=1, patch_duration_steps=5, frustration_medium=1.1,
                    post_intervention_repeat_enabled=True, post_intervention_repeat_reflect_double=True)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)  # #1, budget now exhausted
    check("10a. budget exhausted right after arming REFLECT (max_interventions=1)",
          controller.intervention_count == 1, controller.intervention_count)

    r1 = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("10b. first repeat still absorbed even with budget already exhausted "
          "(absorption doesn't consult budget -- it's a no-op, not an escalation)",
          r1["reflect_repeat_absorbed"] is True and controller.force_terminate_reason is None)

    r2 = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1", "go to shelf 1"],
        recent_observations=["Nothing happens.", "Nothing happens."],
    ))
    check("11. second consecutive repeat attempts to escalate, budget exhausted -> terminates "
          "via post_intervention_repeat_budget_exhausted (not a 2nd intervention)",
          controller.force_terminate_reason == "post_intervention_repeat_budget_exhausted",
          controller.force_terminate_reason)
    check("11b. intervention_count never exceeds max_interventions=1",
          controller.intervention_count <= 1, controller.intervention_count)


def test_c_mode_env_success_wins_even_mid_absorb():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5, frustration_medium=1.1,
                    post_intervention_repeat_enabled=True, post_intervention_repeat_reflect_double=True)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "You win.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
        won=True, done=True, reward=1,
    ))
    check("12. C mode: env_success suppresses even the absorb path -- no reflect_repeat_absorbed, "
          "no override log entry",
          r["reflect_repeat_absorbed"] is False and r["post_intervention_exact_repeat"] is None,
          (r["reflect_repeat_absorbed"], r["post_intervention_exact_repeat"]))
    check("12b. C mode: force_terminate_reason never set on a real win",
          controller.force_terminate_reason is None)


if __name__ == "__main__":
    test_yaml_configs_parse_expected_toggle_values()
    test_disabled_verify_repeat_falls_through()
    test_disabled_reflect_repeat_falls_through()
    test_disabled_replan_repeat_does_not_terminate()
    test_c_mode_verify_still_escalates_on_first_repeat()
    test_c_mode_replan_still_terminates_on_first_repeat()
    test_c_mode_reflect_first_repeat_absorbed_not_escalated()
    test_c_mode_reflect_second_consecutive_repeat_escalates()
    test_c_mode_reflect_streak_broken_by_different_action_resets_absorb_flag()
    test_c_mode_budget_exhausted_still_terminates_on_escalation_attempt()
    test_c_mode_env_success_wins_even_mid_absorb()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
