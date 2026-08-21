"""Tests for the REPLAN sustained-evidence gate (see audit_reports/
ae_reflect_replan_formula_redesign*.md).

Only ONE routing judgment changed in _select_intervention(): the
frustration_high+confidence_low branch no longer returns REPLAN
unconditionally -- it additionally requires
steps_since_meaningful_change >= patch_duration_steps (an already-existing,
unmodified state field, reused unchanged) before returning REPLAN; when
not yet sustained, it falls through to the (also unchanged) invalid_action /
frustration_medium+repeated REFLECT check. No new config field, no new
state, no change to frustration/confidence weights/decay/thresholds, no
change to REFLECT's or VERIFY's own conditions, no change to the
_ESCALATION scheduling mechanism, and no use of unresolved_intervention_count
(deliberately NOT used -- see audit addendum: it is not type-specific and
never clears mid-episode).

Run directly:
    python3 ae/controllers/tests/test_replan_sustained_evidence.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "alfworld_runs_ae"))

from ae.core import InterventionType  # noqa: E402
from ae.controllers.config import AEConfig  # noqa: E402
from ae.controllers.stateful_controller import StatefulController  # noqa: E402

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


def _base_cfg(**overrides):
    kwargs = dict(warmup_steps=3, cooldown_steps=0, max_interventions=20, patch_duration_steps=3,
                   post_intervention_repeat_enabled=False)
    kwargs.update(overrides)
    return AEConfig(**kwargs)


# ── 1: frustration_high+confidence_low+repeated, ssmc < patch_duration -> REFLECT ──
def test_not_yet_sustained_repeat_routes_to_reflect():
    """Two warmup-suppressed invalid actions (with distinct, non-'nothing
    happens' observation text, so local_state_change_proxy=1 keeps
    resetting steps_since_meaningful_change to 0 for those steps) push
    frustration/confidence toward saturation without accumulating
    sustained-stagnation steps; the first, then a genuine repeat of "go to
    shelf 1" reaches frustration_high+confidence_low+repeated while
    steps_since_meaningful_change is still only 1 (< patch_duration_steps=3)."""
    cfg = _base_cfg()
    controller = StatefulController(cfg)
    recent_a, recent_o = [], []
    r = None
    for act, obs in [
        ("do invalid thing", "You can't do that."),
        ("do another invalid", "That doesn't work here."),
        ("go to shelf 1", "Nothing happens."),
        ("go to shelf 1", "Nothing happens."),
    ]:
        ssmc_before = controller.steps_since_meaningful_change
        r = controller.step(**_step_kwargs(act, obs, admissible=("go to shelf 1",),
                                            recent_actions=list(recent_a), recent_observations=list(recent_o)))
        recent_a.append(act)
        recent_o.append(obs)
    check("1a. frustration_high and confidence_low actually reached (real gate, not vacuous)",
          r["state_signature"]["frustration_high"] and r["state_signature"]["confidence_low"])
    check("1b. steps_since_meaningful_change was 1 (< patch_duration_steps=3) at decision time",
          ssmc_before == 1, ssmc_before)
    check("1c. routes to REFLECT, not REPLAN, since not yet sustained",
          r["intervention"] == InterventionType.REFLECT.value, r["intervention"])
    check("1d. reason is the unmodified frustration_medium+repeated REFLECT reason",
          "frustration_medium" in r["intervention_reason"] and "consecutive_exact_action_repeat" in r["intervention_reason"],
          r["intervention_reason"])


# ── 2: frustration_high+confidence_low+repeated, ssmc == patch_duration -> REPLAN ──
def test_exactly_sustained_repeat_routes_to_replan():
    """Same setup as test 1, but with one extra non-repeating no-progress
    step ('go to shelf 2') inserted before the repeat, so
    steps_since_meaningful_change reaches exactly 3 (== patch_duration_steps)
    at the moment the repeat is detected."""
    cfg = _base_cfg()
    controller = StatefulController(cfg)
    recent_a, recent_o = [], []
    r = None
    ssmc_before = None
    for act, obs in [
        ("do invalid thing", "You can't do that."),
        ("do another invalid", "That doesn't work here."),
        ("go to shelf 1", "Nothing happens."),
        ("go to shelf 2", "Nothing happens."),
        ("go to shelf 1", "Nothing happens."),
        ("go to shelf 1", "Nothing happens."),
    ]:
        ssmc_before = controller.steps_since_meaningful_change
        r = controller.step(**_step_kwargs(act, obs, admissible=("go to shelf 1", "go to shelf 2"),
                                            recent_actions=list(recent_a), recent_observations=list(recent_o)))
        recent_a.append(act)
        recent_o.append(obs)
    check("2a. frustration_high and confidence_low actually reached",
          r["state_signature"]["frustration_high"] and r["state_signature"]["confidence_low"])
    check("2b. steps_since_meaningful_change was exactly 3 (== patch_duration_steps) at decision time",
          ssmc_before == 3, ssmc_before)
    check("2c. routes to REPLAN once sustained (>=, not only >)",
          r["intervention"] == InterventionType.REPLAN.value, r["intervention"])
    check("2d. reason explicitly names the sustained-evidence gate",
          "sustained(steps_since_meaningful_change=3" in r["intervention_reason"],
          r["intervention_reason"])


# ── 3: invalid_action, not sustained -> REFLECT (unaffected by the new gate) ──
def test_invalid_action_not_sustained_routes_to_reflect():
    # warmup_steps=0: warmup is a separate, unmodified gate unrelated to
    # what this test checks; leaving it at the default would suppress the
    # very first step regardless of candidate and hide the thing under test.
    cfg = _base_cfg(warmup_steps=0)
    controller = StatefulController(cfg)
    r = controller.step(**_step_kwargs("foo", "Nothing happens.", admissible=("go to bed 1",),
                                        recent_actions=[], recent_observations=[]))
    check("3a. fresh controller, plain invalid_action still routes REFLECT, "
          "reason=='invalid_action', byte-identical to unmodified behavior",
          r["intervention"] == InterventionType.REFLECT.value and r["intervention_reason"] == "invalid_action",
          (r["intervention"], r["intervention_reason"]))


def test_invalid_action_alongside_high_confidence_low_not_sustained_still_reflect():
    """invalid_action co-occurring with a freshly-reached frustration_high+
    confidence_low (not yet sustained, no repeat) must still route REFLECT
    via the untouched `invalid` short-circuit -- confirms the new REPLAN
    gate does not block or alter the invalid_action path at all."""
    cfg = _base_cfg(warmup_steps=0)
    controller = StatefulController(cfg)
    recent_a, recent_o = [], []
    r = None
    for i, act in enumerate(["bad action one", "bad action two"]):
        obs = f"Nonsense response {i}."
        ssmc_before = controller.steps_since_meaningful_change
        r = controller.step(**_step_kwargs(act, obs, admissible=("go to bed 1",),
                                            recent_actions=list(recent_a), recent_observations=list(recent_o)))
        recent_a.append(act)
        recent_o.append(obs)
    check("3b. frustration_high and confidence_low reached via repeated invalid actions",
          r["state_signature"]["frustration_high"] and r["state_signature"]["confidence_low"])
    check("3c. steps_since_meaningful_change not yet sustained at decision time",
          ssmc_before < 3, ssmc_before)
    check("3d. still routes REFLECT via invalid_action, unaffected by the new REPLAN gate",
          r["intervention"] == InterventionType.REFLECT.value and r["intervention_reason"] == "invalid_action",
          (r["intervention"], r["intervention_reason"]))


# ── 4: sustained no-progress alone can trigger REPLAN directly, no prior REFLECT required ──
def test_sustained_replan_fires_without_any_prior_reflect():
    """Re-uses test 2's exact scenario -- confirms that at no point before
    the final REPLAN did any REFLECT ever get armed (intervention_counts
    stays at 0 for reflect right up to the REPLAN arm), i.e. REPLAN does
    NOT require a preceding REFLECT to have fired first."""
    cfg = _base_cfg()
    controller = StatefulController(cfg)
    recent_a, recent_o = [], []
    r = None
    for act, obs in [
        ("do invalid thing", "You can't do that."),
        ("do another invalid", "That doesn't work here."),
        ("go to shelf 1", "Nothing happens."),
        ("go to shelf 2", "Nothing happens."),
        ("go to shelf 1", "Nothing happens."),
        ("go to shelf 1", "Nothing happens."),
    ]:
        r = controller.step(**_step_kwargs(act, obs, admissible=("go to shelf 1", "go to shelf 2"),
                                            recent_actions=list(recent_a), recent_observations=list(recent_o)))
        recent_a.append(act)
        recent_o.append(obs)
    check("4a. final intervention is REPLAN", r["intervention"] == InterventionType.REPLAN.value)
    check("4b. no REFLECT was ever armed in this whole sequence (intervention_counts['reflect']==0)",
          controller.intervention_counts.get("reflect", 0) == 0, controller.intervention_counts)
    check("4c. REPLAN is armed as the controller's very first intervention (intervention_count==1)",
          controller.intervention_count == 1, controller.intervention_count)
    check("4d. current_escalated_from is None (a fresh trigger, not an _ESCALATION-scheduled arm)",
          controller.current_escalated_from is None, controller.current_escalated_from)


# ── 5a: VERIFY completely unaffected ──
def test_verify_routing_completely_unaffected():
    cfg = _base_cfg()
    controller = StatefulController(cfg)
    recent_a, recent_o = [], []
    r = None
    for i in range(8):
        obs = f"You see a completely different thing number {i}."
        r = controller.step(**_step_kwargs(f"go to shelf {i+1}", obs, admissible=(f"go to shelf {i+1}",),
                                            recent_actions=list(recent_a), recent_observations=list(recent_o)))
        recent_a.append(f"go to shelf {i+1}")
        recent_o.append(obs)
        if r["state_signature"]["uncertainty_high"]:
            break
    check("5a. VERIFY still fires on uncertainty_high, unaffected by the new REPLAN gate",
          r["intervention"] == InterventionType.VERIFY.value, r["intervention"])


# ── 5b: pre-existing _ESCALATION mechanism unaffected (invalid_action-driven REFLECT,
#     unresolved at patch expiry, escalates to REPLAN via the untouched _ESCALATION dict) ──
def test_existing_escalation_mechanism_unaffected():
    cfg = _base_cfg(cooldown_steps=3, patch_duration_steps=2)
    controller = StatefulController(cfg)
    recent_a, recent_o = [], []
    r = None
    # invalid_action fires REFLECT immediately (warmup=3 needs lifting first)
    for act, obs in [("x1", "no."), ("x2", "no."), ("x3", "no."), ("bad action", "Invalid.")]:
        r = controller.step(**_step_kwargs(act, obs, admissible=("go to shelf 1",),
                                            recent_actions=list(recent_a), recent_observations=list(recent_o)))
        recent_a.append(act)
        recent_o.append(obs)
    check("5b1. REFLECT armed via invalid_action", r["intervention"] == InterventionType.REFLECT.value)
    # keep failing through patch expiry (patch_duration_steps=2). Observation
    # text must normalize to exactly "nothing happens." (with an unchanged
    # admissible set) for local_state_change_proxy==0 -- any other text
    # (even for an invalid action) registers as a meaningful change and
    # would wrongly resolve the pending REFLECT as "recovered".
    for act in ["still invalid A", "still invalid B", "still invalid C"]:
        r = controller.step(**_step_kwargs(act, "Nothing happens.", admissible=("go to shelf 1",),
                                            recent_actions=recent_a, recent_observations=recent_o))
        recent_a.append(act); recent_o.append("Nothing happens.")
    check("5b2. REFLECT resolved as 'unresolved' via the unchanged patch-expiry evaluation",
          controller.unresolved_intervention_count >= 1, controller.unresolved_intervention_count)
    # wait for cooldown, confirm scheduled escalation fires to REPLAN via the untouched _ESCALATION dict
    escalated = False
    for act in ["waiting1", "waiting2", "waiting3", "waiting4", "waiting5"]:
        r = controller.step(**_step_kwargs(act, "Nothing happens.", admissible=("go to shelf 1",),
                                            recent_actions=recent_a, recent_observations=recent_o))
        recent_a.append(act); recent_o.append("Nothing happens.")
        if r["intervention"] == InterventionType.REPLAN.value and r["escalated_from"] == "reflect":
            escalated = True
            break
    check("5b3. eventually escalates REFLECT->REPLAN via the pre-existing, unmodified "
          "_ESCALATION mechanism (escalated_from=='reflect', not the new sustained-evidence path)",
          escalated)


if __name__ == "__main__":
    test_not_yet_sustained_repeat_routes_to_reflect()
    test_exactly_sustained_repeat_routes_to_replan()
    test_invalid_action_not_sustained_routes_to_reflect()
    test_invalid_action_alongside_high_confidence_low_not_sustained_still_reflect()
    test_sustained_replan_fires_without_any_prior_reflect()
    test_verify_routing_completely_unaffected()
    test_existing_escalation_mechanism_unaffected()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
