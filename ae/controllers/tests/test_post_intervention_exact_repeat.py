"""Tests for the post-intervention exact-repeat rule: hard evidence that a
fired intervention's directive was NOT acted on (the model repeated its
immediately-preceding action verbatim while the intervention is still
awaiting outcome evaluation). This is a strictly higher-priority override
than cooldown/recovery_grace/patch_duration_steps/legacy repeated-action
termination -- see stateful_controller.py::step()'s
post_intervention_exact_repeat block, inserted before the normal
event-detection/warmup/cooldown/budget gating.

    VERIFY  + exact repeat -> escalate to REFLECT
    REFLECT + exact repeat -> escalate to REPLAN
    REPLAN  + exact repeat -> terminate episode (post_replan_exact_repeat)

Deliberately separate from the frustration_medium gate fix
(test_ae_controller.py's F-series / test_signal_semantics.py) -- this file
does not touch frustration/affect weights/decay, warmup, or any config
value. Run directly:
    python3 ae/controllers/tests/test_post_intervention_exact_repeat.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "alfworld_runs_ae"))

from ae.core import InterventionType  # noqa: E402
from ae.controllers.config import AEConfig, load_config  # noqa: E402
from ae.controllers.stateful_controller import StatefulController  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
AE_FULL_YAML = os.path.join(REPO_ROOT, "configs", "controllers", "ae_full.yaml")

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


class FakeLLM:
    def __init__(self, actions):
        self._actions = list(actions)
        self.call_count = 0
        self.model = "Qwen/Qwen3-8B"

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        return self._actions.pop(0) if self._actions else "look"


class ScriptedFakeEnv:
    def __init__(self, script, admissible=("go to sink 1",)):
        self.script = list(script)
        self.admissible = list(admissible)
        self._i = 0
        self.step_calls = 0

    def reset(self):
        self._i = 0
        self.step_calls = 0
        ob = "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in a room.\n\nYour task is to: put a mug in the sink."
        return [ob], {"admissible_commands": [list(self.admissible)], "won": [False]}

    def step(self, actions):
        self.step_calls += 1
        entry = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        ob, reward, done, won = entry
        return [ob], [reward], [done], {"admissible_commands": [list(self.admissible)], "won": [won]}

    def close(self):
        pass


def _step_kwargs(action, observation, admissible=("x",), recent_actions=(), recent_observations=(),
                  reward=0, done=False, won=False):
    return dict(
        action=action, observation=observation,
        admissible_before=list(admissible), admissible_after=list(admissible),
        recent_actions=list(recent_actions), recent_observations=list(recent_observations),
        max_steps=50, is_think_action=False, reward=reward, done=done, won=won,
    )


# ── 1/2/3: the escalation ladder ─────────────────────────────────────────
def test_active_verify_exact_repeat_escalates_to_reflect():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.VERIFY, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("1. active VERIFY + exact repeat escalates to REFLECT immediately",
          r["intervention"] == InterventionType.REFLECT.value, r["intervention"])
    check("1b. the new REFLECT is now itself pending (not resolved)",
          controller.pending_intervention is True and controller.current_intervention_type == InterventionType.REFLECT)
    check("1c. no forced termination (still escalating, not terminating)",
          controller.force_terminate_reason is None)
    check("1d. transition logged explicitly as VERIFY->REFLECT",
          r["post_intervention_exact_repeat"]["transition"] == "VERIFY->REFLECT",
          r["post_intervention_exact_repeat"])
    check("1e. escalation counts toward intervention_count",
          controller.intervention_count == 2, controller.intervention_count)


def test_active_reflect_exact_repeat_escalates_to_replan():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("2. active REFLECT + exact repeat escalates to REPLAN immediately",
          r["intervention"] == InterventionType.REPLAN.value, r["intervention"])
    check("2b. the new REPLAN is now itself pending",
          controller.pending_intervention is True and controller.current_intervention_type == InterventionType.REPLAN)
    check("2c. no forced termination yet", controller.force_terminate_reason is None)
    check("2d. transition logged explicitly as REFLECT->REPLAN",
          r["post_intervention_exact_repeat"]["transition"] == "REFLECT->REPLAN")


def test_active_replan_exact_repeat_terminates():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REPLAN, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("3. active REPLAN + exact repeat terminates the episode",
          controller.force_terminate_reason == "post_replan_exact_repeat", controller.force_terminate_reason)
    check("3b. logged termination_reason field matches",
          r["post_intervention_exact_repeat"]["termination_reason"] == "post_replan_exact_repeat")
    check("3c. transition logged explicitly as REPLAN->TERMINATE",
          r["post_intervention_exact_repeat"]["transition"] == "REPLAN->TERMINATE")
    check("3d. the resolved intervention_outcome is 'unresolved', not left pending",
          controller.current_intervention_outcome == "unresolved")


# ── 4/5: recovery_grace and cooldown cannot block the REPLAN termination ──
def test_replan_repeat_terminates_despite_full_recovery_grace():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20,
                    patch_duration_steps=5, recovery_grace_steps=3)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REPLAN, escalated_from=None)
    check("4a. recovery_grace_remaining is at its full value right after arming",
          controller.recovery_grace_remaining == 3, controller.recovery_grace_remaining)
    controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("4. terminates via post_replan_exact_repeat even with full recovery_grace_remaining=3 available "
          "-- the rule never consults recovery_grace_remaining",
          controller.force_terminate_reason == "post_replan_exact_repeat", controller.force_terminate_reason)


def test_replan_repeat_terminates_despite_active_cooldown():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=10, max_interventions=20, patch_duration_steps=5)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REPLAN, escalated_from=None)
    check("5a. cooldown_remaining is active right after arming",
          controller.cooldown_remaining > 0, controller.cooldown_remaining)
    controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("5. terminates via post_replan_exact_repeat even while cooldown is active "
          "-- the rule never consults cooldown_remaining",
          controller.force_terminate_reason == "post_replan_exact_repeat", controller.force_terminate_reason)


# ── 6/7: canonicalization must not over-trigger (same cases as the
#    consecutive_exact_action_repeat signal tests, re-checked in THIS
#    rule's context: an active intervention must not be force-escalated
#    by a non-repeat) ──────────────────────────────────────────────────
def test_different_shelf_numbers_do_not_trigger_post_intervention_rule():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REPLAN, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 6", "You arrive at shelf 6. On the shelf 6, you see nothing.",
        admissible=("go to shelf 5", "go to shelf 6"),
        recent_actions=["go to shelf 5"],
        recent_observations=["You arrive at shelf 5. On the shelf 5, you see nothing."],
    ))
    check("6. 'go to shelf 5' -> 'go to shelf 6' while REPLAN is active does NOT "
          "force-terminate (different object number, not a real repeat)",
          controller.force_terminate_reason is None and r["post_intervention_exact_repeat"] is None,
          (controller.force_terminate_reason, r["post_intervention_exact_repeat"]))


def test_take_different_object_numbers_do_not_trigger_post_intervention_rule():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REPLAN, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "take soapbar 2 from countertop 1", "You pick up the soapbar 2 from the countertop 1.",
        admissible=("take soapbar 1 from countertop 1", "take soapbar 2 from countertop 1"),
        recent_actions=["take soapbar 1 from countertop 1"],
        recent_observations=["You pick up the soapbar 1 from the countertop 1."],
    ))
    check("7. 'take soapbar 1' -> 'take soapbar 2' while REPLAN is active does NOT "
          "force-terminate (different object, not a real repeat)",
          controller.force_terminate_reason is None and r["post_intervention_exact_repeat"] is None,
          (controller.force_terminate_reason, r["post_intervention_exact_repeat"]))


# ── 8: no active intervention -> legacy behavior completely unaffected ──
def test_no_active_intervention_legacy_behavior_unaffected():
    from agents import ALFWorldAgent
    llm = FakeLLM(["go to nowhere", "go to nowhere"] + ["go to nowhere"] * 60)
    agent = ALFWorldAgent(llm, controller=None, termination_policy="legacy_early_stop")
    env = ScriptedFakeEnv([("Nothing happens.", 0, False, False)], admissible=("go to sink 1",))
    _, success = agent.run(env, "task", "put", to_print=False)
    check("8. controller=None (no active intervention possible at all): legacy "
          "exhausted_repeated behavior is completely unaffected",
          agent.termination_reason == "exhausted_repeated", agent.termination_reason)
    check("8b. success is False", success is False)


# ── 9: non-repeating stuck action still uses the OLD unresolved-escalation
#     path (patch_duration_steps expiry), unaffected by this new rule ────
def test_non_repeating_unresolved_still_escalates_via_old_patch_expiry_path():
    # cooldown_steps>0 (unlike this file's other isolated tests) so the
    # scheduled escalation doesn't fire in the very same call that marks
    # the outcome "unresolved" -- matching how the production config
    # (cooldown_steps=3) actually behaves, and letting this test observe
    # the "unresolved" state distinctly before escalation consumes it.
    cfg = AEConfig(warmup_steps=0, cooldown_steps=3, max_interventions=20, patch_duration_steps=2)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)
    # 3 DISTINCT actions (never an exact repeat of the immediately-
    # preceding one), none showing meaningful change -- must reach patch
    # expiry and auto-escalate via the pre-existing unresolved-outcome
    # mechanism (_ESCALATION), not this new rule.
    steps = [
        ("go to shelf 1", "Nothing happens."),
        ("go to shelf 2", "Nothing happens."),
        ("go to shelf 3", "Nothing happens."),
    ]
    recent_a, recent_o = [], []
    last_r = None
    for action, obs in steps:
        last_r = controller.step(**_step_kwargs(
            action, obs, admissible=("go to shelf 1", "go to shelf 2", "go to shelf 3"),
            recent_actions=list(recent_a), recent_observations=list(recent_o),
        ))
        recent_a.append(action)
        recent_o.append(obs)
    check("9. no post-intervention exact-repeat ever fired (all 3 steps used distinct actions)",
          all(rec["post_intervention_exact_repeat"] is None for rec in controller.step_log))
    check("9b. the old patch-duration-expiry mechanism still evaluated the outcome as unresolved",
          last_r["intervention_outcome"] == "unresolved", last_r["intervention_outcome"])
    check("9c. and still scheduled the pre-existing escalation (reflect -> replan), "
          "waiting on cooldown exactly as before this feature existed",
          controller.escalation_pending_type == InterventionType.REPLAN, controller.escalation_pending_type)
    check("9d. force_terminate_reason was never set by this unrelated path",
          controller.force_terminate_reason is None)


# ── 10: env_success always wins, even with a repeat present ─────────────
def test_env_success_wins_over_post_intervention_repeat():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REPLAN, escalated_from=None)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "You win.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
        won=True, done=True, reward=1,
    ))
    check("10. env_success (won=True) suppresses the post-intervention exact-repeat "
          "rule at the controller level -- no forced termination reason set",
          controller.force_terminate_reason is None and r["post_intervention_exact_repeat"] is None,
          (controller.force_terminate_reason, r["post_intervention_exact_repeat"]))


def test_env_success_wins_over_post_intervention_repeat_agent_level():
    """Integration-level companion to #10: even if the controller-level
    guard were somehow bypassed, agents.py's own `if done: return` already
    runs BEFORE the force_terminate_reason check and unconditionally
    returns success -- confirmed end-to-end through the real agent loop."""
    from agents import ALFWorldAgent
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=5)
    controller = StatefulController(cfg)
    llm = FakeLLM(["go to nowhere", "go to nowhere"])
    env = ScriptedFakeEnv([
        ("Nothing happens.", 0, False, False),
        ("You win.", 1, True, True),
    ], admissible=("go to sink 1",))
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="legacy_early_stop")
    _, success = agent.run(env, "task", "put", to_print=False)
    check("10b. done=True/won=True on the very step a repeat occurs is honored as "
          "success end-to-end, never overridden by the new rule",
          success is True, success)


# ── 11: budget exhaustion caps escalation, never exceeds max_interventions ──
def test_budget_exhausted_terminates_instead_of_a_4th_intervention():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=3, patch_duration_steps=5)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)  # #1
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)  # #2
    controller._arm_intervention(InterventionType.REFLECT, escalated_from=None)  # #3 -- budget now exhausted
    check("11a. budget is now exhausted (intervention_count == max_interventions)",
          controller.intervention_count == cfg.max_interventions, controller.intervention_count)
    r = controller.step(**_step_kwargs(
        "go to shelf 1", "Nothing happens.", admissible=("go to shelf 1",),
        recent_actions=["go to shelf 1"], recent_observations=["Nothing happens."],
    ))
    check("11. exact repeat while active REFLECT and budget exhausted -> terminates "
          "immediately (post_intervention_repeat_budget_exhausted), does NOT arm a 4th",
          controller.force_terminate_reason == "post_intervention_repeat_budget_exhausted",
          controller.force_terminate_reason)
    check("11b. intervention_count never exceeds max_interventions=3",
          controller.intervention_count <= cfg.max_interventions, controller.intervention_count)
    check("11c. transition explicitly logged as budget-exhausted",
          "budget_exhausted" in r["post_intervention_exact_repeat"]["transition"],
          r["post_intervention_exact_repeat"]["transition"])


# ── 12: idx=0's real desklamp pattern (REPLAN active, "use desklamp 1"
#     allowed once, repeated once, must terminate immediately -- must NOT
#     continue for 4-5 more identical steps the way the pre-fix real run
#     did) ──────────────────────────────────────────────────────────────
def test_idx0_desklamp_pattern_terminates_on_first_post_replan_repeat():
    """Real A40 idx=0 (look_at_obj_in_light-Bowl-None-DeskLamp-308) pre-fix
    trajectory: a REPLAN armed at step 11 ('go to drawer 3'); step 12 was
    the FIRST 'use desklamp 1' (allowed, no repeat yet -- action differs
    from step 11's); step 13 repeated 'use desklamp 1' exactly WHILE that
    REPLAN was still pending (patch_duration_steps=3, expiry=14, not yet
    reached) -- pre-fix, this was not caught at all and the episode kept
    repeating the identical action through steps 14-18 before finally
    ending. This test drives the controller directly through the
    equivalent two steps (REPLAN pre-armed, matching step 11's outcome;
    first then repeated 'use desklamp 1') using the real production
    ae_full.yaml config, and confirms termination fires immediately on the
    second occurrence -- exactly 2 controller.step() calls total, never a
    3rd (i.e. never reaching what would have been steps 14-18)."""
    cfg = load_config(AE_FULL_YAML)
    controller = StatefulController(cfg)
    controller._arm_intervention(InterventionType.REPLAN, escalated_from=None)

    r_step12 = controller.step(**_step_kwargs(
        "use desklamp 1", "Nothing happens.", admissible=("use desklamp 1",),
        recent_actions=["go to drawer 3"], recent_observations=["Nothing happens."],
    ))
    check("12a. the first 'use desklamp 1' (idx=0 step 12 equivalent) is allowed "
          "to execute -- no forced termination yet",
          controller.force_terminate_reason is None, controller.force_terminate_reason)

    r_step13 = controller.step(**_step_kwargs(
        "use desklamp 1", "Nothing happens.", admissible=("use desklamp 1",),
        recent_actions=["go to drawer 3", "use desklamp 1"],
        recent_observations=["Nothing happens.", "Nothing happens."],
    ))
    check("12b. the repeated 'use desklamp 1' (idx=0 step 13 equivalent) "
          "terminates the episode immediately via post_replan_exact_repeat",
          controller.force_terminate_reason == "post_replan_exact_repeat", controller.force_terminate_reason)
    check("12c. exactly 2 steps were ever processed -- the episode never "
          "continues to what would have been steps 14-18 (the pre-fix "
          "real run's repeated-forever tail)",
          len(controller.step_log) == 2, len(controller.step_log))


if __name__ == "__main__":
    test_active_verify_exact_repeat_escalates_to_reflect()
    test_active_reflect_exact_repeat_escalates_to_replan()
    test_active_replan_exact_repeat_terminates()
    test_replan_repeat_terminates_despite_full_recovery_grace()
    test_replan_repeat_terminates_despite_active_cooldown()
    test_different_shelf_numbers_do_not_trigger_post_intervention_rule()
    test_take_different_object_numbers_do_not_trigger_post_intervention_rule()
    test_no_active_intervention_legacy_behavior_unaffected()
    test_non_repeating_unresolved_still_escalates_via_old_patch_expiry_path()
    test_env_success_wins_over_post_intervention_repeat()
    test_env_success_wins_over_post_intervention_repeat_agent_level()
    test_budget_exhausted_terminates_instead_of_a_4th_intervention()
    test_idx0_desklamp_pattern_terminates_on_first_post_replan_repeat()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
