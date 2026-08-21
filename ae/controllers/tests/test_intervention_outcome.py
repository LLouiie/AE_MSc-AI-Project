"""Tests for intervention outcome tracking (Part F of the revision spec):
pending_intervention bookkeeping, recovered/unresolved outcome evaluation
on patch expiry, severity escalation (verify->reflect->replan->replan) for
unresolved outcomes, and no cross-episode leakage.

Plain assert-based, same convention as the other controller test files.
Run directly:
    python3 ae/controllers/tests/test_intervention_outcome.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root

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


# A single step that never produces any local state change: invalid
# action, "Nothing happens.", admissible_commands never change.
_STUCK_STEP = dict(
    action="foo", observation="Nothing happens.",
    admissible_before=["go to bed 1"], admissible_after=["go to bed 1"],
    recent_actions=[], recent_observations=[], max_steps=50, is_think_action=False,
)

# A step that DOES produce a real local state change (admissible set
# changes), usable to simulate "the directive worked."
_RECOVERING_STEP = dict(
    action="open cabinet 1", observation="You open the cabinet 1. In it, you see a mug 1.",
    admissible_before=["open cabinet 1"], admissible_after=["close cabinet 1", "take mug 1 from cabinet 1"],
    recent_actions=[], recent_observations=[], max_steps=50, is_think_action=False,
)


def test_unresolved_reflect_escalates_to_replan_after_cooldown():
    """A reflect-tier intervention whose patch expires with no meaningful
    change must, after cooldown, escalate to replan -- even though the
    underlying condition (repeated invalid action) never changed."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=2, max_interventions=10, patch_duration_steps=2)
    controller = StatefulController(cfg)

    fired_types = []
    for _ in range(20):
        rec = controller.step(**_STUCK_STEP)
        if rec["intervention"] != "continue":
            fired_types.append(rec["intervention"])

    check("1. at least one escalation actually fired", len(fired_types) >= 2, fired_types)
    check("1b. some fired intervention has a higher severity than the "
          "first one (escalation happened, not just repeats of the same tier)",
          any(t != fired_types[0] for t in fired_types), fired_types)
    check("1c. unresolved_intervention_count was incremented",
          controller.unresolved_intervention_count > 0, controller.unresolved_intervention_count)
    # At least one record should show escalated_from set.
    escalated_records = [r for r in controller.step_log if r.get("escalated_from")]
    check("1d. at least one step log record shows escalated_from",
          len(escalated_records) > 0)


def test_recovered_outcome_allows_future_same_type_trigger():
    """If the directive's patch window sees a real local state change,
    the outcome is 'recovered', not 'unresolved' -- and a LATER recurrence
    of the same condition must be able to trigger again."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=10, patch_duration_steps=2)
    controller = StatefulController(cfg)

    # Step 1: stuck -> fires an intervention.
    controller.step(**_STUCK_STEP)
    fires_after_first = controller.intervention_count
    check("2. the first stuck step fired an intervention", fires_after_first >= 1)

    # Steps 2-3: patch window active, with a real state change occurring
    # inside it -> outcome should evaluate to "recovered".
    controller.step(**_RECOVERING_STEP)
    rec = controller.step(**_RECOVERING_STEP)
    check("2b. outcome evaluated to recovered", rec["intervention_outcome"] == "recovered",
          rec["intervention_outcome"])
    check("2c. unresolved_intervention_count stayed at 0",
          controller.unresolved_intervention_count == 0)

    # Recurrence: stuck again -> must be able to fire again (re-armed).
    fires_before_recurrence = controller.intervention_count
    rec2 = controller.step(**_STUCK_STEP)
    check("2d. a later recurrence of the same condition fires again after recovery",
          controller.intervention_count > fires_before_recurrence,
          (fires_before_recurrence, controller.intervention_count))


def test_no_retrigger_while_patch_is_still_active():
    """While a patch is still within its duration (not yet expired), no
    outcome has been evaluated yet and no new intervention should fire
    purely from the outcome-tracking mechanism (ordinary cooldown/is_event
    suppression still applies, tested elsewhere)."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=5, max_interventions=10, patch_duration_steps=5)
    controller = StatefulController(cfg)
    controller.step(**_STUCK_STEP)
    fires_after_first = controller.intervention_count
    for _ in range(3):  # still within patch_duration_steps=5 and cooldown_steps=5
        rec = controller.step(**_STUCK_STEP)
        check_name = f"3. still pending, no premature outcome evaluation (step {rec['step']})"
        check(check_name, rec["intervention_outcome"] == "pending", rec["intervention_outcome"])
    check("3b. no new intervention fired while the patch is still active",
          controller.intervention_count == fires_after_first)


def test_unresolved_replan_still_bounded_by_max_interventions():
    """Even once escalated all the way to replan and perpetually
    unresolved, the total intervention count must never exceed
    max_interventions."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=4, patch_duration_steps=1)
    controller = StatefulController(cfg)
    for _ in range(30):
        controller.step(**_STUCK_STEP)
    check("4. intervention_count never exceeds max_interventions",
          controller.intervention_count <= 4, controller.intervention_count)


def test_outcome_tracking_no_cross_episode_leakage():
    """reset() must clear every new Part-F field -- a fresh episode must
    not inherit a pending intervention, an outcome, an escalation, or the
    unresolved counter from the previous one."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=10, patch_duration_steps=1)
    controller = StatefulController(cfg)
    for _ in range(10):
        controller.step(**_STUCK_STEP)
    check("5. pre-reset sanity: unresolved count is nonzero",
          controller.unresolved_intervention_count > 0)
    check("5b. pre-reset sanity: an intervention id was assigned",
          controller.current_intervention_id is not None)

    controller.reset()
    check("5c. reset clears pending_intervention", controller.pending_intervention is False)
    check("5d. reset clears pending_intervention_type", controller.pending_intervention_type is None)
    check("5e. reset clears unresolved_intervention_count", controller.unresolved_intervention_count == 0)
    check("5f. reset clears escalation_pending_type", controller.escalation_pending_type is None)
    check("5g. reset clears current_intervention_id", controller.current_intervention_id is None)
    check("5h. reset clears current_intervention_outcome", controller.current_intervention_outcome is None)
    check("5i. reset clears current_escalated_from", controller.current_escalated_from is None)
    check("5j. reset clears steps_since_meaningful_change", controller.steps_since_meaningful_change == 0)


if __name__ == "__main__":
    test_unresolved_reflect_escalates_to_replan_after_cooldown()
    test_recovered_outcome_allows_future_same_type_trigger()
    test_no_retrigger_while_patch_is_still_active()
    test_unresolved_replan_still_bounded_by_max_interventions()
    test_outcome_tracking_no_cross_episode_leakage()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
