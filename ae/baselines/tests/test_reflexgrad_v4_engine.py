"""Tests for ae/baselines/reflexgrad_v4_engine.py (shared dual-process
state machine for reflexgrad_v4 / reflexion_only_reflexgrad_v4).

Plain assert-based, same convention as ae/baselines/tests/test_react_reflact_anchor.py.
Every LLM "role" is a scripted/mock callable -- no real LLM, no real
ALFWorld environment, per explicit instruction for this phase (A2.1).

Run directly:
    python3 ae/baselines/tests/test_reflexgrad_v4_engine.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root

from ae.baselines import reflexgrad_v4, reflexion_only_reflexgrad_v4  # noqa: E402
from ae.baselines.reflexgrad_v4_engine import ReflexGradV4Engine  # noqa: E402

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


# ── scripted role callables (no real LLM) ────────────────────────────────
class ScoreScript:
    """evaluator_fn stand-in: pops one score per call from a fixed list,
    then returns a high (non-stalling) default once the script runs out."""
    def __init__(self, scores):
        self._scores = list(scores)

    def __call__(self, task, obs, action, next_obs):
        return self._scores.pop(0) if self._scores else 8


class ActorSpy:
    """actor_fn stand-in that always returns a fixed action but records
    every (todo, policy, slow_plan) tuple it was called with, so tests can
    verify what the actor actually "saw" on any given call."""
    def __init__(self, action="go to fridge 1"):
        self._action = action
        self.seen_policies = []

    def __call__(self, task, obs, todo, policy, slow_plan, memory):
        self.seen_policies.append(policy)
        return self._action


def make_role_fns(scores, actor_fn=None):
    return dict(
        decomposer_fn=lambda task, obs: ["find the object", "use the object"],
        actor_fn=actor_fn or (lambda task, obs, todo, policy, slow_plan, memory: "go to fridge 1"),
        evaluator_fn=ScoreScript(scores),
        loss_fn=lambda task, policy, last3: "loss-text",
        gradient_fn=lambda loss, policy: "gradient-text",
        optimizer_fn=lambda policy, gradient: (policy or "") + "|opt",
        trajectory_analyzer_fn=lambda task, last5: "analysis-text",
        causal_diagnoser_fn=lambda analysis, policy: "broken-assumption-text",
        plan_generator_fn=lambda cause, policy: "corrective-plan-text",
    )


def run_steps(engine, n, task="put a mug in the fridge"):
    """Drives n steps through choose_action + after_env_step (env_success
    always False -- these tests are about routing, not termination). Every
    mock actor_fn here always returns a valid non-None action, so
    action_parser_fn's default identity function never produces a parse
    failure in this helper -- see test_parser_failure_* for that path."""
    routes = []
    for i in range(n):
        obs = f"obs{i}"
        parsed_action, _raw = engine.choose_action(task, obs)
        next_obs = f"obs{i + 1}"
        route = engine.after_env_step(task, obs, parsed_action, next_obs, env_success=False)
        routes.append(route)
    return routes


# ── 1. low-progress rule is strict (<4, not <=4) ─────────────────────────
def test_low_rule_is_strict():
    fns = make_role_fns([3, 3, 3, 3, 4])
    engine = reflexgrad_v4.build_engine(**fns, textgrad_enabled=False)
    routes = run_steps(engine, 5)
    check("1. window [3,3,3,3,4] (one boundary value =4) does NOT trigger slow",
          routes[-1] != "slow", routes)

    fns2 = make_role_fns([3, 3, 3, 3, 3])
    engine2 = reflexgrad_v4.build_engine(**fns2, textgrad_enabled=False)
    routes2 = run_steps(engine2, 5)
    check("2. window [3,3,3,3,3] (all strictly <4) DOES trigger slow",
          routes2[-1] == "slow", routes2)


# ── 2. slow triggers after exactly 5 consecutive low scores, calls all 3 roles once ──
def test_slow_triggers_and_calls_all_three_roles_once():
    fns = make_role_fns([2, 2, 2, 2, 2])
    engine = reflexgrad_v4.build_engine(**fns, textgrad_enabled=False)
    routes = run_steps(engine, 5)
    check("3. 5th step routes to slow", routes == ["base", "base", "base", "base", "slow"], routes)
    check("4. trajectory_analyzer called exactly once", engine.calls.trajectory_analyzer == 1)
    check("5. causal_diagnoser called exactly once", engine.calls.causal_diagnoser == 1)
    check("6. plan_generator called exactly once", engine.calls.plan_generator == 1)
    check("7. active_slow_plan is set to the plan_generator's output",
          engine.active_slow_plan == "corrective-plan-text")
    check("8. cooldown_remaining set to cooldown_steps (5)", engine.cooldown_remaining == 5)


# ── 3. slow preempts fast when both conditions are met on the same step ──
def test_slow_preempts_fast_on_same_step():
    # step6 is a multiple of 3 (fast-eligible) AND its trailing 5-window
    # (steps 2-6) are all <4 (slow-eligible). Step1's score (8) sits outside
    # that window so it doesn't interfere.
    fns = make_role_fns([8, 3, 3, 3, 3, 3])
    engine = reflexgrad_v4.build_engine(**fns)  # textgrad_enabled=True (default)
    routes = run_steps(engine, 6)
    check("9. step 3 (no stall yet) took the fast route", routes[2] == "fast", routes)
    fast_calls_after_step3 = (engine.calls.loss, engine.calls.gradient, engine.calls.optimizer)

    check("10. step 6 (both slow- and fast-eligible) routes to slow, not fast",
          routes[5] == "slow", routes)
    fast_calls_after_step6 = (engine.calls.loss, engine.calls.gradient, engine.calls.optimizer)
    check("11. loss/gradient/optimizer call counts unchanged by step 6 (fast did not also fire)",
          fast_calls_after_step6 == fast_calls_after_step3,
          f"{fast_calls_after_step3} vs {fast_calls_after_step6}")
    check("12. slow's three roles each fired exactly once at step 6",
          (engine.calls.trajectory_analyzer, engine.calls.causal_diagnoser, engine.calls.plan_generator) == (1, 1, 1))


# ── 4. fast triggers every 3 steps when nothing stalls ───────────────────
def test_fast_triggers_every_3_steps_when_no_stall():
    fns = make_role_fns([8] * 9)  # never low, never stalls
    engine = reflexgrad_v4.build_engine(**fns)
    routes = run_steps(engine, 9)
    expected = ["base", "base", "fast"] * 3
    check("13. fast fires at steps 3, 6, 9 only", routes == expected, routes)
    check("14. optimizer called exactly 3 times (once per fast step)", engine.calls.optimizer == 3)
    check("15. loss/gradient call counts match optimizer (1:1:1 per role per fast step)",
          engine.calls.loss == 3 and engine.calls.gradient == 3)


# ── 5. cooldown lasts exactly 5 steps and suppresses BOTH slow and fast ──
def test_cooldown_lasts_exactly_5_steps_and_suppresses_fast():
    fns = make_role_fns([2, 2, 2, 2, 2] + [8] * 6)
    engine = reflexgrad_v4.build_engine(**fns)
    routes_1_5 = run_steps(engine, 5)
    check("16. step 5 is the slow trigger", routes_1_5[-1] == "slow", routes_1_5)
    optimizer_calls_before_cooldown = engine.calls.optimizer

    routes_6_11 = run_steps(engine, 6)
    check("17. steps 6-10 (5 steps) are all 'cooldown'", routes_6_11[:5] == ["cooldown"] * 5, routes_6_11)
    check("18. optimizer call count unchanged across the cooldown window (fast suppressed even at step 6/9)",
          engine.calls.optimizer == optimizer_calls_before_cooldown,
          f"before={optimizer_calls_before_cooldown} after={engine.calls.optimizer}")
    check("19. step 11 (cooldown has ended) is no longer forced into cooldown",
          routes_6_11[-1] != "cooldown", routes_6_11)


# ── 6. working memory keeps only the last 10 transitions ────────────────
def test_working_memory_keeps_last_10():
    fns = make_role_fns([])
    engine = reflexgrad_v4.build_engine(**fns)
    for i in range(11):
        engine.memory.append((f"o{i}", f"a{i}", f"o{i + 1}", 5))
    check("20. memory length caps at 10 after 11 appends", len(engine.memory) == 10)
    check("21. the oldest entry (o0) was evicted, o1 is now the first",
          engine.memory.last(10)[0][0] == "o1")
    check("22. the newest entry (o10) is present and last",
          engine.memory.last(10)[-1][0] == "o10")


# ── 7. episode reset clears every piece of state ─────────────────────────
def test_episode_reset_clears_all_state():
    verifier = VerifierScript([False] * 8)  # always False -- exercises calls.todo_verifier
                                             # without changing todo status (attempts stay
                                             # below the default todo_max_attempts=4... note
                                             # this deliberately lets it fail at attempt 4,
                                             # which is fine -- this test only cares that
                                             # reset() zeroes everything afterward.
    fns = make_role_fns([2, 2, 2, 2, 2] + [8] * 3)
    engine = reflexgrad_v4.build_engine(**fns, todo_verifier_fn=verifier)
    engine.decompose("task", "initial observation")
    run_steps(engine, 8)  # triggers slow, sets cooldown/policy/memory/todos/etc.
    check("23. sanity: state was actually populated before reset",
          len(engine.memory) > 0 and engine.cooldown_remaining > 0 and engine.todos
          and engine.calls.total() > 0 and engine.agent_steps > 0 and engine.env_actions > 0
          and engine.calls.todo_verifier > 0)

    engine.reset()
    check("24. memory empty after reset", len(engine.memory) == 0)
    check("25. todos empty after reset", engine.todos == [])
    check("26. base_policy empty after reset", engine.base_policy == "")
    check("27. active_slow_plan is None after reset", engine.active_slow_plan is None)
    check("28. cooldown_remaining is 0 after reset", engine.cooldown_remaining == 0)
    check("29. agent_steps and env_actions are 0 after reset",
          engine.agent_steps == 0 and engine.env_actions == 0)
    check("30. all role call counts (including todo_verifier) are 0 after reset",
          engine.calls.total() == 0 and engine.calls.todo_verifier == 0)
    check("31. route_log empty after reset", engine.route_log == [])


# ── 8. reflexion-only disables TextGrad but keeps slow process ──────────
def test_reflexion_only_disables_textgrad_keeps_slow():
    fns = make_role_fns([8] * 15)  # never stalls -- isolates the fast/base distinction
    engine = reflexion_only_reflexgrad_v4.build_engine(**fns)
    routes = run_steps(engine, 15)
    check("32. no step ever routes to 'fast' when textgrad_enabled=False",
          "fast" not in routes, routes)
    check("33. steps 3/6/9/12/15 (would-be fast steps) fall through to 'base'",
          all(routes[i - 1] == "base" for i in (3, 6, 9, 12, 15)), routes)
    check("34. loss/gradient/optimizer are called zero times across the whole episode",
          (engine.calls.loss, engine.calls.gradient, engine.calls.optimizer) == (0, 0, 0))

    fns2 = make_role_fns([2, 2, 2, 2, 2])
    engine2 = reflexion_only_reflexgrad_v4.build_engine(**fns2)
    routes2 = run_steps(engine2, 5)
    check("35. slow process still fires in reflexion-only mode", routes2[-1] == "slow", routes2)
    check("36. slow's three roles still get called in reflexion-only mode",
          (engine2.calls.trajectory_analyzer, engine2.calls.causal_diagnoser,
           engine2.calls.plan_generator) == (1, 1, 1))


# ── 9. reflexgrad_v4 and reflexion_only_reflexgrad_v4 share ONE engine class ──
def test_both_wrappers_share_the_same_engine_class_not_duplicated():
    fns_a = make_role_fns([])
    fns_b = make_role_fns([])
    engine_a = reflexgrad_v4.build_engine(**fns_a)
    engine_b = reflexion_only_reflexgrad_v4.build_engine(**fns_b)
    check("37. both wrappers construct the exact same ReflexGradV4Engine class",
          type(engine_a) is ReflexGradV4Engine and type(engine_b) is ReflexGradV4Engine
          and type(engine_a) is type(engine_b))

    cfg_a = reflexgrad_v4.build_config()
    cfg_b = reflexion_only_reflexgrad_v4.build_config()
    diffs = {
        f: (getattr(cfg_a, f), getattr(cfg_b, f))
        for f in cfg_a.__dataclass_fields__
        if getattr(cfg_a, f) != getattr(cfg_b, f)
    }
    check("38. the two configs differ ONLY in textgrad_enabled",
          set(diffs.keys()) == {"textgrad_enabled"}, diffs)
    check("39. reflexgrad_v4's textgrad_enabled is True", cfg_a.textgrad_enabled is True)
    check("40. reflexion_only's textgrad_enabled is False", cfg_b.textgrad_enabled is False)


# ── 10. role call-count bookkeeping sanity over a short scripted episode ──
def test_role_call_counts_are_exact_over_a_short_episode():
    fns = make_role_fns([8, 8])  # 2 steps, no stall, no fast (2 < gradient_cadence_k)
    engine = reflexgrad_v4.build_engine(**fns)
    engine.decompose("task", "initial observation")
    run_steps(engine, 2)
    counts = engine.calls.as_dict()
    check("41. decomposer called exactly once", counts["decomposer"] == 1)
    check("42. actor called exactly twice (once per step)", counts["actor"] == 2)
    check("43. evaluator called exactly twice (once per step)", counts["evaluator"] == 2)
    check("44. no slow-role or fast-role calls in these 2 quiet steps",
          counts["trajectory_analyzer"] == 0 and counts["causal_diagnoser"] == 0
          and counts["plan_generator"] == 0 and counts["loss"] == 0
          and counts["gradient"] == 0 and counts["optimizer"] == 0)


# ══════════════════════════ Phase A2.1 additions ══════════════════════════

# ── 11. decomposer gated by use_task_decomposer, never touches step budget ──
def test_decomposer_flag_and_no_step_budget_impact():
    fns = make_role_fns([])
    engine = reflexgrad_v4.build_engine(**fns, use_task_decomposer=True)
    engine.decompose("task", "initial observation")
    check("45. decomposer called once when use_task_decomposer=True", engine.calls.decomposer == 1)
    check("46. decompose() does not touch agent_steps", engine.agent_steps == 0)
    check("47. decompose() does not touch env_actions", engine.env_actions == 0)
    check("48. todos populated from the decomposer's output", len(engine.todos) == 2)

    fns2 = make_role_fns([])
    engine2 = reflexgrad_v4.build_engine(**fns2, use_task_decomposer=False)
    engine2.decompose("task", "initial observation")
    check("49. decomposer NOT called when use_task_decomposer=False", engine2.calls.decomposer == 0)
    check("50. todos stay empty when the decomposer is disabled", engine2.todos == [])


# ── scripted todo_verifier_fn helper (A2.2) ──────────────────────────────
class VerifierScript:
    """todo_verifier_fn stand-in: pops one bool per call from a fixed list,
    then defaults to False once exhausted."""
    def __init__(self, results):
        self._results = list(results)
        self.calls_seen = []  # records (current_todo_content, obs, action, next_obs)

    def __call__(self, current_todo, prev_obs, action, next_obs):
        self.calls_seen.append((current_todo, prev_obs, action, next_obs))
        return self._results.pop(0) if self._results else False


# ── 12. TODO completes via the verifier even when env_success is False ──
# (A2.2 fix: env_success is the WHOLE-episode signal only -- it must never
# be what marks a mid-episode TODO done. See module docstring.)
def test_todo_advances_via_verifier_even_when_env_success_is_false():
    verifier = VerifierScript([True])  # confirms completion on the first real attempt
    fns = make_role_fns([8, 8])
    engine = reflexgrad_v4.build_engine(**fns, todo_verifier_fn=verifier)
    engine.decompose("task", "initial observation")
    check("51. first todo starts active", engine.todos[0].status == "active")
    check("52. second todo starts pending", engine.todos[1].status == "pending")

    parsed_action, _ = engine.choose_action("task", "obs0")
    route = engine.after_env_step("task", "obs0", parsed_action, "obs1", env_success=False)
    check("53. verifier=True marks the active todo done EVEN THOUGH env_success=False",
          engine.todos[0].status == "done")
    check("54. the next pending todo becomes active", engine.todos[1].status == "active")
    check("55. attempts/last_action were recorded on the completed todo",
          engine.todos[0].attempts == 1 and engine.todos[0].last_action == parsed_action)
    check("56. the route this step is still a normal router route, not forced by the TODO event",
          route in ("base", "fast", "cooldown", "slow"), route)
    check("57. the verifier was called exactly once, with the right arguments",
          len(verifier.calls_seen) == 1
          and verifier.calls_seen[0] == ("find the object", "obs0", parsed_action, "obs1"))


# ── 13. slow does NOT auto-fail the active todo -- it only records a reason ──
def test_slow_does_not_auto_fail_todo_but_records_reason():
    fns = make_role_fns([2, 2, 2, 2, 2])
    engine = reflexgrad_v4.build_engine(**fns)  # no verifier configured at all
    engine.decompose("task", "initial observation")
    routes = run_steps(engine, 5)
    check("58. sanity: slow actually triggered", routes[-1] == "slow", routes)
    check("59. the todo that was active during the stall is STILL active, not failed",
          engine.todos[0].status == "active", engine.todos[0].status)
    check("60. the causal_diagnoser's cause was recorded as a failure reason anyway",
          engine.todos[0].failure_reasons == ["broken-assumption-text"])
    check("61. the second todo was NOT auto-advanced to active (no auto-fail happened)",
          engine.todos[1].status == "pending", engine.todos[1].status)


# ── 14. a todo is only marked failed after todo_max_attempts unverified real attempts ──
def test_todo_fails_only_after_max_attempts_reached():
    # 3 failed verifications, one per real step -- must NOT fail yet (default
    # todo_max_attempts=4).
    verifier_3 = VerifierScript([False, False, False])
    fns_3 = make_role_fns([8, 8, 8])
    engine_3 = reflexgrad_v4.build_engine(**fns_3, todo_verifier_fn=verifier_3)
    engine_3.decompose("task", "initial observation")
    run_steps(engine_3, 3)
    check("62. after 3 unverified attempts (below todo_max_attempts=4), todo is still active",
          engine_3.todos[0].status == "active", engine_3.todos[0].status)
    check("63. attempts counted all 3 tries", engine_3.todos[0].attempts == 3)

    # A 4th failed verification -- NOW it must fail and advance.
    verifier_4 = VerifierScript([False, False, False, False])
    fns_4 = make_role_fns([8, 8, 8, 8])
    engine_4 = reflexgrad_v4.build_engine(**fns_4, todo_verifier_fn=verifier_4)
    engine_4.decompose("task", "initial observation")
    run_steps(engine_4, 4)
    check("64. after exactly 4 unverified attempts, the todo is marked failed",
          engine_4.todos[0].status == "failed", engine_4.todos[0].status)
    check("65. the next pending todo becomes active after the 4th failure",
          engine_4.todos[1].status == "active")
    check("66. todo_max_attempts default is 4 (official-repo value, see config docstring)",
          engine_4.config.todo_max_attempts == 4)


# ── 15. todo_verifier call count is exact ────────────────────────────────
def test_todo_verifier_call_count_is_exact():
    verifier = VerifierScript([False, False, True])
    fns = make_role_fns([8, 8, 8])
    engine = reflexgrad_v4.build_engine(**fns, todo_verifier_fn=verifier)
    engine.decompose("task", "initial observation")
    run_steps(engine, 3)
    check("67. todo_verifier called exactly once per real env action (3 steps -> 3 calls)",
          engine.calls.todo_verifier == 3)
    check("68. RoleCallCounts.as_dict() surfaces todo_verifier as its own field",
          engine.calls.as_dict()["todo_verifier"] == 3)


# ── 16. global env_success=True terminates without ever calling the verifier ──
def test_global_env_success_terminates_without_verifier_call():
    verifier = VerifierScript([])  # would return False if ever called -- must NOT be called
    fns = make_role_fns([8])
    engine = reflexgrad_v4.build_engine(**fns, todo_verifier_fn=verifier)
    engine.decompose("task", "initial observation")
    parsed_action, _ = engine.choose_action("task", "obs0")
    route = engine.after_env_step("task", "obs0", parsed_action, "obs1", env_success=True)
    check("69. route is 'terminal' on global success", route == "terminal")
    check("70. the verifier was never called (global success needs no extra verification)",
          len(verifier.calls_seen) == 0)
    check("71. the active todo's status is untouched by env_success alone (still 'active')",
          engine.todos[0].status == "active", engine.todos[0].status)


# ── 17. rollback_todo reopens a done todo and demotes the current one ────
def test_rollback_todo():
    verifier = VerifierScript([True])
    fns = make_role_fns([8, 8])
    engine = reflexgrad_v4.build_engine(**fns, todo_verifier_fn=verifier)
    engine.decompose("task", "initial observation")
    parsed_action, _ = engine.choose_action("task", "obs0")
    engine.after_env_step("task", "obs0", parsed_action, "obs1", env_success=False)
    check("72. sanity: todo0 done, todo1 active before rollback",
          engine.todos[0].status == "done" and engine.todos[1].status == "active")

    reactivated = engine.rollback_todo()
    check("73. rollback_todo reactivates the most recently done todo",
          reactivated is engine.todos[0] and engine.todos[0].status == "active")
    check("74. rollback_todo demotes the previously-active todo back to pending",
          engine.todos[1].status == "pending")


# ── 15. cooldown ends and a subsequent fast update DOES change base_policy
#        and IS what the next actor call sees ─────────────────────────────
def test_fast_update_after_cooldown_changes_policy_actor_sees():
    spy = ActorSpy()
    scores = [2, 2, 2, 2, 2] + [8] * 7  # 5 low (slow @ env_action 5) + 7 high (cooldown 6-10, base 11, fast 12)
    fns = make_role_fns(scores, actor_fn=spy)
    # gradient_cadence_k=6 (not the default 3): with the default, step 3 would
    # legitimately fire fast BEFORE slow ever gets a chance at step 5 (memory
    # only has 3 entries then, no stall yet), contaminating base_policy before
    # the point this test cares about. 6 keeps the first fast-eligible step
    # (6) inside the cooldown window (suppressed) and the next (12) safely
    # after cooldown ends -- isolating exactly the sequence under test.
    engine = reflexgrad_v4.build_engine(**fns, gradient_cadence_k=6)

    routes = run_steps(engine, 12)
    check("75. sanity: slow fired at step 5, cooldown covered steps 6-10, fast fired at step 12",
          routes[4] == "slow" and routes[5:10] == ["cooldown"] * 5 and routes[10] == "base"
          and routes[11] == "fast", routes)

    policy_after_slow = None  # what the actor saw going INTO step 6 (right after slow merged the plan)
    # spy.seen_policies[i] is what the actor saw entering step (i+1); step 6 is index 5.
    policy_after_slow = spy.seen_policies[5]
    check("76. right after slow, base_policy == the merged slow plan (base was empty)",
          policy_after_slow == "corrective-plan-text", policy_after_slow)

    policy_after_cooldown_before_fast = spy.seen_policies[10]  # entering step 11
    check("77. policy unchanged through the whole cooldown window (fast never fired to change it)",
          policy_after_cooldown_before_fast == "corrective-plan-text", policy_after_cooldown_before_fast)

    check("78. base_policy actually changed after the post-cooldown fast trigger (step 12)",
          engine.base_policy != policy_after_cooldown_before_fast, engine.base_policy)
    check("79. the new base_policy is exactly the optimizer's full-replacement output",
          engine.base_policy == "corrective-plan-text|opt", engine.base_policy)

    # Run one more step (13) and confirm the ACTOR was actually handed this
    # updated policy -- not just that the field changed internally.
    run_steps(engine, 1)
    check("80. the next actor call (step 13) was passed the updated post-fast policy",
          spy.seen_policies[-1] == "corrective-plan-text|opt", spy.seen_policies[-1])


# ── stored slow plan is a historical record, not a permanent override ──
def test_stored_slow_plan_is_not_a_permanent_override():
    scores = [2, 2, 2, 2, 2] + [8] * 7
    fns = make_role_fns(scores)
    engine = reflexgrad_v4.build_engine(**fns, gradient_cadence_k=6)  # see previous test's comment
    run_steps(engine, 12)  # same sequence: slow @5, fast fires again @12
    check("81. active_slow_plan still holds the original slow plan text (kept as a record)",
          engine.active_slow_plan == "corrective-plan-text")
    check("82. base_policy has moved past the slow plan alone -- it is NOT a permanent override",
          engine.base_policy != engine.active_slow_plan, engine.base_policy)


# ── parser failure only increments agent_steps/actor calls -- never TODO
#    attempts, the verifier, env_actions, evaluator, or memory ──────────
def test_parser_failure_increments_only_agent_steps_and_actor_calls():
    scripted_outputs = iter(["UNPARSEABLE GARBAGE", "go to fridge 1"])

    def scripted_actor(task, obs, todo, policy, slow_plan, memory):
        return next(scripted_outputs)

    def parser(raw):
        return None if raw == "UNPARSEABLE GARBAGE" else raw

    verifier = VerifierScript([True])
    fns = make_role_fns([8], actor_fn=scripted_actor)
    engine = reflexgrad_v4.build_engine(**fns, action_parser_fn=parser, todo_verifier_fn=verifier)
    engine.decompose("task", "initial observation")

    parsed_action, raw = engine.choose_action("task", "obs0")
    check("83. the parser correctly rejected the unparseable raw output", parsed_action is None)
    engine.after_parse_failure()
    check("84. agent_steps incremented by the failed attempt", engine.agent_steps == 1)
    check("85. actor call count incremented by the failed attempt", engine.calls.actor == 1)
    check("86. env_actions NOT incremented by a parse failure", engine.env_actions == 0)
    check("87. evaluator NOT called on a parse failure", engine.calls.evaluator == 0)
    check("88. memory NOT touched by a parse failure", len(engine.memory) == 0)
    check("89. todo_verifier NOT called on a parse failure", engine.calls.todo_verifier == 0)
    check("90. the active todo's attempts count is untouched by a parse failure",
          engine.todos[0].attempts == 0, engine.todos[0].attempts)

    parsed_action2, raw2 = engine.choose_action("task", "obs0")
    check("91. the second attempt parses into a real action", parsed_action2 == "go to fridge 1")
    engine.after_env_step("task", "obs0", parsed_action2, "obs1", env_success=False)
    check("92. agent_steps now counts both attempts", engine.agent_steps == 2)
    check("93. actor call count now counts both attempts", engine.calls.actor == 2)
    check("94. env_actions incremented only by the real action", engine.env_actions == 1)
    check("95. evaluator called exactly once (only for the real action)", engine.calls.evaluator == 1)
    check("96. memory has exactly one entry (only for the real action)", len(engine.memory) == 1)
    check("97. todo_verifier called exactly once (only for the real action)",
          engine.calls.todo_verifier == 1)
    check("98. the real action's attempt WAS counted on the todo", engine.todos[0].attempts == 1)


if __name__ == "__main__":
    test_low_rule_is_strict()
    test_slow_triggers_and_calls_all_three_roles_once()
    test_slow_preempts_fast_on_same_step()
    test_fast_triggers_every_3_steps_when_no_stall()
    test_cooldown_lasts_exactly_5_steps_and_suppresses_fast()
    test_working_memory_keeps_last_10()
    test_episode_reset_clears_all_state()
    test_reflexion_only_disables_textgrad_keeps_slow()
    test_both_wrappers_share_the_same_engine_class_not_duplicated()
    test_role_call_counts_are_exact_over_a_short_episode()
    test_decomposer_flag_and_no_step_budget_impact()
    test_todo_advances_via_verifier_even_when_env_success_is_false()
    test_slow_does_not_auto_fail_todo_but_records_reason()
    test_todo_fails_only_after_max_attempts_reached()
    test_todo_verifier_call_count_is_exact()
    test_global_env_success_terminates_without_verifier_call()
    test_rollback_todo()
    test_fast_update_after_cooldown_changes_policy_actor_sees()
    test_stored_slow_plan_is_not_a_permanent_override()
    test_parser_failure_increments_only_agent_steps_and_actor_calls()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
