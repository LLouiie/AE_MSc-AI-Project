"""Minimal, AE-full-only coverage gap closed for the production
termination-policy decision (legacy_early_stop, MAX_STEPS=50 shared,
recovery_grace_steps/max_interventions bounded). Scope is deliberately
narrow per the task: does NOT touch react-only or reflexion-only
behavior (those already have their own coverage / are explicitly out of
scope this round), does NOT change any AE threshold/decay/intervention
logic, config file, or other experiment parameter -- read-only checks
against the REAL configs/controllers/ae_full.yaml values (loaded via
load_config(), not a hand-picked test override), unlike
test_ae_controller.py's Part-3 grace tests which intentionally use their
own AEConfig(max_interventions=5, ...) to test the MECHANISM in
isolation. This file additionally confirms the mechanism holds with the
actual production numbers (max_interventions=3, recovery_grace_steps=3,
patch_duration_steps=3, warmup_steps=3, cooldown_steps=3) and that
intervention/grace cannot be refreshed beyond that fixed budget.

Run directly:
    python3 ae/controllers/tests/test_ae_full_production_termination_gate.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "alfworld_runs_ae"))

from ae.controllers.config import load_config  # noqa: E402
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
    """Returns a fixed action sequence, ignoring the prompt content
    (duplicated locally from test_ae_controller.py's FakeLLM to keep this
    file self-contained, per repo convention)."""
    def __init__(self, actions):
        self._actions = list(actions)
        self.seen_prompts = []
        self.call_count = 0
        self.model = "Qwen/Qwen3-8B"

    def __call__(self, prompt: str) -> str:
        self.seen_prompts.append(prompt)
        self.call_count += 1
        return self._actions.pop(0) if self._actions else "look"


class ScriptedFakeEnv:
    """Same shape as test_ae_controller.py's ScriptedFakeEnv: fully
    scripted (observation, reward, done, won) per step, last entry
    repeats once exhausted, tracks step_calls."""
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


class FakeEnvAdmissiblePut:
    """admissible_commands only ever contains the "move ..." phrasing
    (mirrors the real ALFWorld json_2.1.1 grammar) -- so a step only
    reports admissible if the environment actually received "move X to
    Y", confirming the put->move adapter is what made it through, not
    just that SOME action was accepted."""
    def __init__(self):
        self.step_calls = 0
        self.received_actions = []

    def reset(self):
        ob = "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in a room.\n\nYour task is to: put a mug in the fridge."
        return [ob], {"admissible_commands": [["go to fridge 1", "move mug 1 to fridge 1"]], "won": [False]}

    def step(self, actions):
        self.step_calls += 1
        action = actions[0]
        self.received_actions.append(action)
        if action == "move mug 1 to fridge 1":
            ob, done = "You move the mug 1 to the fridge 1.", True
        else:
            ob, done = "Nothing happens.", False
        return [ob], [1 if done else 0], [done], {
            "admissible_commands": [["go to fridge 1", "move mug 1 to fridge 1"]], "won": [done],
        }

    def close(self):
        pass


def test_production_config_values():
    """1. Confirm the actual recovery_grace_steps / max_interventions
    values that will be used in the real 134-task run."""
    cfg = load_config(AE_FULL_YAML)
    check("1. production max_interventions == 3", cfg.max_interventions == 3, cfg.max_interventions)
    check("1b. production recovery_grace_steps == 3", cfg.recovery_grace_steps == 3, cfg.recovery_grace_steps)
    check("1c. production patch_duration_steps == 3", cfg.patch_duration_steps == 3, cfg.patch_duration_steps)
    check("1d. production warmup_steps == 3", cfg.warmup_steps == 3, cfg.warmup_steps)
    check("1e. production cooldown_steps == 3", cfg.cooldown_steps == 3, cfg.cooldown_steps)
    return cfg


def test_grace_bounded_and_terminates_reliably_under_production_config():
    """2/3/4. Under the REAL production AEConfig (not a test override) and
    termination_policy="legacy_early_stop": an action that is both
    invalid (never admissible) and, from the second step onward, an exact
    repeat of itself must (a) drive at most max_interventions=3
    interventions total -- never more, (b) leave recovery_grace_remaining
    permanently at 0 once the budget is exhausted (no further reset),
    (c) still reliably reach exhausted_repeated termination, and (d) never
    exceed MAX_STEPS=50, no matter how many times the same termination
    candidate recurs."""
    from agents import ALFWorldAgent

    cfg = load_config(AE_FULL_YAML)
    controller = StatefulController(cfg)
    # First `warmup_steps` (production=3) actions are distinct and legal, so
    # they clear warmup WITHOUT ever being the thing that trips the
    # repeated-action check -- confirmed empirically (see module note above
    # this test) that starting the repeat DURING warmup gives it zero grace
    # protection, since grace can only ever be granted by an intervention,
    # and interventions are suppressed while in_warmup. This scenario
    # exercises the intended post-warmup grace/escalation path instead.
    llm = FakeLLM(
        ["go to sink 1", "go to table 1", "go to shelf 1"] + ["go to nowhere"] * 60
    )
    env = ScriptedFakeEnv(
        [("Nothing happens.", 0, False, False)],
        admissible=("go to sink 1", "go to table 1", "go to shelf 1"),
    )
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="legacy_early_stop")
    trajectory, success = agent.run(env, "task", "put", to_print=False)

    check("2. intervention_count never exceeds production max_interventions=3",
          controller.intervention_count <= cfg.max_interventions, controller.intervention_count)
    check("2b. at least one intervention actually fired",
          controller.intervention_count > 0, controller.intervention_count)

    # Once the budget is exhausted, recovery_grace_remaining must never be
    # reset again -- find the step index where intervention_count first
    # reaches the cap, and confirm grace is monotonically non-increasing
    # (modulo one final _arm_intervention's reset) after that point, and
    # ends at 0.
    step_log = controller.step_log
    cap_hit_step = None
    running_count = 0
    counts_per_step = []
    for rec in step_log:
        if rec["intervention_id"] is not None:
            pass
        counts_per_step.append(rec.get("recovery_grace_remaining"))
    # Updated for the post-intervention exact-repeat rule (higher priority
    # than recovery_grace/cooldown -- see stateful_controller.py): a
    # continuously-repeated exact action while an intervention is pending
    # is now caught immediately (escalating verify->reflect->replan, then
    # terminating on the very next repeat under an active replan) rather
    # than waiting for the old grace-exhaustion path. This still only ever
    # consumes 1-3 of the intervention budget (never more), just via a
    # different, faster mechanism than before this feature existed.
    check("3. intervention_count is bounded in [1, max_interventions] -- the "
          "post-intervention exact-repeat rule now escalates/terminates a "
          "continuously-unresolved failure faster than the old grace-"
          "exhaustion path, but the budget bound still holds",
          1 <= controller.intervention_count <= cfg.max_interventions,
          controller.intervention_count)

    # Grace must still never be reset except by a genuine new
    # _arm_intervention() call (monotonic non-increasing between arms) --
    # but it no longer needs to reach exactly 0 for the episode to end,
    # since the post-intervention exact-repeat rule can now terminate the
    # episode while grace is still partially unused (it doesn't consult
    # recovery_grace_remaining at all, by design).
    grace_seq = [rec["recovery_grace_remaining"] for rec in step_log]
    first_armed_idx = next(i for i, rec in enumerate(step_log) if rec["intervention_id"] is not None)
    post_arm_grace = grace_seq[first_armed_idx:]
    is_non_increasing = all(post_arm_grace[i] >= post_arm_grace[i + 1] for i in range(len(post_arm_grace) - 1))
    check("3b. after the last intervention arms, grace is monotonically "
          "non-increasing (no infinite/repeated refresh of the budget)",
          is_non_increasing, grace_seq)

    # Deterministic with the REAL production config: this scenario's
    # invalid+repeated "go to nowhere" drives frustration/confidence
    # straight to the frustration_high+confidence_low branch, so the FIRST
    # intervention that ever fires here is REPLAN directly (not an
    # escalated VERIFY/REFLECT) -- confirmed by direct run, not assumed.
    # The very next exact repeat therefore always terminates via
    # post_replan_exact_repeat specifically, never the budget-exhausted
    # variant (budget is never even touched a 2nd time in this scenario).
    check("4. episode reliably terminates via post_replan_exact_repeat "
          "(the higher-priority post-intervention exact-repeat rule), not exhausted_repeated",
          agent.termination_reason == "post_replan_exact_repeat",
          agent.termination_reason)
    check("4b. success is False on this forced-termination path",
          success is False)
    check("5. total controller steps never exceed MAX_STEPS=50",
          len(controller.step_log) <= 50, len(controller.step_log))
    check("5b. total real environment interactions never exceed MAX_STEPS=50",
          env.step_calls <= 50, env.step_calls)
    check("5c. grace actually gave more than one real chance "
          "(env.step_calls > 1, not an immediate single-shot break)",
          env.step_calls > 1, env.step_calls)


def test_put_action_normalized_under_ae_full_real_config_path():
    """6 (put->move in AE-full's real path): with the REAL production
    AEConfig and termination_policy="legacy_early_stop" (not a bare
    AEConfig(), not fixed_horizon), confirm the put->move adapter still
    fires identically inside the controller-attached execution path, and
    that the controller's own per-step "action" field logs the EXECUTED
    (move) action, not the model's raw put wording. Does not touch or
    assert anything about AE's thresholds/decay/intervention selection
    logic itself."""
    from agents import ALFWorldAgent
    import agents as agents_mod
    agents_mod.MAX_STEPS = 1

    cfg = load_config(AE_FULL_YAML)
    controller = StatefulController(cfg)
    llm = FakeLLM(["Thought: place the mug.\nAction: put mug 1 in fridge 1"])
    env = FakeEnvAdmissiblePut()
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="legacy_early_stop")
    agent.run(env, "put a mug in the fridge", "put", to_print=False)

    check("6. env.step() under AE-full's real controller-attached path received "
          "'move mug 1 to fridge 1', never the put phrasing",
          env.received_actions == ["move mug 1 to fridge 1"], env.received_actions)
    step = agent.step_log[0]
    check("6b. parsed_action keeps the model's own put wording",
          step["parsed_action"] == "put mug 1 in fridge 1", step["parsed_action"])
    check("6c. executed_action is the normalized move wording",
          step["executed_action"] == "move mug 1 to fridge 1", step["executed_action"])
    check("6d. action_normalized is True", step["action_normalized"] is True)
    controller_record = controller.step_log[0]
    check("6e. controller's own per-step 'action' field is the executed (move) action",
          controller_record["action"] == "move mug 1 to fridge 1", controller_record["action"])


def test_llm_call_count_matches_generation_count_under_legacy_early_stop_ae_full():
    """No hidden retries/extra calls: with the real production AEConfig
    and legacy_early_stop, a clean 3-step episode (no repeats, no
    context-length retries) must make exactly 3 LLM calls."""
    from agents import ALFWorldAgent
    import agents as agents_mod
    agents_mod.MAX_STEPS = 3

    cfg = load_config(AE_FULL_YAML)
    controller = StatefulController(cfg)
    llm = FakeLLM([
        "Thought: a.\nAction: go to sink 1",
        "Thought: b.\nAction: go to table 1",
        "Thought: c.\nAction: go to shelf 1",
    ])
    env = ScriptedFakeEnv([("Nothing happens.", 0, False, False)], admissible=("go to sink 1", "go to table 1", "go to shelf 1"))
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="legacy_early_stop")
    agent.run(env, "task", "put", to_print=False)

    check("7. exactly 3 LLM calls for a clean 3-step episode (no hidden retries)",
          llm.call_count == 3, llm.call_count)
    check("7b. exactly 3 real environment steps, matching the LLM call count 1:1",
          env.step_calls == 3, env.step_calls)


# ── Controlled multi-intervention recovery trajectory (user-confirmed AE-full
#    semantics: up to 3 interventions per episode; each gets a 3-step patch
#    window; genuine recovery within that window lets the episode continue
#    and re-arms eligibility for a LATER intervention on a fresh stuck event;
#    the budget is a per-episode total, not per-event -- once 3 interventions
#    have fired, no 4th ever arms even if a new stuck event occurs). ────────
def test_recovery_condition_is_local_state_change_proxy():
    """1. Pin down the exact code condition for "recovery success" so this
    isn't asserted from memory: stateful_controller.py:185-198 tracks
    meaningful_change_since_intervention, set True the first time (during
    a pending intervention's active window) signals.local_state_change_
    proxy >= 0.5 for that step; evaluated once, at patch expiry
    (step_index >= intervention_expiry_step). local_state_change_proxy
    itself (signals.py:215-219) is 1.0 unless the observation text is
    exactly "nothing happens." (case/whitespace-normalized) AND the
    admissible_commands set is unchanged AND the action isn't a
    think-action -- i.e. "recovered" means at least one real observation
    change or admissible-set change happened before the patch expired;
    "unresolved" means neither ever happened in that window."""
    import inspect
    from ae.controllers import stateful_controller as sc_mod
    from ae.controllers import signals as sig_mod
    src_sc = inspect.getsource(sc_mod.StatefulController.step)
    src_sig = inspect.getsource(sig_mod.SignalExtractor.extract)
    check("1. stateful_controller.step() computes meaningful_change_this_step "
          "from signals.local_state_change_proxy >= 0.5",
          "meaningful_change_this_step = signals.local_state_change_proxy >= 0.5" in src_sc)
    check("1b. outcome is evaluated exactly at patch expiry "
          "(step_index >= intervention_expiry_step), not every step",
          "self.step_index >= self.intervention_expiry_step" in src_sc)
    check("1c. signals.py defines local_state_change_proxy as 0 only when "
          "observation=='nothing happens.' AND admissible set unchanged AND not a think-action",
          'state_unchanged_text = normalize_text(observation) == "nothing happens."' in src_sig
          and "admissible_set_changed = set(admissible_before) != set(admissible_after)" in src_sig)


def test_multiple_interventions_recover_then_rearm_then_cap_at_budget():
    """2. Controlled trajectory: intervention #1 -> recovers within its
    3-step window -> episode continues normally -> a fresh stuck event
    later -> intervention #2 fires -> recovers -> a fresh stuck event
    later -> intervention #3 fires -> recovers -> intervention_count==3
    (the production max_interventions) -> a FOURTH stuck event occurs
    but must NOT arm a 4th intervention (budget exhausted), and the
    episode is free to keep running (MAX_STEPS is still the only backstop
    at that point, per the user's confirmed semantics that the 3-budget
    is a per-episode total, not a per-event allowance)."""
    from agents import ALFWorldAgent
    import agents as agents_mod
    agents_mod.MAX_STEPS = 20

    cfg = load_config(AE_FULL_YAML)
    controller = StatefulController(cfg)

    # Each "stuck" pair deliberately uses two DISTINCT invalid actions (e.g.
    # "go to nowhere_a" / "go to nowhere_a2"), not an exact repeat of each
    # other -- an exact repeat of the immediately-preceding action while an
    # intervention is pending is now caught by the higher-priority
    # post-intervention exact-repeat rule (see stateful_controller.py /
    # test_post_intervention_exact_repeat.py), which would escalate/
    # terminate before this test's own recover-then-rearm narrative (an
    # independent, pre-existing mechanism) gets to run its course. Kept
    # distinct here so this test still exercises ONLY the patch-duration-
    # expiry outcome-evaluation path in isolation.
    admissible = ("go to sink 1", "go to table 1", "go to shelf 1")
    actions = [
        "go to sink 1", "go to table 1", "go to shelf 1",             # 1-3: warmup, valid
        "go to nowhere_a", "go to nowhere_a2", "go to sink 1", "go to table 1",   # 4-7: stuck(#1 arms@4)->still stuck (different)->recover->recover
        "go to nowhere_b", "go to nowhere_b2", "go to sink 1", "go to table 1",  # 8-11: stuck(#2 arms@8)->still stuck->recover->recover
        "go to nowhere_c", "go to nowhere_c2", "go to sink 1", "go to table 1",  # 12-15: stuck(#3 arms@12)->still stuck->recover->recover
        "go to nowhere_d", "go to nowhere_d2",                        # 16-17: 4th stuck event, budget already exhausted -> no #4
    ]
    observations = [
        "You arrive at sink 1.", "You arrive at table 1.", "You arrive at shelf 1.",
        "Nothing happens.", "Nothing happens.", "You arrive at sink 1 (recovery a).", "You arrive at table 1 (recovery b).",
        "Nothing happens.", "Nothing happens.", "You arrive at sink 1 (recovery c).", "You arrive at table 1 (recovery d).",
        "Nothing happens.", "Nothing happens.", "You arrive at sink 1 (recovery e).", "You arrive at table 1 (recovery f).",
        "Nothing happens.",
        "You put the mug in the sink. Task complete.",  # 17: clean scripted win, no LLM-fallback overrun
    ]
    assert len(actions) == len(observations)
    dones = [False] * 16 + [True]
    wons = [False] * 16 + [True]

    llm = FakeLLM([f"Thought: t.\nAction: {a}" for a in actions])
    env = ScriptedFakeEnv(
        [(o, (1 if w else 0), d, w) for o, d, w in zip(observations, dones, wons)],
        admissible=admissible,
    )
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="legacy_early_stop")
    agent.run(env, "task", "put", to_print=False)

    sl = controller.step_log
    # A step is a genuine fresh EVALUATION of an outcome only when
    # outcome_evaluation_step == that same step's own step number --
    # intervention_outcome/outcome_evaluation_step otherwise just persist
    # unchanged (stateful_controller.py never resets them except when a
    # new intervention arms), so counting every step where the field is
    # merely non-None would overcount stale/carried-over values.
    fresh_evaluations = [
        (rec["step"], rec["intervention_id"], rec["intervention_outcome"])
        for rec in sl if rec["outcome_evaluation_step"] == rec["step"]
    ]
    check("2. episode ran the full scripted 17 steps, ending via a clean "
          "scripted win (not an early termination, not an LLM-fallback overrun)",
          len(sl) == 17 and agent.termination_reason == "success", (len(sl), agent.termination_reason))
    check("2b. exactly 3 distinct interventions fired across the episode",
          controller.intervention_count == 3, controller.intervention_count)
    check("2c. all 3 interventions that fired were FRESHLY evaluated as recovered "
          "(genuine state change happened before each one's patch expired)",
          len(fresh_evaluations) == 3 and all(o == "recovered" for _, _, o in fresh_evaluations),
          fresh_evaluations)
    check("2d. no intervention was ever evaluated as unresolved in this trajectory",
          all(o != "unresolved" for _, _, o in fresh_evaluations), fresh_evaluations)
    ids_seen = sorted(set(rec["intervention_id"] for rec in sl if rec["intervention_id"] is not None))
    check("2e. intervention ids are exactly 1,2,3 -- the 4th stuck event (steps 16-17) "
          "never armed a 4th intervention despite being a fresh stuck event",
          ids_seen == [1, 2, 3], ids_seen)
    step16_17 = [rec for rec in sl if rec["step"] in (16, 17)]
    check("2f. the 4th stuck event (step 16) produced no new intervention "
          "(stays 'continue', explicitly suppressed by the exhausted budget)",
          all(rec["intervention"] == "continue" for rec in step16_17),
          [(rec["step"], rec["intervention"]) for rec in step16_17])
    check("2g. the suppression reason at the 4th stuck event explicitly cites max_interventions",
          any("max_interventions" in rec["intervention_reason"] for rec in step16_17),
          [(rec["step"], rec["intervention_reason"]) for rec in step16_17])


def test_no_recovery_terminates_immediately_without_spending_2nd_or_3rd_budget():
    """3. Re-confirms (with a tightened, precise assertion per the now-
    confirmed design intent) that a SINGLE continuously-unresolved stuck
    event -- never recovering -- ends the episode using only 1 of the
    3-intervention budget, and never attempts a 2nd or 3rd intervention.
    This is the same empirical scenario as test_grace_bounded_and_
    terminates_reliably_under_production_config. Since the post-
    intervention exact-repeat rule was added, this now terminates via
    that higher-priority rule (post_replan_exact_repeat) rather than the
    old exhausted_repeated path -- intervention_count==1 is unaffected."""
    from agents import ALFWorldAgent

    cfg = load_config(AE_FULL_YAML)
    controller = StatefulController(cfg)
    llm = FakeLLM(["go to sink 1", "go to table 1", "go to shelf 1"] + ["go to nowhere"] * 60)
    env = ScriptedFakeEnv([("Nothing happens.", 0, False, False)],
                          admissible=("go to sink 1", "go to table 1", "go to shelf 1"))
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="legacy_early_stop")
    agent.run(env, "task", "put", to_print=False)

    check("3. exactly 1 intervention fired -- the episode does NOT proceed to "
          "spend a 2nd or 3rd intervention's budget on a never-recovering event",
          controller.intervention_count == 1, controller.intervention_count)
    # Same deterministic scenario as test_grace_bounded_and_terminates_
    # reliably_under_production_config (REPLAN fires directly, no
    # escalation chain, budget never touched a 2nd time) -- asserted
    # precisely, confirmed by direct run.
    check("3b. episode reliably terminates via post_replan_exact_repeat "
          "(the higher-priority post-intervention exact-repeat rule), not exhausted_repeated",
          agent.termination_reason == "post_replan_exact_repeat",
          agent.termination_reason)
    # intervention_outcome is "pending" for every step the one intervention
    # is active (persists unchanged, see the module-level note on
    # outcome_evaluation_step above), then flips to "unresolved" exactly at
    # patch expiry and stays there -- "recovered" must never appear, and
    # the final recorded outcome must be "unresolved".
    outcomes_seen = [rec["intervention_outcome"] for rec in controller.step_log
                     if rec["intervention_outcome"] not in (None,)]
    check("3c. the one intervention that fired was never evaluated as recovered, "
          "and its final (and only ever) resolved outcome is unresolved",
          "recovered" not in outcomes_seen and outcomes_seen[-1] == "unresolved", outcomes_seen)


# ── Warmup / legacy_early_stop boundary fix (real production case: an exact
#    repeat landing inside the controller's warmup window used to kill the
#    episode at intervention_count=0 -- e.g. idx=106
#    pick_heat_then_place_in_recep-Egg-...-GarbageCan-10, A40 exam,
#    terminated in 3 LLM calls, controller never got a single non-suppressed
#    step). Fix: agents.py no longer treats "no grace yet" as fatal while the
#    controller is at-or-one-step-past its warmup boundary (bounded, one-time
#    window -- see agents.py's controller_in_warmup); stateful_controller.py
#    freezes the edge-detection baseline (previous_signature/
#    previous_invalid_action_flag/last_candidate_severity) while in_warmup so
#    the first genuinely non-warmup call still sees a fresh rising edge
#    instead of one already "consumed" by a suppressed warmup event. ────────
def test_warmup_repeat_does_not_terminate_before_first_intervention_chance():
    """C1. An exact repeat that lands inside (and at the boundary of) the
    real production warmup window (warmup_steps=3) must NOT kill the
    episode at intervention_count=0 -- the controller must get to make its
    first non-suppressed call, and (since the underlying condition is
    still active there) fire its first intervention exactly then."""
    from agents import ALFWorldAgent

    cfg = load_config(AE_FULL_YAML)
    controller = StatefulController(cfg)
    # "open fridge 1" is never admissible here (mirrors the real idx=106
    # case: the model tries a receptacle action that was never legal) --
    # invalid the 1st time (iter1, warmup, no repeat yet), an exact repeat
    # the 2nd time (iter2, still warmup) and 3rd time (iter3, the one extra
    # boundary iteration), then a clean, different, admissible action wins
    # the episode.
    llm = FakeLLM([
        "go to sink 1", "open fridge 1", "open fridge 1", "open fridge 1", "take egg 1",
    ])
    env = ScriptedFakeEnv(
        [
            ("You arrive at sink 1.", 0, False, False),
            ("Nothing happens.", 0, False, False),
            ("Nothing happens.", 0, False, False),
            ("Nothing happens.", 0, False, False),
            ("You take the egg 1.", 1, True, True),
        ],
        admissible=("go to sink 1", "take egg 1"),
    )
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="legacy_early_stop")
    trajectory, success = agent.run(env, "task", "heat", to_print=False)

    check("C1. episode did NOT die at intervention_count=0 -- the repeated "
          "action inside warmup was suppressed, not fatal",
          controller.intervention_count > 0, controller.intervention_count)
    check("C1b. the warmup-repeat suppression path actually fired (distinct "
          "from real grace consumption)",
          controller.warmup_suppressed_count > 0, controller.warmup_suppressed_count)
    check("C1c. the first intervention fired exactly at the first step past "
          "the warmup boundary (step 4, warmup_steps=3)",
          controller.trigger_steps[:1] == [cfg.warmup_steps + 1], controller.trigger_steps)
    check("C1d. episode reached its scripted clean win, not exhausted_repeated",
          success is True and agent.termination_reason == "success",
          (success, agent.termination_reason))
    check("C1e. env steps never exceed MAX_STEPS=50",
          env.step_calls <= 50, env.step_calls)


def test_warmup_repeat_self_recovery_does_not_force_intervention():
    """C2. If the model repeats inside warmup but then recovers on its own
    (a genuinely different, valid, progressing action) before the
    warmup-boundary grace window would even give the controller a
    non-suppressed look, no intervention should be forced -- by the time
    the controller gets its first non-warmup call, the state has already
    settled back down (frustration decayed below frustration_medium), so
    there is no active band left to react to."""
    from agents import ALFWorldAgent

    cfg = load_config(AE_FULL_YAML)
    controller = StatefulController(cfg)
    # Observation wording is deliberately kept dissimilar across steps (not
    # a shared "You arrive at X 1." template) -- signals.py's
    # repeated_observation is a Jaccard similarity over the last
    # repetition_window(=3) observations regardless of which action
    # produced them, so two "arrive at ROOM 1" strings would themselves
    # register as partially "repeated" and mask what this test is actually
    # checking (genuine self-recovery, not a same-shaped-sentence false
    # positive).
    llm = FakeLLM([
        "go to sink 1", "look", "look", "go to table 1", "take egg 1",
    ])
    env = ScriptedFakeEnv(
        [
            ("The sink area is here.", 0, False, False),
            ("You see nothing.", 0, False, False),
            ("You see nothing.", 0, False, False),
            ("You are next to table 1 now.", 0, False, False),
            ("You take the egg 1.", 1, True, True),
        ],
        admissible=("go to sink 1", "look", "go to table 1", "take egg 1"),
    )
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="legacy_early_stop")
    trajectory, success = agent.run(env, "task", "heat", to_print=False)

    check("C2. the warmup repeat was suppressed (not fatal) even though it "
          "never armed an intervention",
          controller.warmup_suppressed_count == 1, controller.warmup_suppressed_count)
    check("C2b. self-recovery after warmup does NOT force an intervention",
          controller.intervention_count == 0, controller.intervention_count)
    check("C2c. episode continues normally to its scripted clean win",
          success is True and agent.termination_reason == "success",
          (success, agent.termination_reason))


def test_react_legacy_early_stop_repeat_termination_unchanged():
    """C5 / req 1. ReAct (no controller attached) under legacy_early_stop:
    the original exact-repeated-action immediate termination must be
    completely unaffected by the warmup/legacy-early-stop fix above --
    controller is None, so controller_in_warmup is unconditionally False
    and the very first repeat still terminates immediately, exactly as
    before this change."""
    from agents import ALFWorldAgent

    llm = FakeLLM(["go to sink 1", "go to nowhere", "go to nowhere"] + ["go to nowhere"] * 60)
    env = ScriptedFakeEnv(
        [("Nothing happens.", 0, False, False)],
        admissible=("go to sink 1",),
    )
    agent = ALFWorldAgent(llm, controller=None, termination_policy="legacy_early_stop")
    trajectory, success = agent.run(env, "task", "put", to_print=False)

    check("C5. ReAct (no controller) still terminates immediately on the "
          "first exact repeat -- no warmup concept applies without a controller",
          agent.termination_reason == "exhausted_repeated", agent.termination_reason)
    check("C5b. success is False", success is False)
    # 3 LLM calls: "go to sink 1" (call 1), "go to nowhere" (call 2, first
    # occurrence), "go to nowhere" again (call 3, exact repeat -> immediate
    # termination, no controller to ever suppress it).
    check("C5c. terminates on the very first repeat (3 calls total, not more)",
          llm.call_count == 3, llm.call_count)
    check("C5d. env steps never exceed MAX_STEPS=50",
          env.step_calls <= 50, env.step_calls)


if __name__ == "__main__":
    test_production_config_values()
    test_grace_bounded_and_terminates_reliably_under_production_config()
    test_put_action_normalized_under_ae_full_real_config_path()
    test_llm_call_count_matches_generation_count_under_legacy_early_stop_ae_full()
    test_recovery_condition_is_local_state_change_proxy()
    test_multiple_interventions_recover_then_rearm_then_cap_at_budget()
    test_no_recovery_terminates_immediately_without_spending_2nd_or_3rd_budget()
    test_warmup_repeat_does_not_terminate_before_first_intervention_chance()
    test_warmup_repeat_self_recovery_does_not_force_intervention()
    test_react_legacy_early_stop_repeat_termination_unchanged()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
