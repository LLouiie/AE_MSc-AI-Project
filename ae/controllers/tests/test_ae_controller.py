"""Lightweight assert-based tests for the AE controller (no pytest — this
repo's existing convention is plain script + assert, e.g.
hotpotqa_runs/test_inject.py). Run directly:

    python3 ae/controllers/tests/test_ae_controller.py

Covers the 9 cases from the task spec. No live LLM / vLLM / ALFWorld
environment needed — signal/state/controller tests are pure-function, and
the react-mode / wiring test uses a minimal in-file FakeEnv + FakeLLM.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "alfworld_runs_ae"))

from ae.core import InterventionType  # noqa: E402
from ae.controllers.affect_state import (  # noqa: E402
    AffectState, HysteresisTracker, Mode, clip01, initial_state, update_state,
)
from ae.controllers.config import AEConfig  # noqa: E402
from ae.controllers.signals import SignalExtractor, StepSignals  # noqa: E402
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


def make_signals(**kwargs) -> StepSignals:
    base = dict(invalid_action=0.0, repeated_action=0.0, repeated_observation=0.0,
                observation_novelty=0.0, unexpected_outcome=0.0, progress_signal=0.0,
                budget_ratio=0.0)
    base.update(kwargs)
    return StepSignals(**base)


# ── 1. state values always in [0, 1] ─────────────────────────────────────
def test_bounds():
    cfg = AEConfig()
    state = initial_state(cfg)
    extreme_signals = [
        make_signals(invalid_action=1.0, repeated_action=1.0, repeated_observation=1.0,
                     unexpected_outcome=1.0, budget_ratio=1.0),
        make_signals(progress_signal=1.0, observation_novelty=1.0),
    ]
    in_bounds = True
    for _ in range(200):
        for sig in extreme_signals:
            state = update_state(state, sig, cfg)
            for v in (state.uncertainty, state.frustration, state.surprise, state.confidence):
                if not (0.0 <= v <= 1.0):
                    in_bounds = False
    check("1. state values stay in [0,1] under 400 extreme updates", in_bounds)
    check("1b. clip01 clamps out-of-range inputs",
          clip01(-5.0) == 0.0 and clip01(5.0) == 1.0 and clip01(0.4) == 0.4)


# ── 2. repeated failure accumulates frustration ──────────────────────────
def test_frustration_accumulates():
    cfg = AEConfig()
    state = initial_state(cfg)
    fail_signal = make_signals(invalid_action=1.0, repeated_action=1.0)
    history = [state.frustration]
    for _ in range(6):
        state = update_state(state, fail_signal, cfg)
        history.append(state.frustration)
    check("2. frustration rises over repeated invalid/repeated-action steps",
          history[-1] > history[0] and history[-1] > 0.5,
          f"history={[round(h,3) for h in history]}")


# ── 3. progress lowers frustration, raises confidence ────────────────────
def test_progress_relieves():
    cfg = AEConfig()
    state = initial_state(cfg)
    fail_signal = make_signals(invalid_action=1.0, repeated_action=1.0)
    for _ in range(6):
        state = update_state(state, fail_signal, cfg)
    frustration_peak, confidence_after_fail = state.frustration, state.confidence

    progress_signal = make_signals(progress_signal=1.0, observation_novelty=1.0)
    for _ in range(6):
        state = update_state(state, progress_signal, cfg)
    check("3a. progress lowers frustration", state.frustration < frustration_peak,
          f"{state.frustration:.3f} vs peak {frustration_peak:.3f}")
    check("3b. progress raises confidence", state.confidence > confidence_after_fail,
          f"{state.confidence:.3f} vs {confidence_after_fail:.3f}")


# ── 4. surprise decays faster than frustration ───────────────────────────
def test_surprise_decays_faster():
    cfg = AEConfig()
    # Same starting value, then only decay (all-zero signals) — whichever
    # reaches near-zero in fewer steps decays faster. decay_surprise (0.35)
    # < decay_frustration (0.85) by config, so surprise should win.
    state = AffectState(uncertainty=0.5, frustration=0.9, surprise=0.9, confidence=0.5)
    zero_signal = make_signals()
    steps_for_surprise = steps_for_frustration = None
    for i in range(1, 50):
        state = update_state(state, zero_signal, cfg)
        if steps_for_surprise is None and state.surprise < 0.05:
            steps_for_surprise = i
        if steps_for_frustration is None and state.frustration < 0.05:
            steps_for_frustration = i
        if steps_for_surprise and steps_for_frustration:
            break
    check("4. surprise decays to ~0 faster than frustration",
          steps_for_surprise < steps_for_frustration,
          f"surprise took {steps_for_surprise} steps, frustration took {steps_for_frustration}")


# ── 5. hysteresis does not flicker near threshold ────────────────────────
def test_hysteresis_no_flicker():
    cfg = AEConfig()
    tracker = HysteresisTracker()
    band = cfg.frustration_hysteresis  # enter=0.70, exit=0.40
    # Oscillate frustration between just-above-enter and just-below-enter
    # (but above exit) — a naive single-threshold check would flip on
    # every step; hysteresis should stay latched "active" throughout.
    values = [0.75, 0.68, 0.72, 0.66, 0.71, 0.69, 0.73]
    flips = 0
    was_active = None
    for v in values:
        state = AffectState(uncertainty=0.0, frustration=v, surprise=0.0, confidence=1.0)
        tracker.update(state, cfg)
        if was_active is not None and tracker.frustration_active != was_active:
            flips += 1
        was_active = tracker.frustration_active
    check("5. no flicker while oscillating between enter and exit bands",
          flips == 0 and was_active is True, f"flips={flips}, final_active={was_active}")

    # Now actually drop below exit -> should deactivate exactly once.
    state = AffectState(uncertainty=0.0, frustration=0.35, surprise=0.0, confidence=1.0)
    tracker.update(state, cfg)
    check("5b. drops out of high band once below exit threshold", tracker.frustration_active is False)


# ── 6. cooldown blocks dense (re-)triggers ───────────────────────────────
# Revision note: event firing is no longer gated on "the single dominant
# Mode changed" (that mechanism is gone — Mode/current_mode() is logging
# only now), so the old test's "force previous_mode=NORMAL" trick is dead
# code against the new implementation. Rewritten using an explicit
# invalid_action rising edge (interleaving a valid step to reset the edge)
# so the "new event, but cooldown suppresses it" scenario is unambiguous.
def test_cooldown_blocks_retrigger():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=3, max_interventions=20, patch_duration_steps=1)
    controller = StatefulController(cfg)
    invalid_step = dict(action="foo", observation="Nothing happens.",
                        admissible_before=["go to bed 1"], recent_actions=[], recent_observations=[],
                        max_steps=50, is_think_action=False)
    valid_step = dict(action="go to bed 1", observation="You arrive at bed 1, a brand new place.",
                      admissible_before=["go to bed 1"], recent_actions=[], recent_observations=[],
                      max_steps=50, is_think_action=False)

    r1 = controller.step(**invalid_step)  # invalid_action rising edge (False->True): fires
    check("6. invalid_action rising edge triggers an intervention",
          r1["intervention"] != InterventionType.CONTINUE.value, r1["intervention"])
    check("6b. cooldown is armed after the trigger", controller.cooldown_remaining > 0,
          controller.cooldown_remaining)

    controller.step(**valid_step)  # invalid_action -> False, resets the edge; cooldown ticks 3->2
    r3 = controller.step(**invalid_step)  # a FRESH rising edge, but cooldown_remaining is still 2>0
    check("6c. a fresh rising edge within the cooldown window is suppressed by cooldown, not ignored as 'no event'",
          r3["intervention"] == InterventionType.CONTINUE.value and "cooldown" in r3["intervention_reason"],
          r3["intervention_reason"])

    controller.step(**valid_step)  # cooldown ticks 1->0
    r5 = controller.step(**invalid_step)  # cooldown now fully elapsed
    check("6d. once cooldown has fully elapsed, the same rising-edge pattern fires again",
          r5["intervention"] != InterventionType.CONTINUE.value, r5["intervention_reason"])


# ── 6e-6h. StateSignature-based event detection (this revision's core fix) ──
def test_severity_escalation_hidden_by_dominant_mode():
    """The bug this revision fixes: frustration already high (dominant
    mode FRUSTRATED, an intervention already active) and confidence THEN
    also drops low -> replan should fire even though FRUSTRATED stays the
    logged dominant mode throughout (mode-priority collapse hides the
    simultaneously-active confidence_low band)."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=1)
    controller = StatefulController(cfg)
    invalid_step = dict(action="foo", observation="Nothing happens.",
                        admissible_before=["go to bed 1"], recent_actions=[], recent_observations=[],
                        max_steps=50, is_think_action=False)
    saw_frustrated_with_confidence_low = False
    saw_replan_while_dominant_mode_frustrated = False
    r = None
    for _ in range(20):
        r = controller.step(**invalid_step)
        if r["current_mode"] == "FRUSTRATED" and r["state_signature"]["confidence_low"]:
            saw_frustrated_with_confidence_low = True
            if r["intervention"] == InterventionType.REPLAN.value:
                saw_replan_while_dominant_mode_frustrated = True
    check("6e. reached a step where dominant mode=FRUSTRATED but confidence_low is ALSO latched "
          "(exactly what mode-collapse used to hide)",
          saw_frustrated_with_confidence_low)
    check("6f. replan fired once confidence_low also latched, despite dominant mode never leaving FRUSTRATED",
          saw_replan_while_dominant_mode_frustrated, f"last record: {r}")


def test_no_retrigger_while_condition_persists():
    """Same abnormal condition sustained unchanged (no severity increase,
    no rising edges) must not refire every step -- tested in isolation
    from the intervention-outcome-tracking mechanism (Part F) by using a
    patch_duration_steps long enough that no patch ever expires/gets
    evaluated within this test's step count. (With a short patch duration
    an unresolved outcome is SUPPOSED to escalate and refire even while
    the condition persists -- that's Part F's whole point, covered by
    test_unresolved_outcome_escalates_after_cooldown below, not this one.)"""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=50, patch_duration_steps=100)
    controller = StatefulController(cfg)
    invalid_step = dict(action="foo", observation="Nothing happens.",
                        admissible_before=["go to bed 1"], recent_actions=[], recent_observations=[],
                        max_steps=50, is_think_action=False)
    for _ in range(15):  # drive well past the point where severity/flags stabilize
        controller.step(**invalid_step)
    fires_before = controller.intervention_count
    for _ in range(10):  # identical signal, nothing new to escalate or re-enter
        controller.step(**invalid_step)
    fires_after = controller.intervention_count
    check("6g. no new intervention fires while the exact same condition persists unchanged",
          fires_after == fires_before, f"{fires_before} -> {fires_after}")


def test_reentry_after_exit_retriggers():
    """An abnormal band that exits (drops below its hysteresis exit
    threshold) and later re-enters must be able to trigger again."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=50, patch_duration_steps=1)
    controller = StatefulController(cfg)
    invalid_step = dict(action="foo", observation="Nothing happens.",
                        admissible_before=["go to bed 1"], recent_actions=[], recent_observations=[],
                        max_steps=50, is_think_action=False)
    valid_step = dict(action="go to bed 1", observation="You arrive at bed 1, a brand new place with lots to see.",
                      admissible_before=["go to bed 1"], recent_actions=[], recent_observations=[],
                      max_steps=50, is_think_action=False)

    for _ in range(3):
        controller.step(**invalid_step)
    check("6h. frustration_high latched on initial entry", controller.hysteresis.frustration_active)
    fires_at_entry = controller.intervention_count

    for _ in range(15):  # recovery: valid + novel + no invalid_action -> frustration decays out
        controller.step(**valid_step)
    check("6i. frustration_high exited after sustained recovery", not controller.hysteresis.frustration_active)

    for _ in range(3):  # re-enter
        controller.step(**invalid_step)
    check("6j. frustration_high re-entered", controller.hysteresis.frustration_active)
    check("6k. a NEW intervention fired on re-entry, not just the original one at first entry",
          controller.intervention_count > fires_at_entry,
          f"{fires_at_entry} -> {controller.intervention_count}")


# ── 7. max_interventions enforced ────────────────────────────────────────
def test_max_interventions():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=1, max_interventions=2, patch_duration_steps=1)
    controller = StatefulController(cfg)
    invalid_step = dict(action="foo", observation="Nothing happens.",
                        admissible_before=["go to bed 1"], recent_actions=[], recent_observations=[],
                        max_steps=50, is_think_action=False)
    triggered = 0
    for _ in range(12):
        r = controller.step(**invalid_step)
        if r["intervention"] != InterventionType.CONTINUE.value:
            triggered += 1
    check("7. total interventions never exceeds max_interventions",
          triggered <= cfg.max_interventions and controller.intervention_count == cfg.max_interventions,
          f"triggered={triggered}, controller.intervention_count={controller.intervention_count}")


# ── 8. react mode never injects a directive ──────────────────────────────
class FakeLLM:
    """Returns a fixed action sequence, ignoring the prompt content."""
    def __init__(self, actions):
        self._actions = list(actions)
        self.seen_prompts = []
        self.model = "Qwen/Qwen3-8B"  # real model name so count_tokens() can load a tokenizer

    def __call__(self, prompt: str) -> str:
        self.seen_prompts.append(prompt)
        return self._actions.pop(0) if self._actions else "look"


class FakeEnv:
    """Minimal stand-in for ALFWorld's batched env interface, just enough
    for ALFWorldAgent.run(): reset()/step()/close(), info carries
    admissible_commands + won like the real AlfredTWEnv (confirmed
    empirically 2026-07-25)."""
    def __init__(self, n_steps_until_done=4):
        self.n_steps_until_done = n_steps_until_done
        self._step = 0

    def reset(self):
        self._step = 0
        ob = "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in a room.\n\nYour task is to: put a mug in the sink."
        return [ob], {"admissible_commands": [["go to sink 1", "go to table 1"]], "won": [False]}

    def step(self, actions):
        self._step += 1
        done = self._step >= self.n_steps_until_done
        ob = "Nothing happens." if not done else "You put the mug in the sink. Task complete."
        return [ob], [1 if done else 0], [done], {
            "admissible_commands": [["go to sink 1", "go to table 1"]], "won": [done],
        }

    def close(self):
        pass


class ScriptedFakeEnv:
    """FakeEnv whose step() outputs are fully scripted per call: each
    entry is (observation, reward, done, won), independently controllable
    — used to directly test the reward/done/won disambiguation end-to-end
    through ALFWorldAgent.run(). The script's last entry repeats once
    exhausted. Tracks step_calls so tests can confirm whether env.step()
    was actually invoked (react mode's exhaustion check must break BEFORE
    stepping, exactly as before this revision)."""

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


# ── Part 2: unified success determination (won is the sole source of truth) ──
def test_bool_scalar_parsing():
    from agents import _to_bool_scalar, _batch_first

    check("S1. list-wrapped False -> False (NOT bare-list truthiness: a non-empty "
          "[False] is a truthy container around a False value)",
          _to_bool_scalar([False]) is False)
    check("S2. list-wrapped True -> True", _to_bool_scalar([True]) is True)
    check("S3. plain bool scalar passes through", _to_bool_scalar(True) is True and _to_bool_scalar(False) is False)
    check("S4. tuple format works", _to_bool_scalar((False,)) is False)
    try:
        import numpy as np
        check("S5. numpy scalar / 0-d array works",
              _to_bool_scalar(np.array(False)) is False and _to_bool_scalar(np.bool_(True)) is True)
    except ImportError:
        check("S5. numpy scalar / 0-d array works (numpy unavailable, skipped)", True)
    check("S6. _batch_first preserves numeric reward without casting to bool "
          "(reward is logged raw, never treated as a success signal)",
          _batch_first([0.0]) == 0.0 and _batch_first([1]) == 1)


def test_success_determination_integration():
    from agents import ALFWorldAgent

    # done=True but won=False, with a TRUTHY reward -> this is the exact
    # false-positive shape the pre-revision bug produced; must be failure.
    llm = FakeLLM(["go to sink 1"] * 5)
    env = ScriptedFakeEnv([("Nothing happens.", 1, True, False)])
    agent = ALFWorldAgent(llm, controller=None)
    _, success = agent.run(env, "task", "put", to_print=False)
    check("S7. done=True, reward=1 (truthy), won=False -> failure (won is the sole source of truth)",
          success is False)

    # done=True, won=True, with a FALSY reward -> must still be success.
    llm = FakeLLM(["go to sink 1"] * 5)
    env = ScriptedFakeEnv([("You win.", 0, True, True)])
    agent = ALFWorldAgent(llm, controller=None)
    _, success = agent.run(env, "task", "put", to_print=False)
    check("S8. done=True, reward=0 (falsy), won=True -> success (won is the sole source of truth)",
          success is True)

    # Never done at all (times out at MAX_STEPS) -> failure. This is the
    # regression check for "running out the step budget must never again
    # be misreported as success."
    llm = FakeLLM([f"go to sink 1 v{i}" for i in range(60)])
    env = ScriptedFakeEnv([("Nothing happens.", 0, False, False)])
    agent = ALFWorldAgent(llm, controller=None)
    _, success = agent.run(env, "task", "put", to_print=False)
    check("S9. running out MAX_STEPS without env done -> failure, not a false-positive success",
          success is False)


# ── Part 3: intervention vs exhausted_repeated execution order + recovery grace ──
def test_recovery_grace_suppresses_immediate_termination():
    from agents import ALFWorldAgent

    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=5,
                    patch_duration_steps=3, recovery_grace_steps=3)
    controller = StatefulController(cfg)
    # Repeats an inadmissible action forever: invalid_action fires an
    # intervention on step 1, and the identical string trips the
    # action==last_action exhaustion check from step 2 onward. Without
    # grace this would end the episode after essentially one action ever
    # reaching the environment.
    llm = FakeLLM(["go to nowhere"] * 60)
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="legacy_early_stop")
    env = ScriptedFakeEnv([("Nothing happens.", 0, False, False)], admissible=("go to sink 1",))
    trajectory, success = agent.run(env, "task", "put", to_print=False)

    check("G1. an intervention fired (invalid_action)", controller.intervention_count > 0)
    check("G2. repeated-action termination was suppressed at least once by recovery grace",
          controller.suppressed_exhausted_count > 0, controller.suppressed_exhausted_count)
    # G3 updated: continuously repeating the exact same action WHILE an
    # intervention is still pending is now caught by the higher-priority
    # post-intervention-exact-repeat rule (see stateful_controller.py).
    # Deterministic trace with THIS scenario's config (max_interventions=5,
    # so budget is never the binding constraint): step1 invalid_action ->
    # REFLECT arms; step2 exact repeat while REFLECT pending -> escalates
    # to REPLAN (intervention_count=2); step3 exact repeat while REPLAN
    # pending -> terminates. The reason is therefore always exactly
    # post_replan_exact_repeat here, never the budget-exhausted variant
    # (confirmed by direct run, not assumed) -- asserted precisely rather
    # than accepting either reason, since this scenario's outcome is fully
    # determined.
    check("G3. the episode terminates via the higher-priority "
          "post-intervention exact-repeat rule (post_replan_exact_repeat), "
          "not exhausted_repeated",
          agent.termination_reason == "post_replan_exact_repeat",
          getattr(agent, "termination_reason", None))
    check("G3b. intervention_count is exactly 2 (REFLECT then escalated to "
          "REPLAN, never a 3rd since REPLAN terminates rather than escalating)",
          controller.intervention_count == 2, controller.intervention_count)
    check("G4. grace never let the loop run past MAX_STEPS", len(controller.step_log) <= 50)
    check("G5. more than a single action actually reached the environment "
          "(grace gave the directive a real chance, not zero extra steps)",
          env.step_calls > 1, env.step_calls)


def test_recovery_grace_never_bypasses_env_terminal():
    from agents import ALFWorldAgent

    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=5,
                    patch_duration_steps=3, recovery_grace_steps=3)
    controller = StatefulController(cfg)
    llm = FakeLLM(["go to nowhere"] * 10)
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="legacy_early_stop")
    # done=True/won=True on the very first (repeated/invalid) step — a real
    # terminal must be honored immediately regardless of grace bookkeeping.
    env = ScriptedFakeEnv([("You win.", 1, True, True)], admissible=("go to sink 1",))
    _, success = agent.run(env, "task", "put", to_print=False)
    check("G6. a genuine done=True/won=True is honored immediately, never suppressed by grace",
          success is True)


def test_react_mode_exhaustion_unaffected():
    from agents import ALFWorldAgent, _EXHAUSTED_MSG

    llm = FakeLLM(["go to nowhere", "go to nowhere"] + ["go to sink 1"] * 10)
    agent = ALFWorldAgent(llm, controller=None, termination_policy="legacy_early_stop")
    env = ScriptedFakeEnv([("Nothing happens.", 0, False, False)], admissible=("go to sink 1",))
    trajectory, success = agent.run(env, "task", "put", to_print=False)
    check("G7. react mode: an exact repeated action still ends the episode immediately",
          success is False and _EXHAUSTED_MSG in trajectory)
    check("G8. react mode: the repeated action broke BEFORE it ever reached env.step() "
          "(original ordering fully preserved — no grace mechanism applies)",
          env.step_calls == 1, env.step_calls)


def test_react_mode_no_directive():
    from agents import ALFWorldAgent  # alfworld_runs_ae

    # Distinct action strings each step: ALFWorldAgent.run()'s exhaustion
    # check (action == last_action) would otherwise halt the loop after 2
    # steps regardless of FakeEnv's done signal, since a real repeated
    # identical action is legitimately meant to trip that check.
    llm = FakeLLM([f"go to sink 1 (attempt {i})" for i in range(10)])
    agent = ALFWorldAgent(llm, controller=None)
    env = FakeEnv(n_steps_until_done=4)
    trajectory, success = agent.run(env, "put a mug in the sink", "put", to_print=False)

    check("8. controller=None (react mode) never shows a directive in any prompt",
          all("[ACTIVE CONTROL DIRECTIVE]" not in p for p in llm.seen_prompts))
    check("8b. controller=None still runs the ordinary ReAct loop to completion",
          success is True)


def test_ae_full_wiring_smoke():
    """Not one of the 9 required cases, but a cheap end-to-end sanity check
    that the controller is actually reached when wired in (repeats an
    invalid action so an intervention should fire), using the same
    FakeEnv/FakeLLM as test 8."""
    from agents import ALFWorldAgent

    cfg = AEConfig(warmup_steps=0, cooldown_steps=1, max_interventions=3, patch_duration_steps=2)
    controller = StatefulController(cfg)
    # go to a location NOT in admissible_commands -> invalid_action=1.0 each time,
    # but vary the string slightly so the exhaustion check (action==last_action) doesn't fire.
    llm = FakeLLM([f"go to nowhere {i}" for i in range(8)])
    agent = ALFWorldAgent(llm, controller=controller)
    env = FakeEnv(n_steps_until_done=999)  # never naturally completes; loop runs MAX_STEPS or until exhausted
    trajectory, success = agent.run(env, "put a mug in the sink", "put", to_print=False)
    saw_directive = any("[ACTIVE CONTROL DIRECTIVE]" in p for p in llm.seen_prompts)
    check("bonus. ae_full wiring: an intervention directive is actually shown to the LLM",
          saw_directive)


# ── 9. episode reset has no cross-episode leakage ────────────────────────
def test_reset_no_leakage():
    cfg = AEConfig(warmup_steps=0, cooldown_steps=1, max_interventions=5, patch_duration_steps=2)
    controller = StatefulController(cfg)
    invalid_step = dict(action="foo", observation="Nothing happens.",
                        admissible_before=["go to bed 1"], recent_actions=[], recent_observations=[],
                        max_steps=50, is_think_action=False)
    for _ in range(6):
        controller.step(**invalid_step)
    check("9a. state drifted away from initial before reset",
          controller.state.frustration != cfg.initial_frustration or controller.intervention_count > 0)

    controller.reset()
    fresh = initial_state(cfg)
    check("9b. reset() restores exact initial AffectState",
          controller.state.uncertainty == fresh.uncertainty
          and controller.state.frustration == fresh.frustration
          and controller.state.surprise == fresh.surprise
          and controller.state.confidence == fresh.confidence)
    check("9c. reset() clears intervention count / trigger steps / step log",
          controller.intervention_count == 0 and controller.trigger_steps == [] and controller.step_log == [])
    check("9d. reset() clears hysteresis latches and cooldown/patch bookkeeping",
          controller.hysteresis.frustration_active is False
          and controller.cooldown_remaining == 0
          and controller.active_patch_type is None)


# ── frustration_medium repeat-gate fix: consecutive_exact_action_repeat ──
# Real A40 audit finding: `frustration_medium AND (repeated_action OR
# repeated_observation)` fired 19/19 times purely off repeated_observation
# (Jaccard over near-identical ALFWorld template text like "On the shelf N,
# you see nothing."), never once off a genuine repeated action -- routine,
# never-repeating exploration was misjudged as "stuck". The fix gates on
# consecutive_exact_action_repeat (strict adjacent-step canonical-or-
# cleaned-string exact match) instead.
def test_routine_shelf_exploration_does_not_trigger_medium_frustration_reflect():
    """idx=0's real pattern: go to shelf 1/2/3/4/5/6, each a genuinely
    different, never-before-visited location, each returning a
    near-identical 'On the shelf N, you see nothing.' template (or a
    non-empty listing). Even if frustration climbs past frustration_medium
    (0.45) from the accumulated repeated_observation signal (unchanged,
    still feeds affect_state as before), reflect must NOT fire, since no
    two consecutive actions are ever an exact repeat."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=1)
    controller = StatefulController(cfg)
    shelves = [
        ("go to shelf 1", "You arrive at shelf 1. On the shelf 1, you see nothing."),
        ("go to shelf 2", "You arrive at shelf 2. On the shelf 2, you see nothing."),
        ("go to shelf 3", "You arrive at shelf 3. On the shelf 3, you see nothing."),
        ("go to shelf 4", "You arrive at shelf 4. On the shelf 4, you see nothing."),
        ("go to shelf 5", "You arrive at shelf 5. On the shelf 5, you see nothing."),
        ("go to shelf 6", "You arrive at shelf 6. On the shelf 6, you see nothing."),
    ]
    recent_actions, recent_observations = [], []
    saw_frustration_medium = False
    reflect_fired = False
    for action, obs in shelves:
        r = controller.step(
            action=action, observation=obs, admissible_before=[f"go to shelf {i}" for i in range(1, 7)],
            recent_actions=list(recent_actions), recent_observations=list(recent_observations),
            max_steps=50, is_think_action=False,
        )
        if r["state_signature"]["frustration_medium"]:
            saw_frustration_medium = True
        if r["intervention"] == InterventionType.REFLECT.value:
            reflect_fired = True
        recent_actions.append(action)
        recent_observations.append(obs)
    check("F1. frustration_medium was actually reached in this scenario "
          "(so the non-trigger below is a real gate check, not a vacuous one)",
          saw_frustration_medium)
    check("F2. reflect never fires for routine never-repeating shelf "
          "exploration, even once frustration_medium is active",
          not reflect_fired)


def test_genuine_consecutive_repeat_still_triggers_medium_frustration_reflect():
    """The positive control: a real stuck loop (same exact action twice in
    a row, sustained) must still fire reflect once frustration crosses
    frustration_medium -- the fix must not silently disable this branch
    entirely, only its false-positive trigger."""
    # "go to shelf 1" (exploratory family) is used rather than a
    # state-changing action like "use desklamp 1" -- state-changing
    # families producing no effect also raise unexpected_outcome/surprise
    # (see signals.py), which can trigger a VERIFY/escalation path before
    # frustration_medium is ever reached, confounding this specific check.
    # Exploratory families are exempt from unexpected_outcome by design.
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=1)
    controller = StatefulController(cfg)
    recent_actions, recent_observations = [], []
    reflect_fired = False
    reflect_reason = None
    for _ in range(6):
        r = controller.step(
            action="go to shelf 1", observation="Nothing happens.",
            admissible_before=["go to shelf 1"],
            recent_actions=list(recent_actions), recent_observations=list(recent_observations),
            max_steps=50, is_think_action=False,
        )
        if r["intervention"] == InterventionType.REFLECT.value:
            reflect_fired = True
            reflect_reason = r["intervention_reason"]
            break
        recent_actions.append("go to shelf 1")
        recent_observations.append("Nothing happens.")
    check("F3. a genuine consecutive exact repeat still triggers reflect "
          "once frustration_medium is reached", reflect_fired)
    check("F4. the logged reason explicitly names consecutive_exact_action_repeat, "
          "not the old vague 'repeated action/observation'",
          reflect_reason is not None and "consecutive_exact_action_repeat" in reflect_reason,
          reflect_reason)


def test_invalid_action_branch_unaffected_by_repeat_gate_change():
    """invalid_action -> REFLECT must still fire on the very first
    occurrence (rising edge), completely independent of
    consecutive_exact_action_repeat -- this branch never depended on
    `repeated` at all (see _select_intervention: `if invalid: return
    REFLECT, "invalid_action"` short-circuits before the repeat check)."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=1)
    controller = StatefulController(cfg)
    r = controller.step(
        action="foo", observation="Nothing happens.", admissible_before=["go to bed 1"],
        recent_actions=[], recent_observations=[], max_steps=50, is_think_action=False,
    )
    check("F5. invalid_action still fires REFLECT with reason=='invalid_action', unaffected",
          r["intervention"] == InterventionType.REFLECT.value and r["intervention_reason"] == "invalid_action",
          (r["intervention"], r["intervention_reason"]))


def test_frustration_high_confidence_low_branch_unaffected_by_repeat_gate_change():
    """frustration_high AND confidence_low -> REPLAN is checked BEFORE the
    frustration_medium/repeated branch in _select_intervention and never
    reads `repeated` at all -- must still fire exactly as before."""
    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=20, patch_duration_steps=1)
    controller = StatefulController(cfg)
    saw_replan_with_frustration_high_and_confidence_low = False
    for _ in range(20):
        r = controller.step(
            action="foo", observation="Nothing happens.", admissible_before=["go to bed 1"],
            recent_actions=[], recent_observations=[], max_steps=50, is_think_action=False,
        )
        if (r["state_signature"]["frustration_high"] and r["state_signature"]["confidence_low"]
                and r["intervention"] == InterventionType.REPLAN.value):
            saw_replan_with_frustration_high_and_confidence_low = True
            break
    check("F6. frustration_high+confidence_low -> REPLAN still fires, unaffected by the repeat-gate change",
          saw_replan_with_frustration_high_and_confidence_low)


if __name__ == "__main__":
    test_bounds()
    test_frustration_accumulates()
    test_progress_relieves()
    test_surprise_decays_faster()
    test_hysteresis_no_flicker()
    test_cooldown_blocks_retrigger()
    test_severity_escalation_hidden_by_dominant_mode()
    test_no_retrigger_while_condition_persists()
    test_reentry_after_exit_retriggers()
    test_max_interventions()
    test_bool_scalar_parsing()
    test_success_determination_integration()
    test_routine_shelf_exploration_does_not_trigger_medium_frustration_reflect()
    test_genuine_consecutive_repeat_still_triggers_medium_frustration_reflect()
    test_invalid_action_branch_unaffected_by_repeat_gate_change()
    test_frustration_high_confidence_low_branch_unaffected_by_repeat_gate_change()
    test_recovery_grace_suppresses_immediate_termination()
    test_recovery_grace_never_bypasses_env_terminal()
    test_react_mode_exhaustion_unaffected()
    test_react_mode_no_directive()
    test_ae_full_wiring_smoke()
    test_reset_no_leakage()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
