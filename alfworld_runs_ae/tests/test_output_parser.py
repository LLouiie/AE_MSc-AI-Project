"""Tests for the shared Thought/Action output parser and its wiring into
ALFWorldAgent.run() (react.py, reflexion.py, and ae_full.py all go through
this exact same code path -- there is no AE-specific parser).

Plain assert-based, same convention as ae/controllers/tests/test_ae_controller.py.
Run directly:
    python3 alfworld_runs_ae/tests/test_output_parser.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # alfworld_runs_ae

from output_parser import parse_agent_output  # noqa: E402

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


# ── 1. standard labeled two-line form ────────────────────────────────────
def test_standard_thought_action():
    p = parse_agent_output("Thought: I should check the fridge.\nAction: go to fridge 1")
    check("1. standard Thought/Action parses thought", p.parsed_thought == "I should check the fridge.")
    check("1b. standard Thought/Action parses action", p.parsed_action == "go to fridge 1")
    check("1c. standard Thought/Action reports success", p.parse_success is True)
    check("1d. standard Thought/Action has no failure reason", p.parse_failure_reason is None)


# ── 2. Action-only line (no Thought:) ────────────────────────────────────
def test_action_only():
    p = parse_agent_output("Action: take mug 1 from countertop 1")
    check("2. Action-only parses the action", p.parsed_action == "take mug 1 from countertop 1")
    check("2b. Action-only has no thought", p.parsed_thought is None)
    check("2c. Action-only reports success", p.parse_success is True)


# ── 3. bare legal command, no labels at all ──────────────────────────────
def test_bare_command():
    p = parse_agent_output("go to shelf 2")
    check("3. bare command parses as the action verbatim", p.parsed_action == "go to shelf 2")
    check("3b. bare command has no thought", p.parsed_thought is None)
    check("3c. bare command reports success", p.parse_success is True)


def test_bare_think_pseudo_action_backward_compat():
    """The pre-existing alfworld_3prompts.json few-shot style: a bare
    "think: ..." line with no labels is still accepted as a pseudo-action,
    unchanged from before this revision."""
    p = parse_agent_output("think: I need to find a bowl next.")
    check("3d. bare 'think:' pseudo-action still parses (backward compat)",
          p.parsed_action == "think: I need to find a bowl next.")
    check("3e. bare 'think:' pseudo-action reports success", p.parse_success is True)


# ── 4. truncated Thought with no Action -> parse failure, no fabrication ──
def test_truncated_thought_no_action():
    p = parse_agent_output("Thought: I should open the drawer because it might contain")
    check("4. truncated thought-only reports parse failure", p.parse_success is False)
    check("4b. truncated thought-only extracts no action", p.parsed_action is None)
    check("4c. truncated thought-only keeps the partial thought for logging",
          p.parsed_thought is not None and p.parsed_thought.startswith("I should open"))
    check("4d. truncated thought-only reason is truncated_no_action",
          p.parse_failure_reason == "truncated_no_action")


def test_empty_generation():
    p = parse_agent_output("")
    check("4e. empty generation reports parse failure", p.parse_success is False)
    check("4f. empty generation reason is empty_generation", p.parse_failure_reason == "empty_generation")


# ── 5. free-form reasoning text is REJECTED, not silently executed ──────
def test_free_form_reasoning_rejected_as_bare_action():
    """A full sentence of free-form reasoning with no Thought:/Action:
    labels and no resemblance to the ALFWorld action grammar must be
    reported as a parse failure, never silently accepted as a literal
    (nonsensical) action -- bug found 2026-07-25 while auditing Part G
    trajectories."""
    prose = "Okay, let me try to figure this out. The task is to find a vase and put it somewhere safe."
    p = parse_agent_output(prose)
    check("5. free-form reasoning is not accepted as an action", p.parsed_action is None)
    check("5b. free-form reasoning reports parse failure", p.parse_success is False)
    check("5c. free-form reasoning reason is unrecognised_bare_text",
          p.parse_failure_reason == "unrecognised_bare_text")


def test_multiclause_garbage_not_auto_corrected_and_not_accepted():
    """The previous version of this test asserted multi-clause garbage was
    accepted verbatim as a bare action (parse_success=True). That was
    itself the bug (Part G finding #2): free text with no grammar match
    and no admissible_commands match must now be REJECTED, not accepted
    verbatim -- but it also must never be silently replaced by a guessed
    or nearest-admissible command; it just becomes parsed_action=None."""
    garbage = "1. Check cabinets. Open cabinet 1, 2, 3, 4, 5, 6."
    p = parse_agent_output(garbage)
    check("5d. multi-clause garbage is not returned as the parsed action",
          p.parsed_action != garbage)
    check("5e. multi-clause garbage is not replaced by any other literal action either",
          p.parsed_action is None)
    check("5f. multi-clause garbage reports parse failure",
          p.parse_success is False and p.parse_failure_reason == "unrecognised_bare_text")


# ── 5g-5j. legacy '>' prefix / enumeration stripped, semantics preserved ─
def test_labeled_action_strips_legacy_caret_prefix():
    p = parse_agent_output("Action: > open fridge 1")
    check("5g. labeled 'Action: > open fridge 1' strips the stray '>' prefix",
          p.parsed_action == "open fridge 1", p.parsed_action)
    check("5h. labeled form with stray caret still reports success", p.parse_success is True)


def test_bare_legacy_caret_form_parses():
    p = parse_agent_output("> open fridge 1")
    check("5i. bare '> open fridge 1' parses to 'open fridge 1'",
          p.parsed_action == "open fridge 1", p.parsed_action)
    check("5j. bare legacy caret form reports success", p.parse_success is True)


def test_bare_action_recognized_via_admissible_commands_not_grammar():
    """A bare action that doesn't match any known canonicalize_action
    family (e.g. a task-specific admissible command the family regexes
    don't cover) must still be accepted if it exactly matches one of the
    environment's current admissible_commands, passed in by the caller."""
    p = parse_agent_output("examine the vase", admissible_commands=["examine the vase", "go to shelf 1"])
    check("5k. bare action matching admissible_commands (not grammar) is accepted",
          p.parsed_action == "examine the vase" and p.parse_success is True, p)


def test_bare_action_not_in_grammar_or_admissible_is_rejected():
    p = parse_agent_output("dance around the room", admissible_commands=["go to shelf 1", "open drawer 1"])
    check("5l. bare action matching neither grammar nor admissible_commands is rejected",
          p.parsed_action is None and p.parse_success is False
          and p.parse_failure_reason == "unrecognised_bare_text", p)


# ── ALFWorldAgent integration: environment never receives raw_generation,
#    parser identical across baselines, directive doesn't contaminate ─────
class FakeLLMScripted:
    """Returns one scripted raw completion per call, ignoring the prompt
    content (mirrors ae/controllers/tests/test_ae_controller.py's FakeLLM,
    duplicated locally to keep this test file self-contained)."""
    def __init__(self, generations):
        self._gens = list(generations)
        self.seen_prompts = []
        self.last_meta = None
        self.model = "Qwen/Qwen3-8B"  # real model name so count_tokens() can load a tokenizer

    def __call__(self, prompt: str) -> str:
        self.seen_prompts.append(prompt)
        gen = self._gens.pop(0) if self._gens else "Action: look"
        self.last_meta = {"finish_reason": "stop", "was_truncated": False, "usage": "unavailable"}
        return gen


class FakeEnv:
    def __init__(self, n_steps_until_done=999):
        self.n_steps_until_done = n_steps_until_done
        self._step = 0
        self.received_actions = []

    def reset(self):
        self._step = 0
        ob = "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in a room.\n\nYour task is to: put a mug in the sink."
        return [ob], {"admissible_commands": [["go to sink 1", "go to table 1"]], "won": [False]}

    def step(self, actions):
        self.received_actions.append(actions[0])
        self._step += 1
        done = self._step >= self.n_steps_until_done
        ob = "Nothing happens." if not done else "You put the mug in the sink. Task complete."
        return [ob], [1 if done else 0], [done], {
            "admissible_commands": [["go to sink 1", "go to table 1"]], "won": [done],
        }

    def close(self):
        pass


def test_truncated_no_action_never_reaches_environment():
    from agents import ALFWorldAgent

    # Step 1: a clean Thought/Action pair -> env.step() called once.
    # Step 2: Thought with no Action line at all -> must NOT reach env.step().
    # Step 3: recovers with a clean pair again.
    llm = FakeLLMScripted([
        "Thought: I should look around.\nAction: go to sink 1",
        "Thought: I should check somewhere else because",
        "Thought: Let me try the table.\nAction: go to table 1",
    ])
    env = FakeEnv(n_steps_until_done=999)
    agent = ALFWorldAgent(llm, termination_policy="fixed_horizon")
    import agents as agents_mod
    agents_mod.MAX_STEPS = 3
    agent.run(env, "put a mug in the sink", "put", to_print=False)

    check("6. exactly 2 real actions reached env.step() (the parse-failure "
          "step did not)", env.received_actions == ["go to sink 1", "go to table 1"],
          env.received_actions)
    check("6b. the parse-failure step is logged with parse_success=False",
          agent.step_log[1]["parse_success"] is False)
    check("6c. the parse-failure step's parsed_action is None",
          agent.step_log[1]["parsed_action"] is None)
    check("6d. the raw multi-clause garbage was never handed to env.step() "
          "verbatim in any step", all("Thought:" not in a for a in env.received_actions))


def test_react_and_ae_full_share_identical_parser():
    """react.py and ae_full.py both go through ALFWorldAgent.run(), whose
    only parsing call is output_parser.parse_agent_output -- there is no
    separate AE-only parsing path. Verified by confirming both a plain
    (controller=None) and an ae_full-style (controller attached) run
    produce byte-identical parsed_thought/parsed_action/parse_success for
    the same raw generation."""
    from agents import ALFWorldAgent
    from ae.controllers.config import AEConfig
    from ae.controllers.stateful_controller import StatefulController
    import agents as agents_mod
    agents_mod.MAX_STEPS = 2

    gens = ["Thought: I should look around.\nAction: go to sink 1",
            "Thought: Done looking.\nAction: go to table 1"]

    llm_react = FakeLLMScripted(list(gens))
    agent_react = ALFWorldAgent(llm_react, controller=None, termination_policy="fixed_horizon")
    agent_react.run(FakeEnv(), "put a mug in the sink", "put", to_print=False)

    llm_ae = FakeLLMScripted(list(gens))
    controller = StatefulController(AEConfig())
    agent_ae = ALFWorldAgent(llm_ae, controller=controller, termination_policy="fixed_horizon")
    agent_ae.run(FakeEnv(), "put a mug in the sink", "put", to_print=False)

    react_parsed = [(s["parsed_thought"], s["parsed_action"], s["parse_success"]) for s in agent_react.step_log]
    ae_parsed = [(s["parsed_thought"], s["parsed_action"], s["parse_success"]) for s in agent_ae.step_log]
    check("7. react and ae_full produce identical parsed output for the "
          "same raw generations", react_parsed == ae_parsed, (react_parsed, ae_parsed))


def test_controller_directive_does_not_contaminate_parsing():
    """The controller's directive text is spliced into the PROMPT, never
    into the completion the parser reads -- so directive wording (which
    contains neither "Thought:" nor "Action:" labels, see
    intervention_renderer.py) can never be mistaken for the model's own
    output. This test forces an intervention to be active and confirms
    the directive shows up in the prompt sent to the LLM but the parsed
    action still comes only from the (separately scripted) completion."""
    from agents import ALFWorldAgent
    from ae.controllers.config import AEConfig
    from ae.controllers.stateful_controller import StatefulController
    import agents as agents_mod
    agents_mod.MAX_STEPS = 3

    cfg = AEConfig(warmup_steps=0, cooldown_steps=0, max_interventions=3, patch_duration_steps=3)
    controller = StatefulController(cfg)
    # An inadmissible action on step 1 triggers an intervention (REFLECT),
    # whose directive should appear in the step-2 prompt.
    llm = FakeLLMScripted([
        "Thought: try somewhere odd.\nAction: go to nowhere",
        "Thought: recovering.\nAction: go to sink 1",
        "Thought: done.\nAction: go to table 1",
    ])
    agent = ALFWorldAgent(llm, controller=controller, termination_policy="fixed_horizon")
    agent.run(FakeEnv(), "put a mug in the sink", "put", to_print=False)

    saw_directive_in_prompt = any("[ACTIVE CONTROL DIRECTIVE" in p for p in llm.seen_prompts)
    check("8. a directive was actually shown in some prompt", saw_directive_in_prompt)
    check("8b. no directive text leaked into any parsed_action",
          all("ACTIVE CONTROL DIRECTIVE" not in (s["parsed_action"] or "") for s in agent.step_log))
    check("8c. no directive text leaked into any parsed_thought",
          all("ACTIVE CONTROL DIRECTIVE" not in (s["parsed_thought"] or "") for s in agent.step_log))


if __name__ == "__main__":
    test_standard_thought_action()
    test_action_only()
    test_bare_command()
    test_bare_think_pseudo_action_backward_compat()
    test_truncated_thought_no_action()
    test_empty_generation()
    test_free_form_reasoning_rejected_as_bare_action()
    test_multiclause_garbage_not_auto_corrected_and_not_accepted()
    test_labeled_action_strips_legacy_caret_prefix()
    test_bare_legacy_caret_form_parses()
    test_bare_action_recognized_via_admissible_commands_not_grammar()
    test_bare_action_not_in_grammar_or_admissible_is_rejected()
    test_truncated_no_action_never_reaches_environment()
    test_react_and_ae_full_share_identical_parser()
    test_controller_directive_does_not_contaminate_parsing()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
