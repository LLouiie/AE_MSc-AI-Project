"""Integration tests for the put->move ALFWorld action-protocol compat fix,
wired through the real ALFWorldAgent.run() loop (alfworld_runs_ae/agents.py)
-- not just the standalone normalize_alfworld_action() unit tests in
test_alfworld_action_normalize.py.

Covers, with a mocked env/LLM (no real ALFWorld env, no real LLM call):
  11. env.step() actually receives "move X to Y", never "put X in/on Y".
  12. react (controller=None), reflexion (ALFWorldReflectAgent), and
      ae_full (controller attached) all go through the identical fix.
  13. the per-step log carries proposed/parsed action AND executed action
      as distinct fields.
  14. an unconverted action logs parsed_action == executed_action,
      action_normalized=False.
  15. normalization adds no extra LLM call, no extra environment step, and
      does not change the ReAct repeated-action termination rule (which
      must keep comparing the model's own un-normalized parsed_action, not
      the executed/normalized action -- two differently-worded "put"
      paraphrases that normalize to the SAME executed move must NOT be
      treated as a termination-triggering repeat, since the model did not
      actually repeat itself).

Plain assert-based, same convention as test_output_parser.py (whose
FakeLLMScripted/FakeEnv classes this file's helpers mirror, duplicated
locally to keep this file self-contained).

Run directly:
    python3 alfworld_runs_ae/tests/test_put_move_compat_integration.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "hotpotqa_runs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # alfworld_runs_ae

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


class FakeLLMScripted:
    """Returns one scripted raw completion per call, ignoring prompt
    content. Counts calls so tests can assert normalization adds none."""
    def __init__(self, generations):
        self._gens = list(generations)
        self.seen_prompts = []
        self.last_meta = None
        self.model = "Qwen/Qwen3-8B"  # real model name so count_tokens() can load a tokenizer
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.seen_prompts.append(prompt)
        self.call_count += 1
        gen = self._gens.pop(0) if self._gens else "Action: look"
        self.last_meta = {"finish_reason": "stop", "was_truncated": False, "usage": "unavailable"}
        return gen


class FakeEnv:
    """admissible_commands only ever lists the "move ..." phrasing (mirrors
    the real ALFWorld json_2.1.1 PutObject grammar confirmed in this repo's
    audit_reports/AE_CONTROLLER_AUDIT_2026-07-31.md) -- so a step is only
    reported admissible, and only "succeeds", if the environment actually
    received "move X to Y", never "put X in/on Y"."""
    def __init__(self, n_steps_until_done=999):
        self.n_steps_until_done = n_steps_until_done
        self._step = 0
        self.received_actions = []
        self.step_count = 0

    def reset(self):
        self._step = 0
        ob = "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in a room.\n\nYour task is to: put a mug in the fridge."
        return [ob], {
            "admissible_commands": [["go to fridge 1", "move mug 1 to fridge 1"]],
            "won": [False],
        }

    def step(self, actions):
        self.step_count += 1
        action = actions[0]
        self.received_actions.append(action)
        self._step += 1
        done = self._step >= self.n_steps_until_done
        if action == "move mug 1 to fridge 1":
            ob = "You move the mug 1 to the fridge 1."
        elif action.startswith("put "):
            # The real environment's actual behavior for a literal "put ..."
            # string -- see R2_ROOT_CAUSE_REPORT.md / this repo's own audit:
            # silently a no-op, never recognized.
            ob = "Nothing happens."
        else:
            ob = "Nothing happens."
        return [ob], [1 if done else 0], [done], {
            "admissible_commands": [["go to fridge 1", "move mug 1 to fridge 1"]],
            "won": [done],
        }

    def close(self):
        pass


# ── 11/13/14. env.step() receives executed_action; log carries all fields ─
def test_put_action_reaches_env_as_move_and_is_logged():
    from agents import ALFWorldAgent
    import agents as agents_mod
    agents_mod.MAX_STEPS = 1

    llm = FakeLLMScripted(["Thought: place the mug.\nAction: put mug 1 in fridge 1"])
    env = FakeEnv(n_steps_until_done=999)
    agent = ALFWorldAgent(llm, termination_policy="fixed_horizon")
    agent.run(env, "put a mug in the fridge", "put", to_print=False)

    check("11. env.step() received 'move mug 1 to fridge 1', never the put phrasing",
          env.received_actions == ["move mug 1 to fridge 1"], env.received_actions)

    step = agent.step_log[0]
    check("13. parsed_action keeps the model's own put wording",
          step["parsed_action"] == "put mug 1 in fridge 1", step["parsed_action"])
    check("13b. executed_action is the normalized move wording",
          step["executed_action"] == "move mug 1 to fridge 1", step["executed_action"])
    check("13c. raw_generation still contains the model's full original text",
          "put mug 1 in fridge 1" in step["raw_generation"], step["raw_generation"])
    check("13d. action_normalized is True", step["action_normalized"] is True)
    check("13e. action_normalization records the reason", step["action_normalization"] == "put_to_move")
    check("13f. action_is_admissible reflects the EXECUTED action, not the un-normalized put string",
          step["action_is_admissible"] is True, step["action_is_admissible"])


def test_unconverted_action_parsed_equals_executed():
    from agents import ALFWorldAgent
    import agents as agents_mod
    agents_mod.MAX_STEPS = 1

    llm = FakeLLMScripted(["Thought: look around.\nAction: go to fridge 1"])
    env = FakeEnv(n_steps_until_done=999)
    agent = ALFWorldAgent(llm, termination_policy="fixed_horizon")
    agent.run(env, "put a mug in the fridge", "put", to_print=False)

    step = agent.step_log[0]
    check("14. unconverted action: parsed_action == executed_action",
          step["parsed_action"] == step["executed_action"] == "go to fridge 1", step)
    check("14b. action_normalized is False for an unconverted action", step["action_normalized"] is False)
    check("14c. action_normalization is None for an unconverted action", step["action_normalization"] is None)


# ── 12. react (controller=None) and ae_full (controller attached) share it ─
def test_react_and_ae_full_share_the_same_normalization():
    from agents import ALFWorldAgent
    from ae.controllers.config import AEConfig
    from ae.controllers.stateful_controller import StatefulController
    import agents as agents_mod
    agents_mod.MAX_STEPS = 1

    gens = ["Thought: place the mug.\nAction: put mug 1 in fridge 1"]

    llm_react = FakeLLMScripted(list(gens))
    env_react = FakeEnv()
    agent_react = ALFWorldAgent(llm_react, controller=None, termination_policy="fixed_horizon")
    agent_react.run(env_react, "put a mug in the fridge", "put", to_print=False)

    llm_ae = FakeLLMScripted(list(gens))
    env_ae = FakeEnv()
    controller = StatefulController(AEConfig())
    agent_ae = ALFWorldAgent(llm_ae, controller=controller, termination_policy="fixed_horizon")
    agent_ae.run(env_ae, "put a mug in the fridge", "put", to_print=False)

    check("12. react and ae_full send the identical executed_action to env.step()",
          env_react.received_actions == env_ae.received_actions == ["move mug 1 to fridge 1"],
          (env_react.received_actions, env_ae.received_actions))
    check("12b. react and ae_full log the identical executed_action",
          agent_react.step_log[0]["executed_action"] == agent_ae.step_log[0]["executed_action"])

    # ---- controller's own per-step record: action= must be the EXECUTED
    # action (see agents.py's controller.step(action=executed_action, ...))
    # so invalid_action/unexpected_outcome are judged against what the
    # environment actually received, not the pre-fix put string.
    controller_record = controller.step_log[0]
    check("12c. controller-side step log's 'action' field is the executed (move) action, "
          "not the un-normalized parsed put action",
          controller_record["action"] == "move mug 1 to fridge 1", controller_record["action"])


def test_reflexion_shares_the_same_normalization():
    """reflexion.py's ALFWorldReflectAgent.run_trial() is a thin wrapper
    around the exact same ALFWorldAgent.run() -- confirm the fix applies
    there too via ae/baselines/reflexion.run_episode's real entry point."""
    from ae.baselines import reflexion as reflexion_baseline
    import agents as agents_mod
    agents_mod.MAX_STEPS = 1

    llm = FakeLLMScripted(["Thought: place the mug.\nAction: put mug 1 in fridge 1"])
    env = FakeEnv(n_steps_until_done=1)  # done on first step -> no reflect() call needed
    result = reflexion_baseline.run_episode(
        env, "put a mug in the fridge", "put", act_llm=llm, reflect_llm=FakeLLMScripted([]),
        max_trials=1, to_print=False,
    )
    check("12d. reflexion's env.step() also received the normalized move action",
          env.received_actions == ["move mug 1 to fridge 1"], env.received_actions)
    check("12e. reflexion's step_log carries the same executed_action field",
          result["trial_records"][0]["step_log"][0]["executed_action"] == "move mug 1 to fridge 1")


# ── 15. no extra LLM call / env step; termination rule unaffected ────────
def test_normalization_adds_no_extra_llm_call_or_env_step():
    from agents import ALFWorldAgent
    import agents as agents_mod
    agents_mod.MAX_STEPS = 3

    llm = FakeLLMScripted([
        "Thought: a.\nAction: put mug 1 in fridge 1",
        "Thought: b.\nAction: go to fridge 1",
        "Thought: c.\nAction: put mug 1 on fridge 1",
    ])
    env = FakeEnv(n_steps_until_done=999)
    agent = ALFWorldAgent(llm, termination_policy="fixed_horizon")
    agent.run(env, "put a mug in the fridge", "put", to_print=False)

    check("15. exactly one LLM call per step (3 steps -> 3 calls, normalization added none)",
          llm.call_count == 3, llm.call_count)
    check("15b. exactly one env.step() call per step (3 steps -> 3 env.step() calls)",
          env.step_count == 3, env.step_count)


def test_termination_rule_uses_unnormalized_parsed_action_not_executed():
    """Two DIFFERENT model phrasings ("put mug 1 in fridge 1" then "put mug
    1 on fridge 1") both normalize to the SAME executed_action ("move mug 1
    to fridge 1"), but the model did NOT literally repeat its own output --
    the exact-match repeated-action early-termination check (legacy_early_
    stop) must still compare parsed_action (unnormalized), so this must
    NOT be treated as a repeat and must NOT terminate early."""
    from agents import ALFWorldAgent
    import agents as agents_mod
    agents_mod.MAX_STEPS = 3

    llm = FakeLLMScripted([
        "Thought: a.\nAction: put mug 1 in fridge 1",
        "Thought: b.\nAction: put mug 1 on fridge 1",  # different wording, same normalized target
        "Thought: c.\nAction: go to fridge 1",
    ])
    env = FakeEnv(n_steps_until_done=999)
    agent = ALFWorldAgent(llm, termination_policy="legacy_early_stop")
    agent.run(env, "put a mug in the fridge", "put", to_print=False)

    check("termination rule: two differently-worded put actions "
          "(normalizing to the same executed move) do not trigger exhausted_repeated",
          agent.termination_reason != "exhausted_repeated", agent.termination_reason)
    check("termination rule: all 3 steps actually ran (no early break)",
          len(agent.step_log) == 3, len(agent.step_log))
    check("termination rule: env received both normalized moves, not collapsed into one call",
          env.received_actions == ["move mug 1 to fridge 1", "move mug 1 to fridge 1", "go to fridge 1"],
          env.received_actions)

    # Sanity control: an EXACT repeat of parsed_action (same wording twice)
    # must still trigger the legacy early-termination rule unchanged.
    llm2 = FakeLLMScripted([
        "Thought: a.\nAction: put mug 1 in fridge 1",
        "Thought: b.\nAction: put mug 1 in fridge 1",  # exact repeat of parsed_action
        "Thought: c.\nAction: go to fridge 1",
    ])
    env2 = FakeEnv(n_steps_until_done=999)
    agent2 = ALFWorldAgent(llm2, termination_policy="legacy_early_stop")
    agent2.run(env2, "put a mug in the fridge", "put", to_print=False)
    check("control: an EXACT repeat of parsed_action still triggers exhausted_repeated (rule unchanged)",
          agent2.termination_reason == "exhausted_repeated", agent2.termination_reason)


if __name__ == "__main__":
    test_put_action_reaches_env_as_move_and_is_logged()
    test_unconverted_action_parsed_equals_executed()
    test_react_and_ae_full_share_the_same_normalization()
    test_reflexion_shares_the_same_normalization()
    test_normalization_adds_no_extra_llm_call_or_env_step()
    test_termination_rule_uses_unnormalized_parsed_action_not_executed()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
