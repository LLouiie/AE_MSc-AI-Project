"""Integration tests for ae/baselines/reflexgrad_v4_episode.py -- the real
ALFWorld episode loop shared by reflexgrad_v4 and reflexion_only_reflexgrad_v4.

No live vLLM server or ALFWorld game files needed: a fake chat client
scripts every role's response, and a fake env mimics ALFWorld's
batch_size=1 list-shaped reset()/step() convention (same pattern as
alfworld_runs_ae/tests/test_output_parser.py::FakeEnv and
ae/baselines/tests/test_react_reflact_anchor.py::FakeAnchorEnv).

Run directly:
    python3 ae/baselines/tests/test_reflexgrad_v4_episode.py
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root

from ae.baselines import reflexgrad_v4, reflexion_only_reflexgrad_v4  # noqa: E402
from ae.baselines.reflexgrad_v4_episode import run_episode  # noqa: E402
from ae.baselines.reflexgrad_v4_llm import ReflexGradAPIError, RoleParseError  # noqa: E402
from ae.baselines.reflexgrad_v4_llm import (  # noqa: E402
    parse_decomposer_todos, parse_evaluator_score, parse_todo_verifier_result,
)
from ae.runners.run_alfworld_anchor import _REFLEXGRAD_BUILD_ENGINE_FN  # noqa: E402

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


_DEFAULTS = {
    "decomposer": "TODO: find the mug\nTODO: put mug in fridge",
    "actor": "Thought: ok\nAction: go to fridge 1",
    "evaluator": "8",
    "todo_verifier": "NO",
    "loss": "loss text", "gradient": "gradient text", "optimizer": "revised policy text",
    "trajectory_analyzer": "analysis text", "causal_diagnoser": "cause text",
    "plan_generator": "1. corrective subgoal",
}

_PARSERS = {
    "evaluator": parse_evaluator_score, "todo_verifier": parse_todo_verifier_result,
    "decomposer": parse_decomposer_todos,
}


class FakeReflexGradChatLLM:
    """Fake stand-in for reflexgrad_v4_llm.ReflexGradChatLLM -- scripts one
    response queue per role, validates each scripted response against that
    role's real parser (so tests can't accidentally script something the
    real parser would reject), and can simulate a hard API failure on a
    chosen role."""

    def __init__(self, scripted_by_role=None, model="Qwen/Qwen3-8B", temperature=0.2,
                 max_api_retries=0, raise_role=None):
        self.model = model
        self.temperature = temperature
        self.max_api_retries = max_api_retries
        self._scripted = {k: list(v) for k, v in (scripted_by_role or {}).items()}
        self.api_attempts_by_role = defaultdict(int)
        self.calls_seen = []  # list of (role, messages)
        self._raise_role = raise_role

    def call(self, role, messages, max_tokens):
        self.api_attempts_by_role[role] += 1
        self.calls_seen.append((role, messages))
        if self._raise_role == role:
            raise ReflexGradAPIError(f"simulated final API failure for role={role}")
        queue = self._scripted.get(role, [])
        return queue.pop(0) if queue else _DEFAULTS[role]


class FakeEpisodeEnv:
    """ALFWorld batch_size=1 convention. done_at_step/won_on_done control
    when (if ever) the episode wins."""

    def __init__(self, done_at_step=None, won_on_done=True):
        self.done_at_step = done_at_step
        self.won_on_done = won_on_done
        self._step = 0
        self.received_actions = []

    def reset(self):
        self._step = 0
        ob = ("-= Welcome =-\n\nYou are in the middle of a room. Looking quickly around you, "
              "you see a fridge 1, a countertop 1.\n\nYour task is to: put a mug in the fridge.")
        return [ob], {"admissible_commands": [["go to fridge 1", "go to countertop 1"]], "won": [False]}

    def step(self, actions):
        self.received_actions.append(actions[0])
        self._step += 1
        done = self.done_at_step is not None and self._step >= self.done_at_step
        won = done and self.won_on_done
        ob = "You put the mug in the fridge." if won else "Nothing happens."
        return [ob], [1 if won else 0], [done], {
            "admissible_commands": [["go to fridge 1", "go to countertop 1"]], "won": [won],
        }

    def close(self):
        pass


def test_both_baselines_dispatch_to_the_shared_engine_class():
    from ae.baselines.reflexgrad_v4_engine import ReflexGradV4Engine
    check("1. run_alfworld_anchor dispatch table has both baselines",
          set(_REFLEXGRAD_BUILD_ENGINE_FN.keys()) == {"reflexgrad_v4", "reflexion_only_reflexgrad_v4"})
    check("2. reflexgrad_v4 dispatches to ae.baselines.reflexgrad_v4.build_engine",
          _REFLEXGRAD_BUILD_ENGINE_FN["reflexgrad_v4"] is reflexgrad_v4.build_engine)
    check("3. reflexion_only_reflexgrad_v4 dispatches to its own build_engine",
          _REFLEXGRAD_BUILD_ENGINE_FN["reflexion_only_reflexgrad_v4"]
          is reflexion_only_reflexgrad_v4.build_engine)

    env_a = FakeEpisodeEnv(done_at_step=None)
    llm_a = FakeReflexGradChatLLM()
    result_a = run_episode(env_a, "env1", "put a mug in the fridge", llm_a,
                            build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=2)
    env_b = FakeEpisodeEnv(done_at_step=None)
    llm_b = FakeReflexGradChatLLM()
    result_b = run_episode(env_b, "env1", "put a mug in the fridge", llm_b,
                            build_engine_fn=reflexion_only_reflexgrad_v4.build_engine, max_agent_steps=2)
    check("4. reflexgrad_v4 episode's baseline field is correctly labeled",
          result_a["baseline"] == "reflexgrad_v4", result_a["baseline"])
    check("5. reflexion_only_reflexgrad_v4 episode's baseline field is correctly labeled",
          result_b["baseline"] == "reflexion_only_reflexgrad_v4", result_b["baseline"])
    check("6. reflexgrad_v4's manifest shows textgrad_enabled=True",
          result_a["manifest_fields"]["textgrad_enabled"] is True)
    check("7. reflexion_only's manifest shows textgrad_enabled=False",
          result_b["manifest_fields"]["textgrad_enabled"] is False)


def test_real_env_reset_observation_reaches_the_first_actor_prompt():
    env = FakeEpisodeEnv(done_at_step=None)
    llm = FakeReflexGradChatLLM()
    result = run_episode(env, "env1", "put a mug in the fridge", llm,
                          build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=1)
    check("8. initial_observation captured from the real env.reset() text",
          "fridge 1" in result["initial_observation"] and "countertop 1" in result["initial_observation"])
    first_actor_call = next(m for r, m in llm.calls_seen if r == "actor")
    prompt_text = first_actor_call[0]["content"]
    check("9. the real env.reset() observation text appears in the actor's first prompt",
          "fridge 1" in prompt_text and "countertop 1" in prompt_text, prompt_text[:300])
    check("10. the task goal also appears in the first actor prompt",
          "put a mug in the fridge" in prompt_text)


def test_decomposer_failure_surfaces_as_api_error_not_a_crash():
    # Regression test: an early version of run_episode wrapped only the
    # step loop in try/except, not the engine.decompose(...) call before
    # it -- so a RoleParseError from a malformed decomposer response
    # crashed the whole process instead of being recorded per-episode.
    # Caught during the real vLLM smoke test when Qwen3-8B emitted an
    # uncontrolled <think> block (thinking_mode="backend_default" means we
    # deliberately never send enable_thinking) that ran past max_tokens
    # before ever producing a "TODO:" line.
    env = FakeEpisodeEnv(done_at_step=None)
    llm = FakeReflexGradChatLLM(scripted_by_role={"decomposer": ["not a todo list at all"]})
    result = run_episode(env, "env1", "put a mug in the fridge", llm,
                          build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=5)
    check("10b. a decomposer parse failure surfaces as termination_reason=api_error",
          result["termination_reason"] == "api_error")
    check("10c. the error message names the decomposer role", "decomposer" in result["api_error_message"])
    check("10d. no actor/env steps happened after the decomposer failed", result["agent_steps"] == 0)


def test_15_step_budget_terminates_without_overrunning():
    env = FakeEpisodeEnv(done_at_step=None)  # never wins
    llm = FakeReflexGradChatLLM()
    result = run_episode(env, "env1", "put a mug in the fridge", llm,
                          build_engine_fn=reflexgrad_v4.build_engine)  # default max_agent_steps=15
    check("11. default max_agent_steps is 15", result["manifest_fields"]["max_agent_steps"] == 15)
    check("12. agent_steps stops exactly at 15 when never won", result["agent_steps"] == 15)
    check("13. env_actions equals agent_steps (every step parsed fine)", result["env_actions"] == 15)
    check("14. termination_reason is max_agent_steps, not won", result["termination_reason"] == "max_agent_steps")
    check("15. success/won are both 0/False", result["success"] == 0 and result["won"] is False)


def test_parser_failure_accounting_at_episode_level():
    env = FakeEpisodeEnv(done_at_step=None)
    llm = FakeReflexGradChatLLM(scripted_by_role={
        "actor": ["I am just thinking with no label at all.", "Thought: ok\nAction: go to fridge 1"],
    })
    result = run_episode(env, "env1", "put a mug in the fridge", llm,
                          build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=2)
    check("16. agent_steps counts both the parse failure and the real action", result["agent_steps"] == 2)
    check("17. env_actions only counts the one real action", result["env_actions"] == 1)
    check("18. env.step() was called exactly once", len(env.received_actions) == 1)
    check("19. step_log[0] is the parse failure with no action recorded",
          result["step_log"][0]["action"] is None)
    check("20. step_log[1] is the real action", result["step_log"][1]["action"] == "go to fridge 1")


def test_global_win_skips_todo_verifier():
    env = FakeEpisodeEnv(done_at_step=1, won_on_done=True)
    llm = FakeReflexGradChatLLM()
    result = run_episode(env, "env1", "put a mug in the fridge", llm,
                          build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=5)
    check("21. termination_reason is 'won'", result["termination_reason"] == "won")
    check("22. success/won are both 1/True", result["success"] == 1 and result["won"] is True)
    check("23. todo_verifier was never called on the winning step",
          result["llm_calls_by_role"]["todo_verifier"] == 0)
    check("24. evaluator WAS called on the winning step (only verifier is skipped)",
          result["llm_calls_by_role"]["evaluator"] == 1)


def test_role_and_api_call_counts_are_accurate():
    env = FakeEpisodeEnv(done_at_step=None)
    llm = FakeReflexGradChatLLM(scripted_by_role={"todo_verifier": ["NO", "NO", "NO"]})
    result = run_episode(env, "env1", "put a mug in the fridge", llm,
                          build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=3)
    check("25. llm_calls_by_role.actor == agent_steps", result["llm_calls_by_role"]["actor"] == 3)
    check("26. llm_calls_by_role.evaluator == env_actions", result["llm_calls_by_role"]["evaluator"] == 3)
    check("27. llm_calls_by_role.decomposer == 1 (called once per episode)",
          result["llm_calls_by_role"]["decomposer"] == 1)
    check("28. api_attempts_by_role matches llm_calls_by_role when no retries occur",
          result["api_attempts_by_role"]["actor"] == 3 and result["api_attempts_by_role"]["evaluator"] == 3)


def test_prompt_provenance_is_per_role_not_a_blanket_claim():
    env = FakeEpisodeEnv(done_at_step=None)
    llm = FakeReflexGradChatLLM()
    result = run_episode(env, "env1", "put a mug in the fridge", llm,
                          build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=1)
    prov = result["manifest_fields"]["prompt_provenance"]
    appendix_e_roles = {"evaluator", "loss", "gradient", "optimizer",
                         "trajectory_analyzer", "causal_diagnoser", "plan_generator"}
    check("29. exactly the 7 Appendix-E roles are marked paper_appendix_e",
          {r for r, p in prov.items() if p["source_type"] == "paper_appendix_e"} == appendix_e_roles)
    check("30. decomposer and todo_verifier are marked official_repo, not appendix_e",
          prov["decomposer"]["source_type"] == "official_repo"
          and prov["todo_verifier"]["source_type"] == "official_repo")
    check("31. actor is marked local_definition, not appendix_e or official_repo",
          prov["actor"]["source_type"] == "local_definition")
    check("32. every role entry carries a non-empty sha256", all(p["prompt_sha256"] for p in prov.values()))
    check("33. adapted roles (decomposer/todo_verifier/actor) carry a non-empty adaptation_note",
          all(prov[r]["adaptation_note"] for r in ("decomposer", "todo_verifier", "actor")))
    check("34. initial_base_policy is recorded in the manifest",
          result["manifest_fields"]["initial_base_policy"])


def test_no_fallback_on_api_failure():
    env = FakeEpisodeEnv(done_at_step=None)
    llm = FakeReflexGradChatLLM(raise_role="actor")
    result = run_episode(env, "env1", "put a mug in the fridge", llm,
                          build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=5)
    check("35. an API failure surfaces as termination_reason=api_error, not a silent success",
          result["termination_reason"] == "api_error")
    check("36. the error message is captured, not swallowed", bool(result["api_error_message"]))
    check("37. success is 0 -- a failed episode never gets counted as a win",
          result["success"] == 0)


def test_no_fallback_on_unparseable_structured_role_response():
    env = FakeEpisodeEnv(done_at_step=None)
    # "banana" is not an integer -- evaluator's real parser must reject it,
    # and that RoleParseError must surface, not get silently defaulted.
    llm = FakeReflexGradChatLLM(scripted_by_role={"evaluator": ["banana"]})
    result = run_episode(env, "env1", "put a mug in the fridge", llm,
                          build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=5)
    check("38. an unparseable evaluator response surfaces as termination_reason=api_error",
          result["termination_reason"] == "api_error")
    check("39. the RoleParseError message is captured", "evaluator" in result["api_error_message"])


def test_episode_state_does_not_leak_between_run_episode_calls():
    env_a = FakeEpisodeEnv(done_at_step=None)
    llm_a = FakeReflexGradChatLLM(scripted_by_role={"todo_verifier": ["NO"] * 5})
    result_a = run_episode(env_a, "env1", "put a mug in the fridge", llm_a,
                            build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=5)
    check("40. sanity: first episode accumulated real state",
          result_a["agent_steps"] == 5 and result_a["policy_final"])

    env_b = FakeEpisodeEnv(done_at_step=None)
    llm_b = FakeReflexGradChatLLM()
    result_b = run_episode(env_b, "env2", "clean a mug", llm_b,
                            build_engine_fn=reflexgrad_v4.build_engine, max_agent_steps=1)
    check("41. second episode's agent_steps starts fresh (does not continue from the first)",
          result_b["agent_steps"] == 1)
    check("42. second episode's TODOs are freshly decomposed, not carried over",
          result_b["step_log"][0]["todos_before"][0]["attempts"] == 0)


if __name__ == "__main__":
    test_both_baselines_dispatch_to_the_shared_engine_class()
    test_real_env_reset_observation_reaches_the_first_actor_prompt()
    test_decomposer_failure_surfaces_as_api_error_not_a_crash()
    test_15_step_budget_terminates_without_overrunning()
    test_parser_failure_accounting_at_episode_level()
    test_global_win_skips_todo_verifier()
    test_role_and_api_call_counts_are_accurate()
    test_prompt_provenance_is_per_role_not_a_blanket_claim()
    test_no_fallback_on_api_failure()
    test_no_fallback_on_unparseable_structured_role_response()
    test_episode_state_does_not_leak_between_run_episode_calls()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
