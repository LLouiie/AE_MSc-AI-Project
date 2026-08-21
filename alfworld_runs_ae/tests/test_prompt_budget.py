"""Tests for the shared 40960-token context budget / deterministic
truncation added 2026-07-27 to fix the Reflexion x ALFWorld exam context
overflow (see agents.py's MAX_MODEL_LEN docstring for the full root-cause
writeup). Covers: the actual root-cause fix (reflection memory must store
only the reflection text, not trajectory+reflection), the two truncation
helpers (_fit_action_prompt / _fit_reflect_prompt), and the no-sys.exit /
retry-once-then-raise behavior on a real context-length error.

Plain assert-based, same convention as the other test files in this repo.
Run directly:
    python3 alfworld_runs_ae/tests/test_prompt_budget.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "hotpotqa_runs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # alfworld_runs_ae

import agents  # noqa: E402
from agents import (  # noqa: E402
    ALFWorldAgent, ALFWorldReflectAgent, PromptBudgetExceededError,
    _fit_action_prompt, _fit_reflect_prompt, count_tokens,
)

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


MODEL = "Qwen/Qwen3-8B"


def _long_history(n_steps, filler_words_per_obs=40):
    """A synthetic (thought, action, observation) history long enough that
    n_steps of it comfortably exceeds a small artificial token budget."""
    filler = " ".join(["item"] * filler_words_per_obs)
    return [
        (f"thought about step {i}", f"go to shelf {i}", f"You see {filler} on shelf {i}.")
        for i in range(n_steps)
    ]


# ── 1. _fit_action_prompt drops oldest reflections before touching history ─
def test_fit_action_prompt_drops_oldest_reflections_first():
    goal = "put a mug in the sink"
    # Each "reflection" here mimics the FIXED format (short text only, not
    # a full trajectory) -- three short reflections plus a short history
    # fits comfortably in a normal budget, but we use an artificially
    # small budget to force the drop-oldest-reflection path without
    # needing a real trajectory-sized blob.
    reflections = [f"Reflection trial {i}: I should check the cabinet next time." for i in range(3)]
    history = _long_history(2, filler_words_per_obs=5)
    full_text, _ = _fit_action_prompt(
        "put", goal, reflections, "", "initial observation here", history,
        MODEL, max_new_tokens=50, task_id="t1", extra_margin=0,
    )
    tokens_full = count_tokens(full_text, MODEL)
    # Budget tight enough that not all 3 reflections + history can fit,
    # but loose enough that dropping reflections alone (not history) saves it.
    tight_budget_new_tokens = agents.MAX_MODEL_LEN - agents.PROMPT_SAFETY_MARGIN - (tokens_full - 30)
    text, info = _fit_action_prompt(
        "put", goal, reflections, "", "initial observation here", history,
        MODEL, max_new_tokens=tight_budget_new_tokens, task_id="t1", extra_margin=0,
    )
    check("1. tightened budget actually forced some truncation",
          info["tokens_after"] < info["tokens_before"], info)
    check("1b. dropped category mentions reflections, not history steps",
          any("reflection" in d for d in info["dropped"])
          and not any("history step" in d for d in info["dropped"]), info["dropped"])
    check("1c. the newest reflection (trial 2) is still present in the fitted prompt",
          "trial 2" in text, text[-400:])


# ── 2. still over budget with zero reflections -> drop oldest history ──────
def test_fit_action_prompt_falls_back_to_dropping_oldest_history():
    goal = "put a mug in the sink"
    history = _long_history(30, filler_words_per_obs=30)
    full_text, _ = _fit_action_prompt(
        "put", goal, [], "", "initial observation here", history,
        MODEL, max_new_tokens=50, task_id="t2", extra_margin=0,
    )
    tokens_full = count_tokens(full_text, MODEL)
    # Budget far too small for the whole history -> must drop oldest steps.
    forced_new_tokens = agents.MAX_MODEL_LEN - agents.PROMPT_SAFETY_MARGIN - (tokens_full // 2)
    text, info = _fit_action_prompt(
        "put", goal, [], "", "initial observation here", history,
        MODEL, max_new_tokens=forced_new_tokens, task_id="t2", extra_margin=0,
    )
    check("2. history truncation actually reduced token count",
          info["tokens_after"] < info["tokens_before"], info)
    check("2b. dropped category mentions history steps",
          any("history step" in d for d in info["dropped"]), info["dropped"])
    check("2c. the most recent step (shelf 29) is kept", "shelf 29" in text)
    check("2d. the initial observation is kept", "initial observation here" in text)
    check("2e. the oldest step (shelf 0) was dropped", "shelf 0" not in text)


# ── 3. _fit_reflect_prompt: same drop-oldest-history priority ───────────────
def test_fit_reflect_prompt_drops_oldest_history_keeps_recent_and_initial_ob():
    goal = "put a mug in the sink"
    history = _long_history(30, filler_words_per_obs=30)
    full_prompt, _ = _fit_reflect_prompt(
        goal, "put", "initial observation here", history, MODEL,
        budget=agents.MAX_MODEL_LEN, task_id="t3",
    )
    tokens_full = count_tokens(full_prompt, MODEL)
    text, info = _fit_reflect_prompt(
        goal, "put", "initial observation here", history, MODEL,
        budget=tokens_full // 2, task_id="t3",
    )
    check("3. reflect prompt truncation reduced token count",
          info["tokens_after"] < info["tokens_before"], info)
    check("3b. most recent history step kept", "shelf 29" in text)
    check("3c. initial observation kept", "initial observation here" in text)
    check("3d. oldest history step dropped", "shelf 0" not in text)


# ── 4. root-cause fix: reflection memory stores ONLY the reflection text ───
class FakeActLLM:
    def __init__(self, actions):
        self._actions = list(actions)
        self.model = MODEL
        self.last_meta = None

    def __call__(self, prompt):
        a = self._actions.pop(0) if self._actions else "go to shelf 1"
        self.last_meta = {"finish_reason": "stop", "was_truncated": False, "usage": "unavailable"}
        return f"Thought: t\nAction: {a}"


class FakeReflectLLM:
    def __init__(self, reflection_text):
        self._reflection_text = reflection_text
        self.model = MODEL
        self.last_meta = None
        self.seen_prompts = []

    def __call__(self, prompt):
        self.seen_prompts.append(prompt)
        return self._reflection_text


class FakeEnvAlwaysFails:
    def reset(self):
        ob = "-= Welcome =-\n\nYou are in a room.\n\nYour task is to: put a mug in the sink."
        return [ob], {"admissible_commands": [["go to shelf 1"]], "won": [False]}

    def step(self, actions):
        return (["Nothing happens."], [0], [False],
                {"admissible_commands": [["go to shelf 1"]], "won": [False]})

    def close(self):
        pass


def test_reflection_memory_stores_only_reflection_text_not_full_trajectory():
    """Regression test for the actual root cause of the context overflow:
    self.reflections must contain only the short reflection text (matching
    the original alfworld_runs/generate_reflections.py::update_memory),
    never `last_trajectory + reflection` (an entire ~50-step trial)."""
    agents_mod = agents
    agents_mod.MAX_STEPS = 3
    act_llm = FakeActLLM(["go to shelf 1", "go to shelf 2", "go to shelf 3"] * 3)
    reflect_llm = FakeReflectLLM("I should check the cabinet next time instead.")
    agent = ALFWorldReflectAgent(act_llm=act_llm, reflect_llm=reflect_llm,
                                  termination_policy="fixed_horizon")
    agent.run_trial(FakeEnvAlwaysFails(), "put a mug in the sink", "put", to_print=False)
    agent.reflect("put a mug in the sink", "put")

    check("4. exactly one reflection was stored", len(agent.reflections) == 1, agent.reflections)
    check("4b. the stored reflection is ONLY the reflection text",
          agent.reflections[0] == "I should check the cabinet next time instead.",
          agent.reflections[0])
    check("4c. the stored reflection does NOT contain the trajectory's own "
          "Action:/Observation: replay lines (would indicate the old bug)",
          "Action: go to shelf" not in agent.reflections[0]
          and "Observation:" not in agent.reflections[0])
    agents_mod.MAX_STEPS = 50


# ── 5. context-length error: retry once, then raise (never sys.exit) ───────
class FakeLLMAlwaysOverBudget:
    def __init__(self):
        self.model = MODEL
        self.last_meta = None
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        import httpx
        from openai import BadRequestError
        resp = httpx.Response(400, request=httpx.Request("POST", "http://fake/v1/completions"))
        raise BadRequestError(
            "This model's maximum context length is 40960 tokens. However, you requested "
            "256 output tokens and your prompt contains at least 99999 input tokens.",
            response=resp, body=None,
        )


def test_context_length_error_retries_once_then_raises_custom_error():
    llm = FakeLLMAlwaysOverBudget()
    agent = ALFWorldAgent(llm, termination_policy="fixed_horizon")
    raised = None
    try:
        agent.run(FakeEnvAlwaysFails(), "put a mug in the sink", "put", to_print=False)
    except PromptBudgetExceededError as e:
        raised = e
    check("5. a context-length error raises PromptBudgetExceededError, not a bare crash",
          raised is not None)
    check("5b. exactly one retry happened (2 total LLM calls), not an infinite loop",
          llm.calls == 2, llm.calls)
    check("5c. the raised error carries before/after token counts for logging",
          raised is not None and raised.tokens_before is not None and raised.tokens_after is not None)


if __name__ == "__main__":
    test_fit_action_prompt_drops_oldest_reflections_first()
    test_fit_action_prompt_falls_back_to_dropping_oldest_history()
    test_fit_reflect_prompt_drops_oldest_history_keeps_recent_and_initial_ob()
    test_reflection_memory_stores_only_reflection_text_not_full_trajectory()
    test_context_length_error_retries_once_then_raises_custom_error()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
