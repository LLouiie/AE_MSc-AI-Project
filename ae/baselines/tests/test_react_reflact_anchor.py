"""Tests for ae/baselines/react_reflact_anchor.py (published-anchor ReAct).

Plain assert-based, same convention as alfworld_runs_ae/tests/test_output_parser.py.
Run directly:
    python3 ae/baselines/tests/test_react_reflact_anchor.py

Where a *real* (non-mocked) check is possible without a live vLLM server --
confirming the actual Qwen3-8B chat template really closes the <think> block
when enable_thinking=False -- this file uses the real local tokenizer cache
(HF_HOME) rather than mocking that specific behavior away. Everything that
requires an actual network call to the model server is mocked via a fake
OpenAI-shaped client and clearly labeled as such.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root

from ae.baselines.react_reflact_anchor import (  # noqa: E402
    MAX_AGENT_STEPS, NativeThinkBlockError, QwenChatAnchorLLM, run_episode,
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


# ── fakes: OpenAI-shaped chat client (no network) ────────────────────────
class _FakeUsage:
    def __init__(self, p=10, c=5, t=15):
        self.prompt_tokens, self.completion_tokens, self.total_tokens = p, c, t


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = types.SimpleNamespace(content=content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_FakeChoice(content, finish_reason)]
        self.usage = _FakeUsage()


class _FakeChatCompletions:
    def __init__(self, scripted_contents, default="Action: look"):
        self._contents = list(scripted_contents)
        self._default = default
        self.seen_calls = []

    def create(self, **kwargs):
        self.seen_calls.append(kwargs)
        content = self._contents.pop(0) if self._contents else self._default
        return _FakeResponse(content)


class FakeOpenAIClient:
    def __init__(self, scripted_contents, default="Action: look"):
        self.chat = types.SimpleNamespace(
            completions=_FakeChatCompletions(scripted_contents, default)
        )


# ── fake ALFWorld env (batch_size=1 list-shaped returns, same convention
#    as alfworld_runs_ae/tests/test_output_parser.py::FakeEnv) ────────────
class FakeAnchorEnv:
    def __init__(self, done_at_step=None, won_on_done=True):
        self.done_at_step = done_at_step
        self.won_on_done = won_on_done
        self._step = 0
        self.received_actions = []

    def reset(self):
        self._step = 0
        ob = "-= Welcome =-\n\nYou are in a room.\n\nYour task is to: put a mug in the sink."
        return [ob], {"admissible_commands": [["go to sink 1", "go to cabinet 1"]], "won": [False]}

    def step(self, actions):
        self.received_actions.append(actions[0])
        self._step += 1
        done = self.done_at_step is not None and self._step >= self.done_at_step
        won = done and self.won_on_done
        ob = "You put the mug in the sink." if won else "Nothing happens."
        return [ob], [1 if won else 0], [done], {
            "admissible_commands": [["go to sink 1", "go to cabinet 1"]], "won": [won],
        }

    def close(self):
        pass


def test_constants_pin_production_defaults():
    check("1. MAX_AGENT_STEPS default is 30 (published-anchor budget)", MAX_AGENT_STEPS == 30)


def test_chat_client_requests_thinking_off_via_extra_body():
    fake_client = FakeOpenAIClient(["Action: look"])
    llm = QwenChatAnchorLLM(model="Qwen/Qwen3-8B", base_url="unused", client=fake_client)
    llm([{"role": "user", "content": "hi"}])
    call = fake_client.chat.completions.seen_calls[0]
    check("2. request used chat.completions (not raw completions)", "messages" in call)
    check("3. request set chat_template_kwargs.enable_thinking=False via extra_body",
          call.get("extra_body") == {"chat_template_kwargs": {"enable_thinking": False}},
          detail=repr(call.get("extra_body")))


def test_native_think_block_raises_not_silently_swallowed():
    fake_client = FakeOpenAIClient(["<think>reasoning here</think>\nAction: look"])
    llm = QwenChatAnchorLLM(model="Qwen/Qwen3-8B", base_url="unused", client=fake_client)
    raised = False
    try:
        llm([{"role": "user", "content": "hi"}])
    except NativeThinkBlockError:
        raised = True
    check("4. a completion containing native <think> raises NativeThinkBlockError", raised)


def test_real_qwen3_tokenizer_confirms_thinking_off_closes_block():
    """Not mocked: loads the actual Qwen/Qwen3-8B tokenizer (same HF_HOME
    cache agents.py::_get_tokenizer already uses for token counting) and
    confirms applying its real chat template with enable_thinking=False
    inserts the closed <think>\\n\\n</think>\\n\\n stub immediately after the
    assistant tag -- the actual mechanism react_reflact_anchor relies on,
    verified independently of whether the vLLM server is reachable right now."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    except Exception as e:  # pragma: no cover -- environment without the local HF cache
        check("5. real Qwen3-8B tokenizer available locally (SKIPPED if not)", True,
              f"skipped: {e}")
        return
    msgs = [{"role": "user", "content": "hello"}]
    rendered_off = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
    rendered_default = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    check("5. enable_thinking=False renders a pre-closed <think></think> stub",
          rendered_off.rstrip().endswith("<think>\n\n</think>"), repr(rendered_off[-60:]))
    check("6. default (thinking on) rendering does NOT pre-close <think> (contrast case)",
          not rendered_default.rstrip().endswith("<think>\n\n</think>"), repr(rendered_default[-60:]))


def test_parser_failure_increments_agent_steps_not_env_actions():
    fake_client = FakeOpenAIClient([
        "I am just thinking out loud with no label at all.",
        "Action: go to cabinet 1",
    ])
    llm = QwenChatAnchorLLM(model="Qwen/Qwen3-8B", base_url="unused", client=fake_client)
    env = FakeAnchorEnv(done_at_step=None)
    result = run_episode(env, "pick_and_place_simple-Mug-None-Sink-1", llm,
                          task_id="t1", max_agent_steps=2)
    check("7. agent_steps counts both the parse failure and the real action", result["agent_steps"] == 2)
    check("8. env_actions only counts the one real env.step() call", result["env_actions"] == 1)
    check("9. step_log[0] recorded as a parse failure", result["step_log"][0]["parse_success"] is False)
    check("10. step_log[1] recorded as a parse success", result["step_log"][1]["parse_success"] is True)
    check("11. env.step() was called exactly once (matches env_actions)", len(env.received_actions) == 1)


def test_success_only_from_env_won_true_and_false():
    # done=True but won=False -> success must be 0, not inferred from "done".
    fake_client_a = FakeOpenAIClient(["Action: go to sink 1"] * 3)
    llm_a = QwenChatAnchorLLM(model="Qwen/Qwen3-8B", base_url="unused", client=fake_client_a)
    env_a = FakeAnchorEnv(done_at_step=1, won_on_done=False)
    result_a = run_episode(env_a, "pick_and_place_simple-Mug-None-Sink-1", llm_a,
                            max_agent_steps=5)
    check("12. env done=True, won=False -> success=0 (not inferred from done alone)",
          result_a["success"] == 0 and result_a["termination_reason"] == "env_done_without_success")

    fake_client_b = FakeOpenAIClient(["Action: go to sink 1"] * 3)
    llm_b = QwenChatAnchorLLM(model="Qwen/Qwen3-8B", base_url="unused", client=fake_client_b)
    env_b = FakeAnchorEnv(done_at_step=1, won_on_done=True)
    result_b = run_episode(env_b, "pick_and_place_simple-Mug-None-Sink-1", llm_b,
                            max_agent_steps=5)
    check("13. env done=True, won=True -> success=1", result_b["success"] == 1
          and result_b["termination_reason"] == "success")


def test_agent_step_budget_terminates_exactly_and_does_not_overrun():
    fake_client = FakeOpenAIClient([], default="Action: go to sink 1")  # never done
    llm = QwenChatAnchorLLM(model="Qwen/Qwen3-8B", base_url="unused", client=fake_client)
    env = FakeAnchorEnv(done_at_step=None)
    result = run_episode(env, "pick_and_place_simple-Mug-None-Sink-1", llm, max_agent_steps=5)
    check("14. agent_steps stops exactly at the budget when never done", result["agent_steps"] == 5)
    check("15. env_actions equals agent_steps here (every step parsed fine)", result["env_actions"] == 5)
    check("16. termination_reason is max_agent_steps, not success", result["termination_reason"] == "max_agent_steps")
    check("17. success is 0 (never won)", result["success"] == 0)


def test_run_episode_selects_demo_matching_env_name_category():
    fake_client = FakeOpenAIClient(["Action: go to sink 1"])
    llm = QwenChatAnchorLLM(model="Qwen/Qwen3-8B", base_url="unused", client=fake_client)
    env = FakeAnchorEnv(done_at_step=1, won_on_done=True)
    result = run_episode(env, "pick_and_place_simple-Mug-None-Sink-1", llm, max_agent_steps=1)
    check("18. mpo_category resolved from env_name prefix", result["mpo_category"] == "pick_and_place")
    check("19. first_turn_messages has exactly one message before the model's first reply",
          len(result["first_turn_messages"]) == 1 and result["first_turn_messages"][0]["role"] == "user")


def test_manifest_fields_present_and_correctly_labeled():
    fake_client = FakeOpenAIClient(["Action: go to sink 1"])
    llm = QwenChatAnchorLLM(model="Qwen/Qwen3-8B", base_url="unused", client=fake_client)
    env = FakeAnchorEnv(done_at_step=1, won_on_done=True)
    result = run_episode(env, "pick_and_place_simple-Mug-None-Sink-1", llm, max_agent_steps=1)
    mf = result["manifest_fields"]
    check("20. protocol_kind=published_anchor", mf["protocol_kind"] == "published_anchor")
    check("21. api_mode=chat_completions", mf["api_mode"] == "chat_completions")
    check("22. thinking_control_method names the actual mechanism used",
          mf["thinking_control_method"] == "chat_template_kwargs.enable_thinking=false")
    check("23. mpo_source_commit is a pinned 40-char SHA, not 'main'",
          len(mf["mpo_source_commit"]) == 40 and mf["mpo_source_commit"] != "main")
    check("24. llm_calls_by_role has exactly one role: actor",
          result["llm_calls_by_role"] == {"actor": 1})


if __name__ == "__main__":
    test_constants_pin_production_defaults()
    test_chat_client_requests_thinking_off_via_extra_body()
    test_native_think_block_raises_not_silently_swallowed()
    test_real_qwen3_tokenizer_confirms_thinking_off_closes_block()
    test_parser_failure_increments_agent_steps_not_env_actions()
    test_success_only_from_env_won_true_and_false()
    test_agent_step_budget_terminates_exactly_and_does_not_overrun()
    test_run_episode_selects_demo_matching_env_name_category()
    test_manifest_fields_present_and_correctly_labeled()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
