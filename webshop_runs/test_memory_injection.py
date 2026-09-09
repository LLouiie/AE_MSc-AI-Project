from unittest.mock import patch

from webshop_trial import webshop_run


class FakeEnv:
    def step(self, _idx, action):
        if action == "reset":
            return "Instruction: buy the matching item\n[Search]", 0.0, False
        return "OK.", 0.0, False


def test_reflexion_memory_reaches_actor_prompt():
    prompts = []

    def fake_llm(prompt, stop):
        prompts.append(prompt)
        return "think[test]"

    with patch("webshop_trial.llm", side_effect=fake_llm):
        webshop_run(
            "fixed_0",
            FakeEnv(),
            "BASE PROMPT",
            ["old one", "old two", "use a specific search"],
            to_print=False,
        )

    assert prompts
    assert "Your memory for the task below:" in prompts[0]
    assert "use a specific search" in prompts[0]
    assert "Instruction: buy the matching item" in prompts[0]


def test_react_prompt_is_unchanged_without_memory():
    prompts = []

    def fake_llm(prompt, stop):
        prompts.append(prompt)
        return "think[test]"

    with patch("webshop_trial.llm", side_effect=fake_llm):
        webshop_run("fixed_0", FakeEnv(), "BASE PROMPT", [], to_print=False)

    assert prompts
