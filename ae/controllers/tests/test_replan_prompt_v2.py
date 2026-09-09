"""Exact Appendix A directive prompt tests."""
from ae.core import InterventionType
from ae.controllers.intervention_renderer import render_directive


def test_verify_exact():
    assert render_directive(InterventionType.VERIFY) == (
        "[ACTIVE CONTROL DIRECTIVE: VERIFY]\n"
        "Re-check the current goal and latest observation.\n"
        "Identify what information is still missing.\n"
        "Prefer one information-gathering action before changing the whole strategy.\n"
        "Respond with exactly two lines:\n"
        "Thought: <state what needs checking and why>\n"
        "Action: <one executable environment command>"
    )


def test_reflect_exact_without_commands():
    assert render_directive(InterventionType.REFLECT) == (
        "[ACTIVE CONTROL DIRECTIVE: REFLECT]\n"
        "Diagnose why the recent actions did not change the environment.\n"
        "Do not repeat the same failed action.\n"
        "Choose a corrected action from the current environment state.\n"
        "Respond with exactly two lines:\n"
        "Thought: <identify the problem and explain the correction>\n"
        "Action: <one executable environment command>"
    )


def test_replan_exact():
    assert render_directive(
        InterventionType.REPLAN,
        steps_since_meaningful_change=3,
        last_nonprogress_action="go to desk 1",
        last_nonprogress_observation="Nothing happens.",
    ) == (
        "[ACTIVE CONTROL DIRECTIVE: REPLAN]\n"
        "No observed state change has occurred for 3 steps.\n"
        "The most recent action without observed progress was go to desk 1.\n"
        "The environment returned: Nothing happens.\n"
        "Revise the ineffective part of the current strategy.\n"
        "Plan a route with 2-4 subgoals from the current observed state, preserving any subgoals already completed.\n"
        "Execute only the next unmet subgoal.\n"
        "Respond with exactly two lines:\n"
        "Thought: <give the revised route as A -> B -> C>\n"
        "Action: <one executable environment command>"
    )


def test_reflect_commands_are_sorted_and_capped():
    commands = [f"go to shelf {i}" for i in range(60)]
    prompt = render_directive(InterventionType.REFLECT, admissible_commands=commands)
    assert "first 40 of 60, alphabetical" in prompt
    assert sum(line.startswith("  go to shelf ") for line in prompt.splitlines()) == 40
    assert prompt.endswith(
        "Respond with exactly two lines:\n"
        "Thought: <identify the problem and explain the correction>\n"
        "Action: <one executable environment command>"
    )
