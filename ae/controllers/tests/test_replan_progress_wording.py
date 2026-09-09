"""REPLAN runtime-field and compatibility tests."""
from ae.core import InterventionType
from ae.controllers.intervention_renderer import render_directive


def test_replan_uses_current_runtime_fields():
    prompt = render_directive(
        InterventionType.REPLAN,
        goal="unused compatibility field",
        steps_since_meaningful_change=7,
        last_nonprogress_action="open fridge 1",
        last_nonprogress_observation="Nothing happens.",
        current_observation="unused compatibility field",
    )
    assert "No observed state change has occurred for 7 steps." in prompt
    assert "The most recent action without observed progress was open fridge 1." in prompt
    assert "The environment returned: Nothing happens." in prompt
    assert "unused compatibility field" not in prompt


def test_replan_zero_is_rendered_without_switching_templates():
    prompt = render_directive(InterventionType.REPLAN, steps_since_meaningful_change=0)
    assert "No observed state change has occurred for 0 steps." in prompt
    assert "Execute only the next unmet subgoal." in prompt
    assert "The latest step produced" not in prompt


def test_admissible_commands_do_not_change_replan():
    kwargs = dict(
        steps_since_meaningful_change=2,
        last_nonprogress_action="look",
        last_nonprogress_observation="OK.",
    )
    assert render_directive(InterventionType.REPLAN, **kwargs) == render_directive(
        InterventionType.REPLAN, admissible_commands=["inventory"], **kwargs
    )
