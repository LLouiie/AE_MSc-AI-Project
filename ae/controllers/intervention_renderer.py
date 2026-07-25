"""Renders an InterventionType into the short control-directive text shown
to the agent. AE never generates actions itself — it only nudges the
existing ReAct agent's next Thought/Action via this directive block, which
is injected transiently into the prompt for the LLM call and never written
into the persistent trajectory/history (see alfworld_runs_ae/agents.py's
loop for where it's spliced in and dropped)."""

from __future__ import annotations

from ae.core import InterventionType

_DIRECTIVES = {
    InterventionType.VERIFY: (
        "Re-check the current goal and latest observation.\n"
        "Identify what information is still missing.\n"
        "Prefer one information-gathering action before changing the whole strategy."
    ),
    InterventionType.REFLECT: (
        "Diagnose why the recent actions did not change the environment.\n"
        "Do not repeat the same failed action.\n"
        "Choose a corrected action from the current environment state."
    ),
    InterventionType.REPLAN: (
        "Restate the task objective.\n"
        "Discard the current ineffective strategy.\n"
        "Construct a short plan of 2-4 subgoals from the current state.\n"
        "Execute only the next subgoal now."
    ),
}


def render_directive(intervention: InterventionType) -> str:
    body = _DIRECTIVES.get(intervention)
    if body is None:
        return ""
    return f"[ACTIVE CONTROL DIRECTIVE]\n{body}"
