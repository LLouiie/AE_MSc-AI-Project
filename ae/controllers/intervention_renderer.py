"""Render the Appendix A AE directives used by the ReAct agent.

The directive is transiently inserted into the next model prompt and is not
written to trajectory history. REFLECT may additionally show the current
environment-provided admissible commands, sorted and capped at 40.
"""

from __future__ import annotations

from typing import Optional

from ae.core import InterventionType

_STATIC_DIRECTIVES = {
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
}

_ALFWORLD_OUTPUT = (
    "Respond with exactly two lines:\n"
    "Thought: <state what needs checking and why>\n"
    "Action: <one executable environment command>"
)

_ALFWORLD_REFLECT_OUTPUT = (
    "Respond with exactly two lines:\n"
    "Thought: <identify the problem and explain the correction>\n"
    "Action: <one executable environment command>"
)

# REFLECT admissible-action grounding: cap on how many commands to list.
# Real dev/practice environments expose ~15-40 admissible commands per step
# (measured: 21 at reset in a look_at_obj_in_light room). The cap is a
# prompt-budget guard for pathological rooms only; when it trips the
# directive says so explicitly rather than silently showing a truncated
# list the model might read as exhaustive.
_MAX_LISTED_ADMISSIBLE = 40


def _render_reflect(*, admissible_commands) -> str:
    """REFLECT prompt-v3: the static v1 diagnosis lines, plus -- when the
    caller supplied them -- the environment's own admissible-command list
    for the next step.

    Why this and not more static advice: REFLECT-v2 tried a
    static line telling the model what "Nothing happens." usually means.
    It failed, because the correct recovery is task-type-dependent and
    sometimes semantically opposite (examine: stop re-navigating and
    interact directly; heat: the object must be HELD, so take it back out
    of the microwave first), so any single prescription is wrong somewhere.
    Listing what the environment will actually accept sidesteps the whole
    problem: it prescribes no recovery action at all, states no claim that
    can be wrong for a task type, and lets the model pick.

    Evidence this targets the dominant failure mode (this session's
    practice-log case study, ae_full_nopir_v2_5_oldctrl_replanpromptv2_
    oneshot0_seed42_a40_practice): all 42/42 practice failures contain at
    least one inadmissible action, 591 in total, versus a median successful
    episode length of only 10.5 steps -- i.e. failing episodes spend most
    of their budget on actions the environment silently ignores. Worked
    example (look_at_obj_in_light-Bowl-None-DeskLamp-308): at step 17 the
    model was holding the bowl AND standing at desk 1 with the desklamp on
    it -- one `use desklamp 1` from success -- but had just tried
    `use desklamp 1 on bowl 2` and `turn on desklamp 1` (both inadmissible,
    both "Nothing happens."), walked away, and burned the remaining 33
    steps alternating between two desks. `use desklamp 1` was in the
    admissible list at that exact step.

    This is grounding, not auto-correction: the list is injected into the
    ephemeral prompt only, exactly like every other directive (never into
    history/trajectory), and the model still chooses and emits its own
    action -- nothing here rewrites, substitutes, or filters what gets
    executed against the environment.
    """
    body = _STATIC_DIRECTIVES[InterventionType.REFLECT]
    if not admissible_commands:
        # Honest fallback: no list available (e.g. a think-step, or a caller
        # that doesn't track admissible commands) -- render v1 unchanged
        # rather than claiming an empty environment.
        return body
    commands = [str(c).strip() for c in admissible_commands if str(c).strip()]
    if not commands:
        return body
    shown = sorted(commands)
    truncated = len(shown) > _MAX_LISTED_ADMISSIBLE
    if truncated:
        shown = shown[:_MAX_LISTED_ADMISSIBLE]
    listing = "\n".join(f"  {c}" for c in shown)
    header = (
        "The environment accepts exactly these commands right now"
        + (f" (first {_MAX_LISTED_ADMISSIBLE} of {len(commands)}, alphabetical):"
           if truncated else ":")
    )
    return (
        f"{body}\n"
        f"{header}\n"
        f"{listing}\n"
        "Any command not in this list will do nothing. Choose your next "
        "Action from this list, copying its wording exactly."
    )

# Honest placeholders for the (practically never-hit, since REPLAN can only
# fire after warmup + several steps) case where no prior action/observation
# has been recorded yet -- never silently render "None" or an empty string,
# and never fabricate a plausible-looking action/observation in its place.
_NO_NONPROGRESS_ACTION = "(no non-progress action recorded this episode)"
_NO_NONPROGRESS_OBSERVATION = "(no corresponding observation recorded this episode)"

def _render_replan(
    *, goal: Optional[str], steps_since_meaningful_change: Optional[int],
    last_nonprogress_action: Optional[str], last_nonprogress_observation: Optional[str],
    current_observation: Optional[str],
) -> str:
    action_text = last_nonprogress_action if last_nonprogress_action else _NO_NONPROGRESS_ACTION
    observation_text = (
        last_nonprogress_observation if last_nonprogress_observation else _NO_NONPROGRESS_OBSERVATION
    )
    ssmc = steps_since_meaningful_change if steps_since_meaningful_change is not None else 0
    return (
        "[ACTIVE CONTROL DIRECTIVE: REPLAN]\n"
        f"No observed state change has occurred for {ssmc} steps.\n"
        f"The most recent action without observed progress was {action_text}.\n"
        f"The environment returned: {observation_text}\n"
        "Revise the ineffective part of the current strategy.\n"
        "Plan a route with 2-4 subgoals from the current observed state, "
        "preserving any subgoals already completed.\n"
        "Execute only the next unmet subgoal.\n"
        "Respond with exactly two lines:\n"
        "Thought: <give the revised route as A -> B -> C>\n"
        "Action: <one executable environment command>"
    )


def render_directive(
    intervention: InterventionType, *, goal: Optional[str] = None,
    steps_since_meaningful_change: Optional[int] = None,
    last_nonprogress_action: Optional[str] = None, last_nonprogress_observation: Optional[str] = None,
    current_observation: Optional[str] = None,
    admissible_commands: Optional[list] = None,
) -> str:
    if intervention == InterventionType.REPLAN:
        return _render_replan(
            goal=goal, steps_since_meaningful_change=steps_since_meaningful_change,
            last_nonprogress_action=last_nonprogress_action,
            last_nonprogress_observation=last_nonprogress_observation,
            current_observation=current_observation,
        )
    if intervention == InterventionType.REFLECT:
        return ("[ACTIVE CONTROL DIRECTIVE: REFLECT]\n"
                + _render_reflect(admissible_commands=admissible_commands)
                + "\n" + _ALFWORLD_REFLECT_OUTPUT)
    body = _STATIC_DIRECTIVES.get(intervention)
    if body is None:
        return ""
    return f"[ACTIVE CONTROL DIRECTIVE: VERIFY]\n{body}\n{_ALFWORLD_OUTPUT}"
