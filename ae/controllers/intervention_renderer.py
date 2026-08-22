"""Renders an InterventionType into the short control-directive text shown
to the agent. AE never generates actions itself — it only nudges the
existing ReAct agent's next Thought/Action via this directive block, which
is injected transiently into the prompt for the LLM call and never written
into the persistent trajectory/history (see alfworld_runs_ae/agents.py's
loop for where it's spliced in and dropped).

REPLAN prompt-v2 (see audit_reports/ae_intervention_prompt_audit*.md): the
v1 REPLAN directive was a static template asking the model to "restate the
task" and "construct a short plan of 2-4 subgoals" without ever actually
supplying the task text, the specific action that just failed, or the
current observation -- real-log audit found the model rarely engaged with
those instructions and, when it DID try to comply in full, it broke the
enforced one-Thought/one-Action output format twice (producing an
unparseable action). v2 supplies these variables explicitly (reusing state
the controller already tracks, plus the raw goal/action/observation text
the caller already has -- no new state machine) and compresses the
"restate + new route + why" ask into a single Thought line using compact
"A -> B -> C" notation, matching what the enforced output format can
actually hold.

REPLAN prompt-v2 "B+" semantic fix (see
audit_reports/ae_replan_prompt_v2_last_action_semantics_audit.md): the
first v2 cut labeled the most recent action "Last failed action" even when
that action was simply the immediately-preceding one, unconditionally,
regardless of whether it had actually made progress -- real-log replay
found 5/15 REPLAN windows where a genuinely-progressing action would have
been mislabeled this way. The controller now only ever offers an action
that registered NO observed local state change (local_state_change_proxy <
0.5, the same progress signal steps_since_meaningful_change already uses),
and the wording below says exactly that ("no observed progress"), not
"failed" -- a null observation only proves the absence of a detected
change, not that the action was semantically wrong.

REPLAN prompt-v2 progress-aware wording fix (see
audit_reports/ae_replan_prompt_v2_b_plus_report.md section 9 and
ae_replan_prompt_v2_progress_wording_report.md): REPLAN's directive is
shown for the full patch_duration_steps window regardless of whether the
model recovers partway through it (unchanged, by design -- this fix does
NOT shorten or end REPLAN early). But the wording itself was static
regardless of steps_since_meaningful_change, so a window where the model
recovered on its very first directed step showed a literally
self-contradicting "No meaningful progress has been made for 0 steps" next
to instructions to discard the (just-productive) route -- confirmed for
real in 5/15 REPLAN windows (7 directive instances) in the 134-task
sustained-only log. The body now branches on steps_since_meaningful_change
(the same existing field, no new signal): > 0 keeps the original
"no-progress, discard and replan" framing; == 0 switches to a
"progress-observed, continue the productive route" framing that still
carries the goal, the current observation, and an explicit prohibition on
returning to the earlier no-progress action -- it never claims zero
progress or tells the model to discard a route that just worked.

REFLECT prompt-v3, admissible-action grounding (this workdir, 2026-08-13):
REFLECT's directive now additionally carries the environment's own
admissible-command list for the next step, when the caller supplies one.
See _render_reflect()'s docstring for the full rationale, the practice-log
evidence (42/42 failures contain inadmissible actions, 591 total, vs a
median successful episode of 10.5 steps), and why this replaces the
rejected REFLECT-v2 static-advice approach
(audit_reports/ae_reflect_prompt_v2_rejected.md). VERIFY is unchanged,
byte-for-byte, from v1; so is REFLECT's own v1 body text, which the v3
listing is appended to rather than replacing.
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

    Why this and not more static advice (see
    audit_reports/ae_reflect_prompt_v2_rejected.md): REFLECT-v2 tried a
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
_NO_CURRENT_OBSERVATION = "(no observation recorded this episode)"
_NO_TASK_TEXT = "(task text unavailable)"


def _render_replan(
    *, goal: Optional[str], steps_since_meaningful_change: Optional[int],
    last_nonprogress_action: Optional[str], last_nonprogress_observation: Optional[str],
    current_observation: Optional[str],
) -> str:
    task_description = goal.strip() if goal else _NO_TASK_TEXT
    action_text = last_nonprogress_action if last_nonprogress_action else _NO_NONPROGRESS_ACTION
    nonprogress_obs_text = (
        last_nonprogress_observation if last_nonprogress_observation else _NO_NONPROGRESS_OBSERVATION
    )
    current_obs_text = current_observation if current_observation else _NO_CURRENT_OBSERVATION
    ssmc = steps_since_meaningful_change if steps_since_meaningful_change is not None else 0

    if ssmc > 0:
        status_line = f"No meaningful progress has been made for {ssmc} steps."
        guidance_line = "Discard the current ineffective strategy and create a different route."
        plan_bullets = (
            "- restate the exact goal;\n"
            "- give a different 2-4-subgoal route using A -> B -> C;\n"
            "- explain why the next action differs from the most recent action with no observed progress."
        )
        subgoal_instruction = "the first subgoal"
        bottom_prohibition = "Do not repeat the most recent action with no observed progress."
    else:
        # Real progress was registered on the step immediately before this
        # directive was rendered (steps_since_meaningful_change reset to 0)
        # -- REPLAN keeps showing for the rest of its patch_duration_steps
        # window regardless (unchanged), but the wording must not claim
        # zero progress or ask the model to discard a route that just
        # worked. Two further precision fixes: (1) real call-sequence audit
        # found the step that reset ssmc to 0 can occur BEFORE REPLAN
        # actually finishes arming within that same step() call (arming
        # happens after that step's own bookkeeping) -- "since REPLAN was
        # activated" is therefore not always an accurate time reference, so
        # this only claims what's directly observed: the latest step itself
        # produced a change. (2) local_state_change_proxy>=0.5 only proves
        # an observed state change happened, not that the resulting route
        # is actually beneficial for the task -- the wording must not
        # assert "productive" as a settled fact.
        status_line = "The latest step produced an observed state change."
        guidance_line = (
            "Continue from the latest observation and do not discard the updated route "
            "solely because REPLAN remains active."
        )
        plan_bullets = (
            "- restate the exact goal;\n"
            "- continue the current productive route with the next subgoal using A -> B -> C;\n"
            "- explain why this next action follows from the latest observation, "
            "not from the earlier no-progress action."
        )
        subgoal_instruction = "the next subgoal"
        bottom_prohibition = "Do not return to the most recent action with no observed progress."

    return (
        "[ACTIVE CONTROL DIRECTIVE: REPLAN]\n"
        "\n"
        f"Original task: {task_description}\n"
        f"{status_line}\n"
        f"Most recent action with no observed progress: {action_text}\n"
        f"Environment response to that action: {nonprogress_obs_text}\n"
        f"Current observation: {current_obs_text}\n"
        "\n"
        f"{guidance_line}\n"
        "\n"
        "In one concise Thought line:\n"
        f"{plan_bullets}\n"
        "\n"
        f"Then output exactly one executable Action line and execute only {subgoal_instruction}.\n"
        f"{bottom_prohibition}\n"
        "Do not assume any state that has not been observed."
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
        return ("[ACTIVE CONTROL DIRECTIVE]\n"
                + _render_reflect(admissible_commands=admissible_commands))
    body = _STATIC_DIRECTIVES.get(intervention)
    if body is None:
        return ""
    return f"[ACTIVE CONTROL DIRECTIVE]\n{body}"
