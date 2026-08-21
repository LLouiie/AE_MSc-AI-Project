# fixed_horizon termination policy: tested and rejected

Status: **REJECTED. One-shot AE keeps `legacy_early_stop` as the official
termination policy. The frozen 80/134 (practice 58/100 + exam 22/34) result
is unaffected** — no code change was needed for this experiment
(`--termination-policy` is a pre-existing CLI flag), and the decision rule
was pre-registered against practice alone before either run was submitted.

## 1. Hypothesis and why it seemed promising

`legacy_early_stop` force-ends an episode with `termination_reason=
"exhausted_repeated"` (success=0 by construction) the moment
`consecutive_exact_action_repeat` fires past its grace window.
`fixed_horizon` removes that early exit and lets the episode run to
MAX_STEPS=50 / a real win / env-terminal instead, with the controller's
REFLECT/REPLAN/VERIFY logic otherwise unaffected. In the frozen official
practice log, 20/100 episodes (6/34 exam) ended via `exhausted_repeated` —
guaranteed failures under `legacy_early_stop`. The reasoning offered at the
time: since early-stopping can only end an already-failing trajectory
sooner, switching to `fixed_horizon` should be **weakly dominant** on
success rate for a fixed trajectory — same or better, never worse.

## 2. Why that reasoning was wrong for a real (non-replay) comparison

The "can only help" argument is only valid for a **counterfactual replay**
of the identical trajectory with only the termination trigger changed. A
real fixed_horizon run is an **entirely independent LLM/vLLM session** —
at temperature=0 this project has repeatedly confirmed sessions are not
bit-identical (see the dev60 noise-floor work, `feedback_ae_standing_constraints.md`).
An independent run can diverge from the very first token, so a task that
succeeded under `legacy_early_stop` is not guaranteed to still succeed
under `fixed_horizon` — noise can flip previously-successful episodes to
failures just as easily as it can rescue previously-doomed ones. This is
exactly what the result shows.

## 3. Result (practice, decision-determining)

| | legacy_early_stop (frozen) | fixed_horizon |
|---|---|---|
| practice | 58/100 | **57/100** |

Paired per-task comparison (matched on `(env_name, goal)`, the required
pairing key for this task list — see the demo-selection ablation's
pairing-key gotcha):

| | count |
|---|---|
| both succeed | 54 |
| both fail | 39 |
| success -> fail (fixed_horizon regressed) | 4 |
| fail -> success (fixed_horizon rescued) | 3 |
| **net** | **-1** |

Of the 20 practice tasks that were `exhausted_repeated` under
`legacy_early_stop` (guaranteed failures there), only **2** succeeded under
`fixed_horizon`
(`pick_heat_then_place_in_recep-Cup-None-Cabinet-10`,
`pick_two_obj_and_place-PepperShaker-None-Drawer-10`). The other 18 either
ran out the clock (`env_done_without_success` rose from 19 to 36) or hit
`max_steps` (3 to 7) — i.e. most of the extra budget was spent without
producing a different outcome.

## 4. Exam (recorded for completeness only, not used for the decision)

Submitted in parallel purely to save wall-clock time; the adopt/reject
decision above was locked from practice alone, before this number was
examined, per `feedback_ae_standing_constraints.md`.

| | legacy_early_stop (frozen) | fixed_horizon |
|---|---|---|
| exam | 22/34 | 19/34 |

Direction is consistent with practice (flat-to-down), reinforcing rather
than contradicting the practice-based rejection.

## 5. Conclusion

`fixed_horizon` is rejected. Practice shows a net -1 (57 vs 58), well
within this project's noise floor and not a rescue of the
`exhausted_repeated` population it was meant to fix (2/20 recovered, offset
by 4 unrelated success->fail flips elsewhere). One-shot AE's official
number stays 80/134 (58/100 practice + 22/34 exam) under
`legacy_early_stop`. The general lesson for future candidates: a "can only
help, never hurt" argument for a config change is only sound under
counterfactual-replay reasoning, not for two independent non-deterministic
runs — always treat it as a hypothesis to test, not a guarantee, and still
require the standing >=3-repeat-before-concluding discipline for anything
without an unambiguous single-run result this decisive.
