# sustained-only REPLAN routing: 3x repeat-verified, rejected

Status: **REJECTED. REPLAN's `frustration_high and confidence_low` branch stays
unconditional in the official one-shot line.** No code change was needed — the
experiment ran entirely inside an isolated workdir (`sustained_only_workdir/`,
job 273526) that was never merged.

## 1. What was tested

The single-variable routing change first tried in
`ae_full_sustained_v1_134_report.md`: in `stateful_controller.py::_select_intervention()`,
REPLAN arms only once `steps_since_meaningful_change >= patch_duration_steps`;
if not yet sustained, control falls through to the REFLECT/VERIFY checks below
instead of returning REPLAN.

That earlier report ran this under a *different* configuration (cooldown5,
two-shot demos, the 134-task split), got 99/134 vs 102/134 on a **single** run,
and explicitly declined to draw a conclusion, stating it needed a
repeat-verification pass before real-vs-noise could be separated. That follow-up
was never done. This is it, under the one-shot line specifically.

Isolation: `diff -rq` confirmed `ae/controllers/stateful_controller.py` was the
only file differing from the official one-shot copy (sha256=
`f0b53c9bf5180f2714c122c8abc49dde7726b708cbcd8d47fa2d73f57f94908d`); the full CPU
regression suite passed; the sbatch preflight re-verified both before running.

## 2. Result — 3 repeats, per the >=3-run standing rule

| config | runs (successes / 60) | mean | stdev |
|---|---|---|---|
| REPLAN unconditional (official) | 29, 25, 28, 33 (n=4) | **28.75** | 3.30 |
| sustained-only | 28, 22, 24 (n=3) | **24.67** | 3.06 |

Difference of means: **-4.08**. This is the first candidate in this line whose
aggregate delta approaches the noise band's own width rather than sitting
comfortably inside it — but the aggregate is not what settles it.

## 3. Where the loss comes from: examine collapses

| type | unconditional (4 runs) | mean | sustained-only (3 runs) | mean | delta |
|---|---|---|---|---|---|
| **examine** | 2, 2, 2, 3 | 2.25 | **1, 0, 1** | **0.67** | **-1.58** |
| puttwo | 3, 2, 3, 3 | 2.75 | 3, 1, 1 | 1.67 | -1.08 |
| clean | 7, 4, 6, 6 | 5.75 | 7, 3, 4 | 4.67 | -1.08 |
| heat | 3, 4, 3, 6 | 4.00 | 3, 4, 4 | 3.67 | -0.33 |
| cool | 5, 4, 5, 6 | 5.00 | 5, 5, 5 | 5.00 | 0.00 |
| put | 9, 9, 9, 9 | 9.00 | 9, 9, 9 | 9.00 | 0.00 |

**examine is the decisive one: its three sustained-only draws (1, 0, 1) do not
overlap the four unconditional draws (2, 2, 2, 3) at all.** puttwo and clean each
lose about a point on average but their ranges do overlap, so on their own they
would not be separable from noise; examine is what carries the result.

The mechanism is consistent with the change's design. Delaying REPLAN hands the
first intervention slot to REFLECT, which is a local "fix the last action" nudge.
examine tasks fail precisely when the agent needs to abandon a whole search route
(the practice-log case study found examine failures thrash between locations for
30+ steps), which is REPLAN's job, not REFLECT's. Gating REPLAN behind a sustained
no-progress window therefore withholds the one intervention type that addresses
examine's actual failure mode until it is too late in the episode to matter.

Stated as a limit: this is a coherent post-hoc explanation of an observed
per-type effect, not an independently tested causal claim.

## 4. Conclusion

REPLAN's routing stays unconditional. The earlier single-run 99-vs-102 result was
correctly treated as inconclusive at the time; with 3 repeats the direction is
now consistent and localized to examine, which is enough to close the question.
Do not re-propose sustained-only gating for REPLAN without a mechanism that
specifically protects examine-type route-abandonment.

Alongside `ae_reflect_prompt_v2_rejected.md`, this closes the second of the two
intervention-routing/wording ideas carried over from earlier in the project.
