# REFLECT prompt-v2 ("Nothing happens" fix): tested and rejected

Status: **REJECTED. One-shot AE's official REFLECT directive remains v1 (byte-identical,
sha256=c4824c0606a87eedc32aa96564c9ddcedafc1fd11a7b212a3131382fe43367ba), unchanged.**
The 134-task official one-shot baseline result is unaffected — this experiment ran entirely
inside an isolated workdir (`reflect_v2_workdir/`) that was never merged into
`one_shot_oldcontroller_promptv2_seed42`, verified by sha256 immediately before writing
this report.

## 1. Motivation

Real-trajectory replay of every one-shot examine-type failure (18/18, both AE and ReAct
dev/practice baselines) found the same pattern: the model re-issues `go to <receptacle N>`
for a receptacle already reachable from its current position (ALFWorld groups several
numbered receptacles under one physical "go to" location), gets `"Nothing happens."`, and
REFLECT's v1 directive never told it what that response usually means. Hypothesis: one
unconditional added line naming the concrete recovery move ("the target may already be
reachable... try a direct interaction instead of re-navigating") would fix examine without
touching REPLAN/VERIFY or any controller logic.

## 2. Experiment

Isolated workdir `reflect_v2_workdir/` (rsync copy of the merged one-shot line), single
change: the `InterventionType.REFLECT` entry in `ae/controllers/intervention_renderer.py`
gained one unconditional line. REPLAN and VERIFY untouched (diff-verified). Run: 2x repeat
on the frozen 60-task dev set (job 273236, both runs exit 0), compared against the existing
4-run REFLECT-v1 baseline on the same dev set (task #82's 3 noise runs + the sweep's
baseline replicate, job 273214): 29, 25, 28, 33 (mean 28.75, stdev 3.30 — wider than
originally estimated from 3 runs alone).

## 3. Result: task-type breakdown (10 tasks/type on dev60)

| type | REFLECT-v1 (4 runs) | REFLECT-v2 (2 runs) | verdict |
|---|---|---|---|
| examine (the target) | 2,2,2,3 (mean 2.25) | 2,2 (mean 2.00) | **no improvement** — the fix didn't even help its own target |
| heat | 3,4,3,6 (mean 4.00) | 2,2 (mean 2.00) | **regressed**, both reps below the full v1 range |
| cool | 5,4,5,6 (mean 5.00) | 5,4 (mean 4.50) | within noise, no reliable signal |
| puttwo | 2,2,3,3 (mean 2.75) | 4,4 (mean 4.00) | both reps above v1 range, mechanism unconfirmed |
| put/clean | flat | flat | no signal |

Overall dev60 total (29 and 27) lands inside the widened REFLECT-v1 noise band
(25–33) — indistinguishable from noise at the aggregate level. The task-type
breakdown is what surfaces the real story.

## 4. Root-cause trace confirming the heat regression is real, not noise

`dev_idx=39` (`heat cup ... cabinet`) is a clean natural experiment: identical prompt
history up to the exact same failure step across a REFLECT-v1 run (succeeded) and a
REFLECT-v2 run (failed), diverging at the single step where REFLECT fires:

```
Action: put cup 1 in microwave 1  -> succeeds
Action: heat cup 1 with microwave 1  -> "Nothing happens."   [REFLECT fires]
```

- **REFLECT-v1** (succeeded): next action `take cup 1 from microwave 1`, then
  `heat cup 1 with microwave 1` succeeds. Correct fix: ALFWorld's `heat` action requires
  holding the object, not having it already placed inside the microwave.
- **REFLECT-v2** (failed): next action repeats `heat cup 1 with microwave 1` verbatim —
  directly violating REFLECT's own "do not repeat the same failed action" line, because
  the new line's wording ("target may already be reachable... try a direct interaction
  instead") reads naturally as "just retry the interaction," which is exactly wrong here.
  The episode then thrashes (close/reopen microwave, stoveburner, a second cup) and
  terminates by repeatedly issuing an invalid `finish` action.

This is the mechanistic confirmation, not just a statistical one: heat's correct recovery
(remove object, then re-interact) is the **opposite** of examine's correct recovery (skip
re-navigation, interact directly without moving anything) — the same "Nothing happens."
surface symptom has semantically incompatible fixes depending on task type. A single static
line cannot be correct for both, and empirically it wasn't correct for either (examine flat,
heat down).

## 5. Why the fix is rejected outright rather than made more targeted

Considered and rejected: branching the REFLECT directive on `task_type`. Two independent
reasons, not just one:

1. **It wouldn't even be justified by what the data show.** The premise for going more
   targeted would be "the general version works but needs sharpening." That's not what
   happened — the general version didn't help its own intended target (examine, flat at
   2.25->2.00). There's no validated effect to preserve by adding branches.
2. **It contradicts this project's own design principle.** AE's premise is a general
   affect-driven controller, not a per-task-type lookup table. The one precedent in this
   codebase for conditioning a directive's wording (REPLAN prompt-v2's
   `steps_since_meaningful_change`-based branch, see
   `ae_replan_prompt_v2_progress_wording_report.md`) branches on a signal the controller
   already tracks for every task — never on the ALFWorld-specific `task_type` label, which
   has no meaning outside this one benchmark and would be indistinguishable from hand-coding
   answers to the 60 dev tasks used to discover the pattern.

## 6. Conclusion

One-shot AE's REFLECT directive stays at v1. No code change needed (never merged into the
official line). `reflect_v2_workdir/` is retained as-is (isolated, harmless) as the record
of this experiment; no further action on it. Any future attempt to give REFLECT
"Nothing happens." guidance should condition on a general, already-tracked controller
signal (as REPLAN prompt-v2 did) — never on `task_type` — or should not prescribe a
specific recovery action at all and instead push the model to reason from the object's
own state.
