# `react_cool_0` is malformed — the one-shot baseline's cool score was depressed by a prompt-data defect

## 1. Summary

`react_cool_0` in `alfworld_runs/prompts/alfworld_3prompts.json` is the **only
defective demonstration among all 18** in that file (6 task types x 3 indices).
It is also, by coincidence of the pre-registered `demo_indices: [0]` choice,
exactly the demo the one-shot configuration serves for every `cool` task — and
`cool` is the task type whose score improves most when that demo is replaced.

This reframes the demo-index result for cool. It is not "we tried three options
and kept the highest score" (a procedure that capitalises on noise). The defect
was identified by inspecting the prompt file's structure, independently of any
success rate, and it singles out the same demo the scores single out.

## 2. The defect

Verbatim, `react_cool_0` lines 13-16:

```
 13 | > take pan 1 from stoveburner 3
 14 | > think: Now I find a pan (1). Next, I need to take it.
 15 | OK.
 16 | You pick up the pan 1 from the stoveburner 3.
```

The action on line 13 is not followed by its observation. Instead the *thought
that motivates it* appears afterwards (line 14), and the action's actual
observation (line 16) is displaced past that thought's `OK.` acknowledgement.
Every other demo in the file — and the rest of this one — follows
`> think: ... / OK.` then `> <action> / <observation>`.

Two lines also use `>action` with no space (lines 9, and two others), against
the `> action` form used everywhere else in all 18 demos.

Under one-shot this matters more than it would under two-shot: the model is
shown exactly one worked example of the task type, so a scrambled
think/action/observation pairing in that example has no clean counterpart to
average against.

## 3. Scan of all 18 demos

Checked for (a) `>` immediately followed by a non-space, and (b) an action line
immediately followed by a `> think:` line rather than by an observation:

| demo | `>`-no-space | action-then-think | verdict |
|---|---|---|---|
| **react_cool_0** | **3** | **1** | **defective** |
| all other 17 | 0 | 0 | clean |

## 4. Scores

Per-type `cool` successes out of 21, pooled from every full-134 run on file:

| demo index | draws | mean |
|---|---|---|
| idx0 (defective) | 8, 9, 12, 10, 10 | 9.80 |
| idx1 | 11 | 11.00 (n=1) |
| **idx2** | **16, 18** | **17.00** |

Both idx2 draws sit above the maximum of all five idx0 draws, so the ranges do
not overlap. The idx0 draws come from runs whose controller settings differ
(baseline, fixed_horizon, REFLECT-v3, maxint6, maxint12), which is why they
serve as a spread estimate rather than strict replicates; their tight 8-12
range is itself informative.

## 5. What this does and does not establish

Established: `react_cool_0` is malformed; it is the only malformed demo in the
file; it is the demo the one-shot baseline used for cool; and replacing it
raises cool's score by roughly 7 points on 21 tasks across two independent
draws.

Not established: that the malformation is *the* cause of the gap. `react_cool_1`
is clean and still scored 11/21 (n=1), below idx2's 17 — so demo content
clearly matters beyond well-formedness alone, and idx2 may simply also be a
better-matched example (it demonstrates opening a closed container to search
it, which many cool tasks require, whereas idx0 never opens anything). A clean
causal test would be to repair `react_cool_0`'s ordering in place and re-run it;
that has not been done.

## 6. Consequence for the reported number

The frozen 80/134 one-shot baseline was serving a defective ICL example for
21 of its 134 tasks. That is a property of the prompt data, not of the AE
controller, and it depressed the baseline for every method that uses the same
one-shot prompt — ReAct included, since all baselines share
`demo_config`/`_build_base_prompt()`. Any cross-method comparison drawn from
runs using `demo_indices: [0]` inherits this.
