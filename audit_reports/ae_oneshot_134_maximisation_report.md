# One-shot AE on 134 tasks: what raised the score and what did not

Work done 2026-08-13 into 2026-08-14 under a standing instruction to raise the
134-task one-shot AE number by any means, with the user explicitly directing
(twice) that tuning use practice **and** exam together.

**Therefore every number below is fitted on all 134 tasks and must be reported
as such, not as a generalization result.** The one component that also
reproduces on the held-out exam split specifically is flagged where it appears.

---

## 0. Final numbers (2026-08-16, supersedes section 1)

The max-score configuration was re-run **20 independent times** (job array
274469, all 20 verified to cover 134 unique tasks, zero failures):

```
95 93 98 92 94   93 92 94 96 92   90 93 90 97 97   95 96 94 94 99
```

| configuration | success rate | 134-task score | n |
|---|---|---|---|
| ReAct, shared demo mapping | 63.7% ± 1.6 | 85.3 ± 2.1 | 3 |
| AE, shared demo mapping | 64.9% ± 0.7 | 87.0 ± 1.0 | 3 |
| **AE + tuned demo mapping** | **70.3% ± 1.8** | **94.2 ± 2.5** | **20** |

± is standard deviation; over the 20 runs the standard error is ±0.6 (score)
and ±0.4pp (rate). Observed range 90-99.

**The earlier 98.0 ± 6.2 (91/100/103) was inflated and is withdrawn.** Those
three repeats shared one long-lived vLLM server, which lets the prefix cache
warm across repeats; a Cochran Q permutation test rejected their
exchangeability at p=0.0034, and the trend was monotone 91 → 100 → 103. Both
100 and 103 lie above the maximum of the 20 clean runs. Each of the 20 repeats
here starts its own vLLM server, and the within-task position means are flat
(94.2 / 93.8 / 94.2 / 94.6 for the 1st/2nd/3rd/4th repeat of each array task),
confirming the drift is gone.

At the user's request the best 5 of the 20 were also computed: 99, 98, 97, 97,
96 → 97.4 ± 1.1. **That is a best-5-of-20 selection, not an n=5 experiment,
and must not be reported as the latter.**

### 0.1 Why repeats vary at all: it is not the seed

Determinism probe, job 274442 — same 40 tasks run twice per arm, strictly
serially:

| server flags | trajectories byte-identical | tasks flipped |
|---|---|---|
| default (`enable_prefix_caching=True`, `enforce_eager=False`) | 10/40 | 5 |
| `--enforce-eager --no-enable-prefix-caching --max-num-seqs 1` | **40/40** | **0** |

So the variation is numerical, from prefix caching and CUDA-graph kernel
selection, and it disappears entirely when those are disabled.

It is **not** seed variance, and no seed sweep can address it:

- every run in the project used `--seed 42`; the seed was never varied
- the seed is inert — `run_alfworld.py:95` calls `random.seed(args.seed)` and
  `random.` appears nowhere else in `ae/`
- both LLMs are `temperature=0` (`run_alfworld.py:49,52`), i.e. greedy argmax,
  so there is no sampling RNG for a seed to control in the first place

The 20 runs above deliberately keep the default (nondeterministic) flags so
they remain comparable with every earlier run in this report; only the
shared-server drift was removed.

### 0.2 Consequence for the AE-vs-ReAct claim

Paired across the 3 repeats of the earlier max-score config, 87 tasks always
succeeded, 26 always failed, and 21 (15.7%) flipped — all score movement lives
in those 21. AE 87.0 ± 1.0 vs ReAct 85.3 ± 2.1 is a +1.7 gap with overlapping
ranges. **"Significantly outperforms" is not supportable**; the two baselines
have not been re-run at n=20.

---

## 1. Headline (historical — superseded by section 0)

| configuration | 134-task score | n |
|---|---|---|
| frozen one-shot baseline | 80/134 | 1 |
| **best per-type demo mapping (measured)** | **89, 85 → mean 87.0** | **2** |
| final demo mapping | *did not run — see below* | 0 |
| final mapping, predicted from per-type means | 92.1/134, plausible range 88-96 | — |

**The measured, defensible result is 87.0/134 (89 and 85, two independent
repeats) against a frozen baseline of 80/134.** That mapping used
put/clean/heat=idx0, cool/examine=idx2, puttwo=idx1.

The "final" mapping (heat switched to idx2, examine reverted to idx0) is a
refinement made after those repeats, on further per-type draws. It was
submitted for confirmation but **never ran**: every GPU partition on the
cluster (a40, a100, a30) is saturated with multi-day jobs, with 10+ jobs queued
ahead. SLURM's own estimate put the start over 24 hours out. Its 92.1 figure is
therefore a projection from per-type means, not a measured 134-task run, and
must be labelled as such.

I made this marginally worse: seeing the long estimate, I cancelled the two
queued jobs and resubmitted them with shorter time limits, guessing the 10-hour
limit was blocking backfill. It was not — the real cause is that all seven a40
GPUs are held by jobs with 12 to 46 hours remaining. The resubmission lost the
jobs' accumulated queue age and pushed the estimated start from 05:20 to 10:20
the next day. The jobs (274027 AE, 274028 ReAct, one repeat each) remain queued
and will run when capacity frees.

The prediction's range comes from resampling the actually-observed per-type
draws (20k bootstrap, 90% interval [88, 96]). It reflects run-to-run spread
already seen rather than a model-based confidence interval, and with only 3-5
draws per type it understates the tails — treat it as a plausible range. It is
stated here because tonight repeatedly showed that single-point estimates on
this benchmark mislead: three separate candidates (REFLECT-v3 83, maxint6 82,
heat idx0's 14) looked real as point estimates and evaporated on repetition.

**Every point gained came from the ICL demonstration mapping. The AE controller
contributed nothing that survived repetition.**

## 2. What survived: the per-type demo mapping

`demo_config.py` serves one demonstration per task type from
`alfworld_3prompts.json`, which holds three (`react_{type}_{0,1,2}`) for each.
The pre-registered one-shot config used index 0 uniformly. Chosen on **means of
repeated draws**, not single-draw maxima:

| type | chosen | draws for chosen | mean | idx0 draws | idx0 mean |
|---|---|---|---|---|---|
| put | idx0 | 23, 23, 23 | 23.00 | — | — |
| clean | idx0 | 23, 24, 23 | 23.33 | — | — |
| examine | idx0 | 3, 3, 7, 4, 2 | 3.80 | — | — |
| **heat** | **idx2** | 13, 14, 16 | **14.33** | 14, 8, 10, 9 | 10.25 |
| **cool** | **idx2** | 16, 18, 18 | **17.33** | 8, 9, 12, 10, 10 | 9.80 |
| **puttwo** | **idx1** | 11, 11, 9 | **10.33** | 9, 7, 7, 7, 7 | 7.40 |

put and clean are never displaced: they are the only two whose index-0 demo is
confirmed verbatim against ReflAct's paper (Appendix K). examine is kept at
index 0 because switching buys nothing (idx2 mean 3.67 vs idx0 3.80).

### 2.0 The three switches are NOT equally well supported

Comparing the observed ranges rather than just the means:

| type | idx0 range | chosen range | overlap | strength |
|---|---|---|---|---|
| **cool** | [8, 12] n=5 | [16, 18] n=3 | none | **strong** — fully disjoint, +7.5 on means |
| puttwo | [7, 9] n=5 | [9, 11] n=3 | touches at 9 | moderate — +2.9 on means |
| heat | [8, 14] n=4 | [13, 16] n=3 | yes | **weaker** — +4.1 on means but overlapping |

Only cool is established beyond argument. heat's switch rests on means
(10.25 → 14.33) while its ranges overlap, because idx0's single 14 sits inside
idx2's range — and that same 14 is the outlier draw that caused my earlier
wrong call to keep heat at idx0. The switch is the better bet on the evidence
available, but it is a bet, and it deserves more draws before being treated as
settled. puttwo sits in between.

Inspection of the newly adopted demos found nothing anomalous: `react_heat_2`
and `react_puttwo_1` are both well-formed (correct think/action/observation
ordering, consistent `> ` prefixes), and `react_heat_2` happens to demonstrate
the correct "heat while holding the object" mechanic — though so does
`react_heat_0`, so that is not what separates them.

### 2.1 cool is not a tuning result — it is a defect fix

`react_cool_0` is **the only malformed demonstration among all 18** in the
prompt file. Its lines 13-16 scramble the think/action/observation pairing: the
action's observation is displaced past a later thought's `OK.`, and three lines
use `>action` with no space against the `> action` form used everywhere else.
Full detail: `ae_oneshot_cool_demo_defect.md`.

That defect was found by inspecting the prompt file's structure, independently
of any success rate, and it singles out the same demo the scores single out.
The gain also reproduces on the held-out exam split: **cool exam went 1/5 → 5/5
in both independent repeats**, and practice 7/16 → 13/16 in both. A result
produced purely by fitting 134 tasks would not be expected to behave that way.

## 3. What did not survive

| change | result | verdict |
|---|---|---|
| REFLECT-v3 (admissible-action grounding) | 83/134 (n=1) | died on repeats |
| max_interventions = 6 | 82/134 (n=1) | died on repeats |
| max_interventions = 12 | 80/134 (n=1) | nothing |
| REFLECT-v3 + maxint6 combined | 82, 81 → **81.5** (n=2) | **no gain over baseline 80** |
| fixed_horizon termination | 76/134 (n=1) | no evidence of help |
| sustained-only REPLAN routing | dev60 mean 24.67 vs 28.75 (n=3) | rejected, examine collapses |
| REFLECT prompt-v2 ("Nothing happens" advice) | dev60 mean 28 vs 28.75 (n=2) | rejected, hurt heat |

The controller-side story is worth stating plainly: the single-draw results
(83, 82) looked like gains and were not. Only running the combination twice
revealed 81.5 — i.e. baseline. See `ae_sustained_only_replan_rejected.md` and
`ae_reflect_prompt_v2_rejected.md` for the two rejections with their own
repeat evidence.

### 3.1 Why the controller could not help much — the diagnosis that motivated it

Analysis of the 42 practice failures in the frozen baseline:

- all 42/42 contain at least one inadmissible action; **591 in total**
- median *successful* episode is 10.5 steps and no success ever exceeded 38 of
  the 50-step budget, so failing episodes burn their budget on actions the
  environment silently ignores
- **38 of 42 failures exhaust the 3-intervention budget**, then run a median 19
  further steps (max 38) with no directive active
- only 22.1% of steps inside failing episodes have any directive active;
  **74.5% of the 591 inadmissible actions occur with nothing active**

This is why REFLECT-v3 was built (ground the model in the environment's own
admissible-command list rather than prescribing a recovery action) and why
max_interventions was swept. Both were reasonable given the diagnosis; neither
produced a reproducible gain. REFLECT-v3's one draw did concentrate where it
was aimed — examine 3→7, cool 8→12, its two target types — but put went 23→21
and puttwo 9→7, and the combination run at n=2 landed at baseline.

## 4. Methodological record

**Two selection errors I made and corrected**, both the same mistake — choosing
on a single lucky draw:

1. **heat kept at idx0** on a first-draw comparison of idx0=14 vs idx2=13.
   Three further idx0 draws came in at 8, 10, 9, exposing the 14 as an outlier;
   idx2 (13, 14, 16) is about 4 points better. Corrected.
2. **examine switched to idx2** on a single 5-vs-3 draw. With repeats both
   indices sit at ~3.7-3.8. Reverted to the pre-registered idx0.

**Noise floor.** Repeated draws of identical configurations give: dev60 one-shot
AE 29, 25, 28, 33 (mean 28.75, sd 3.30 on 60 tasks); heat idx0 14, 8, 10, 9
(range 6 on 23 tasks); cool idx0 8-12; put 23, 23, 23 (stable). Noise is
concentrated in heat/cool/examine/puttwo; put and clean are steady. A standing
rule of >=3 repeats before accepting or rejecting a dev60 candidate was added
to the project constraints as a direct result.

**Two infrastructure bugs, both mine, both found and fixed:**

1. `GROUPS` is a bash reserved readonly array (the caller's group IDs).
   Assigning to it was silently ignored and `$GROUPS` expanded to a numeric
   GID, so a job went looking for `configs/demos/1520.yaml`. Renamed.
2. Two co-scheduled jobs both hard-coded vLLM port 8000; the second silently
   attached to the first's server and died when that job exited. Now each job
   derives a port from `SLURM_JOB_ID`, refuses to start if the port already
   answers, and re-checks after readiness that its *own* vLLM process is still
   alive. Worth noting the failure mode: had both jobs stayed alive they would
   have shared one model server while recording separate results — silently
   contaminated data, no error. The crash was luck.

## 5. Open item: baseline fairness

All baselines share `demo_config`/`_build_base_prompt()`, so the frozen ReAct
one-shot number (68/134) also ran on the malformed `react_cool_0`. Reporting AE
on the fixed mapping against ReAct on the defective one would give AE a better
prompt than its own baseline and make the headline gap meaningless. ReAct under
the identical final mapping is running as job 273992.

Frozen per-type comparison, both on index 0 (ReAct → AE): put 22→23,
clean 23→23, heat 11→14, cool 9→8, examine 0→3, puttwo 3→9; total 68→80 (+12).
AE's edge does **not** come from cool (it is −1 there), so the fix is not
expected to simply transfer to ReAct — but that has to be measured, not assumed,
and until 273992 reports, no claim about the post-fix AE-vs-ReAct gap is
supportable.
