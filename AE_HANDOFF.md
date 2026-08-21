# AE — Handoff Document

**Scope of this document:** how the AE controller is designed and why, what state the
experiments are in right now, and what should happen next. Ablation *design* is
deliberately out of scope — the author has not settled on it yet and does not want it
pre-empted. Do not propose an ablation matrix unless asked.

Last updated: 2026-08-21.

---

## 0. Orientation

| | |
|---|---|
| Repo | `/homes/jy625/projects/AE3`, branch `feature/intra-episode-ae` |
| Remote | `https://github.com/LLouiie/AE3.git` (no upstream tracking set) |
| Run copies | `/vol/gpudata/jy625-ae-data/scratch_ae3/` — **isolated snapshots**, not the repo |
| Python | `/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3` |
| Model | Qwen3-8B, served locally by vLLM, `temperature=0` |
| Task suite | ALFWorld, 134 tasks (first 100 = `practice`, last 34 = `exam`) |
| Cluster | Slurm, A40 GPUs, **QOS caps this user at 6 submitted jobs** |
| Notifications | jobs self-report to `https://ntfy.sh/AE_new_alert` on exit |

This is a master's-thesis project. The deliverable is an argument, not a product: the
numbers only matter insofar as they support or refute the design claim in §1.

---

## 1. What AE is, in one page

**The claim.** An LLM agent running ReAct inside a single episode already emits enough
observable evidence to tell when it is stuck — repeated actions, illegal actions,
observations that stop changing. Existing self-correction methods (Reflexion, ADaPT,
ReflexGrad) act on that evidence either *between* episodes (Reflexion reflects after
failure and retries) or by *restructuring the task* up front (ADaPT decomposes). AE asks a
narrower question:

> Can a cheap, deterministic, **intra-episode** controller — no extra LLM calls, no memory
> across episodes, no retries — read those signals as they arrive and intervene in time to
> rescue the episode?

**"AE" = appraisal engine.** The controller maintains four scalar variables named after
affective constructs (uncertainty / frustration / surprise / confidence). This naming is
**not** a claim about machine emotion, and the code says so explicitly
(`ae/controllers/affect_state.py` module docstring). The names label *what observable
pattern each variable tracks*. If you are writing about this, keep that framing — it is
appraisal-theory-*inspired* control, and overclaiming here is the fastest way to make the
thesis indefensible.

**What AE is architecturally.** A side-car on an otherwise unmodified ReAct loop. The same
`ALFWorldAgent` serves ReAct, Reflexion, and AE; AE is the case where the optional
`controller=` parameter is non-`None` (`alfworld_runs_ae/agents.py:287`). AE never chooses
an action. It only injects a short directive block into the *next* prompt, which the
existing ReAct agent then reacts to on its own.

**Three hard invariants.** Every one of these is load-bearing for the cost argument, and
all three are enforced in code:

1. **Zero extra LLM calls.** Directives are fixed template strings
   (`ae/controllers/intervention_renderer.py`), spliced into the prompt of the ReAct call
   that was going to happen anyway. `ae_summary.intervention_llm_calls` is hardcoded `0`
   and was empirically `0` in all 20 production runs.
2. **Zero cross-episode state.** `StatefulController` is constructed fresh per episode and
   discarded (`ae/baselines/ae_full.py:22`). No experience bank, no retrieval, no
   parameter updates, no environment reset.
3. **Zero LLM in the control path.** Signal extraction is string normalization, token-set
   Jaccard, and canonical action parsing. No embeddings, no LLM judge.

---

## 2. The design, layer by layer

Data flows `signals → affect state → hysteresis bands → rule tree → gates → directive`.
Files: `ae/controllers/{signals,affect_state,stateful_controller,intervention_renderer,canonical_action,config}.py`.

### 2.1 Layer 1 — Signals (`signals.py`, 338 lines)

Ten floats in `[0,1]` computed per `(action, observation)` transition. The ones that matter:

| Signal | What it actually measures |
|---|---|
| `invalid_action` | action not in `admissible_commands` **as captured before the action** (see trap §5.1). `think:` actions are exempt — they are the agent's scratchpad convention, not env commands. |
| `repeated_action` | max canonical-signature similarity against the last 3 actions |
| `consecutive_exact_action_repeat` | **strict**: current vs. the single immediately-preceding action only, exact canonical match. Not windowed. |
| `repeated_observation` / `observation_novelty` | Jaccard on observation text vs. last 3; novelty is the complement |
| `local_state_change_proxy` | did the world observably change — ALFWorld's own `"Nothing happens."` string, **or** the admissible-command set changing |
| `unexpected_outcome` | legal *state-changing* action (open/close/take/put/heat/cool/clean/toggle) that produced no local state change. Exploratory families (go/look/examine/…) never raise this. |
| `information_gain_proxy` | novel observation or changed admissible set |
| `progress_signal` | **NOT task progress** — see below |
| `budget_ratio` | `step_index / max_steps` |

Two things to internalize here:

**`progress_signal` is misnamed and the code admits it.** ALFWorld exposes no per-step
partial credit — `won` is only ever `True` on the terminal successful step. `progress_signal`
is an *observation-novelty proxy*: "a legal, non-think action whose observation was novel
enough to look unstuck." The name survives only because renaming it would touch
`StepSignals`, every `progress_signal`/`progress_relief` config key, the log schema, and
the test suite for no behavioral gain. **Never describe it as ground-truth task progress in
the write-up.**

**Three distinct proxies, deliberately separated.** `local_state_change_proxy`,
`unexpected_outcome`, and `information_gain_proxy` were once one observation-similarity
heuristic. That version over-fired: `go to shelf 2` → `you see nothing` got flagged as
surprise purely from text similarity, even though it is completely routine search. The
family-based split (`_STATE_CHANGING_FAMILIES` vs `_EXPLORATORY_FAMILIES`) is the fix.

**`consecutive_exact_action_repeat` exists for the same reason.** `repeated_observation`
over-triggers on same-template-different-location exploration (two different empty shelves
both say "you see nothing"). The strict adjacent-step signal is the *only* thing gating the
`frustration_medium` → REFLECT branch. Don't "simplify" it back into `repeated_action`.

### 2.2 Layer 2 — Affect state (`affect_state.py`)

Four variables, one update rule, applied independently:

```
new_state = clip01( decay * previous_state + signal_delta )
signal_delta = Σ (weight_i * signal_i) + progress_relief * progress_signal
```

Weights and decays live entirely in YAML (`configs/controllers/*.yaml`) — nothing is
hardcoded. The frozen production config is
**`configs/controllers/ae_full_final_nopir_v2_5.yaml`**:

```yaml
initial:  uncertainty 0.30  frustration 0.00  surprise 0.00  confidence 0.60
decay:    uncertainty 0.75  frustration 0.85  surprise 0.35  confidence 0.75

weights:
  uncertainty:  repeated_observation +0.25   budget_ratio +0.10   progress_relief -0.30
  frustration:  invalid_action +0.35  repeated_action +0.30  repeated_observation +0.20
                progress_relief -0.35
  surprise:     unexpected_outcome +0.60     observation_novelty +0.25
  confidence:   progress_signal +0.35  observation_novelty +0.10
                invalid_action -0.25   repeated_action -0.15
```

`progress_signal` appears in every variable's formula — positively for confidence, as a
negative `progress_relief` term for uncertainty and frustration. That is the concrete
implementation of "visible progress should calm the controller down."

**`clip01` bounds every variable to `[0,1]`.** Consequence worth knowing: a hysteresis
`enter: 1.01` is an unreachable threshold, i.e. a surgical per-band off-switch that needs
no code change. That is how config-only band disabling is done.

### 2.3 Layer 3 — Hysteresis and the state signature

Each abnormal band is a **latch**, not a threshold test (`HysteresisTracker`): it activates
on crossing `enter` and deactivates only on crossing `exit`. Production values:

```
uncertainty   enter 0.65 / exit 0.35
frustration   enter 0.70 / exit 0.40
surprise      enter 0.60 / exit 0.30
confidence_low enter 0.35 / exit 0.55   # inverted: entered from above
```

This prevents flicker for values oscillating near one cutoff.

`compute_signature()` then produces a `StateSignature` — **all five booleans at once**
(`uncertainty_high`, `frustration_medium`, `frustration_high`, `surprise_high`,
`confidence_low`).

> **Design history you must not undo.** The first version gated interventions on "the single
> dominant `Mode` changed." That silently missed in-place severity escalation: with
> frustration already latched high, confidence then dropping low left the dominant mode
> still `FRUSTRATED`, so REPLAN could never fire. `Mode` / `current_mode()` still exist but
> are **logging-only projections**. Event detection reads the full signature.

### 2.4 Layer 4 — Intervention selection (`_select_intervention`, `stateful_controller.py:532`)

Three intervention types, ordered `CONTINUE(0) < VERIFY(1) < REFLECT(2) < REPLAN(3)`.
The rule tree, in evaluation order:

1. **REPLAN** — `frustration_high AND confidence_low` **AND**
   `steps_since_meaningful_change >= patch_duration_steps`.
   The sustained-evidence clause matters: `repeated_action` carries weight on both
   frustration (+) and confidence (−), so a single exact repeat can push both bands
   simultaneously within a few steps. That is local-fault evidence for REFLECT, not
   plan-level failure. If not yet sustained, **fall through** to rule 2.
2. **REFLECT** — `invalid_action`, **or** (`frustration_medium` AND
   `consecutive_exact_action_repeat`).
3. **VERIFY** — `uncertainty_high` **or** `surprise_high`.
4. Otherwise `CONTINUE`.

Note each band has exactly one consumer, which is what makes config-only band-disabling a
clean experiment.

### 2.5 Layer 5 — Firing gates

A matched candidate is not enough. Firing additionally requires an **event**:

```
is_event = candidate != CONTINUE and (severity_upgrade or invalid_rising_edge or flag_rising_edge)
```

i.e. a genuinely new or escalating condition, not a condition that has merely persisted.
Then four budget gates (all `3` in production): `warmup_steps`, `cooldown_steps`,
`max_interventions`, `patch_duration_steps`; plus `recovery_grace_steps = 3`.

> **Subtle and previously buggy:** the edge-detection baseline
> (`previous_signature` / `previous_invalid_action_flag` / `last_candidate_severity`) is
> **frozen during warmup** rather than rolled forward. Otherwise a condition active since
> step 1 would have its rising edge silently consumed by suppressed warmup steps and would
> read as "no new event" forever. `self.state` and the hysteresis latches are *not* frozen —
> only the edge bookkeeping.

### 2.6 Layer 6 — Outcome tracking and escalation

The edge-detection rule above only fires on *new* events. But if a REPLAN fires and the
affect state simply stays pinned — the underlying problem never fixed — no further rising
edge ever occurs, so no second intervention could ever fire. A second, orthogonal
mechanism closes that hole:

- Every fired intervention is **pending** until `patch_duration_steps` expires.
- At expiry (and **only** at expiry — never per-step), the outcome is judged by
  `local_state_change_proxy`: did the world change at all while the directive was active?
- **`recovered`** → clear pending, reset `last_candidate_severity = 0` so the same severity
  can re-fire on recurrence without waiting for a full hysteresis exit/re-entry cycle.
- **`unresolved`** → schedule an escalated follow-up (`VERIFY→REFLECT→REPLAN→REPLAN`)
  that fires as soon as cooldown clears, still bounded by `max_interventions`.

There is also a **PIR** path (post-intervention exact repeat): if the model repeats its
previous action verbatim while an intervention is still pending, that is decisive evidence
the directive was ignored, and it bypasses cooldown/grace/patch-expiry entirely.
**PIR is disabled in the production config** (`post_intervention_repeat_enabled: false`) —
see §4.

### 2.7 The directive

```
[ACTIVE CONTROL DIRECTIVE]
<3–4 imperative lines>
```

VERIFY asks for one information-gathering action before changing strategy. REFLECT asks for
a diagnosis and an explicitly different action. REPLAN asks for objective restatement plus
a 2–4 subgoal plan, executing only the next subgoal.

**Injection discipline (important):** the directive is appended to the *ephemeral prompt
string* for exactly one LLM call and is **never written into `history`**
(`alfworld_runs_ae/agents.py:334`). The persistent trajectory therefore stays a clean ReAct
trajectory. If you ever see directive text inside `history`, that is a bug.

---

## 3. ALFWorld experiment design

This section is the full protocol. Everything in it is already encoded in the sbatch
scripts under `/vol/gpudata/jy625-ae-data/scratch_ae3/sbatch_scripts/`; read it before
changing any of them.

### 3.1 Task suite and splits

The benchmark is **ALFWorld** (text-only TextWorld variant), on the **ExpeL 134-task
solvable subset** of `valid_unseen` — `data/alfworld/alfworld_tasks_suffix.json`, taken
from `LeapLabTHU/ExpeL`. This is the same suite ReAct/Reflexion/ReflexGrad report on, which
is what makes the baseline numbers comparable at all.

The 134 tasks are split **by position, not by sampling**: first 100 = `practice`, last 34 =
`exam` (`alfworld_runs_ae/environment.py`, `N_PRACTICE = 100` / `N_EXAM = 34`). The split is
deterministic and has never been reshuffled.

Six task types, unevenly distributed:

| Type | Prefix in gamefile | all 134 | practice | exam |
|---|---|---|---|---|
| `put` | `pick_and_place` | 24 | 18 | 6 |
| `clean` | `pick_clean_then_place` | 31 | 24 | 7 |
| `heat` | `pick_heat_then_place` | 23 | 19 | 4 |
| `cool` | `pick_cool_then_place` | 21 | 16 | 5 |
| `examine` | `look_at_obj` | 18 | 14 | 4 |
| `puttwo` | `pick_two_obj` | 17 | 9 | 8 |

Two consequences worth planning around. **The exam split is small and lopsided** — 4 tasks
in `heat` and `examine` each, but 8 in `puttwo`, which is over-represented relative to its
share of the full set. A per-type exam number is therefore nearly meaningless on its own.
And **`puttwo` is the hardest type**, so exam scores run below practice scores for reasons
that have nothing to do with generalization.

> **The split's original purpose was no-test-set-tuning: tune on practice, report on exam.
> That protocol was explicitly overridden by the author ("带着 exam 一起调").** Every tuned
> number in §6 is fitted on all 134. The split is still worth reporting separately, but it
> can no longer be presented as a held-out result. Any *further* tuning goes back to
> practice-only.

### 3.2 What one episode is

- Backbone: **ReAct** — interleaved `think:` and environment actions, one LLM call per step.
- Horizon: `MAX_STEPS = 50` (`alfworld_runs_ae/agents.py:23`).
- Success: ALFWorld's own `won == True`. Binary, no partial credit — the environment
  exposes none (see §5.2).
- Termination policy: **`legacy_early_stop`** — an exact repeated action ends the episode
  early. Under AE this early stop can be suppressed by the controller's
  `recovery_grace_steps` window, and that suppression is counted and logged
  (`grace_consumed_count`, `suppressed_exhausted_count`, `final_exhausted_count`).

> `fixed_horizon` is the code-level default (`DEFAULT_TERMINATION_POLICY`) and *removes* the
> early stop entirely, which was expected to make the two arms more comparable. It was
> tested and **rejected** (§4) — every production run passes
> `--termination-policy legacy_early_stop` explicitly. Do not drop that flag.

### 3.3 Prompting: strictly one-shot, with a per-type demo mapping

The protocol is **one-shot** and the author has said explicitly not to change it. Demos come
from ALFWorld's standard `alfworld_3prompts.json`, which ships three worked trajectories per
task type. Their quality is not uniform, so the production runs pick a per-type index:

| Group | Task types | Demo config | Demo index |
|---|---|---|---|
| A | `put`, `examine` | `configs/demos/one_shot_v1.yaml` | 0 |
| B | `heat`, `cool`, `clean` | `one_shot_v1_index2` | 2 |
| C | `puttwo` | `one_shot_v1_index1` | 1 |

This is why every run is executed as **three groups × two splits = six sub-runs per repeat**
(`--task-types` selects the group). The six sub-runs are then summed into one 134-task score.

**Why the mapping exists.** `react_cool_0` is malformed — the only defective demo of the 18 —
and it is exactly the one a naive index-0 one-shot serves for `cool`. Measured: index 2 got
17/21 on `cool` where index 0 got 9.8/21. The mapping corrects a demo defect; it is not
controller tuning. Note also that the 80→87/134 jump measured earlier traced **entirely** to
demo choice, and every *controller* change tried on top of it failed to replicate.

> **Any baseline compared against AE must use this identical mapping.** Otherwise the
> comparison measures demo quality, not method. This is why the ADaPT port needed a new
> one-shot executor path, and why Reflexion is run with `--demo-config`.

### 3.4 Model and serving

Qwen3-8B (local weights at `/vol/gpudata/jy625-ae-data/models/Qwen3-8B`), served by vLLM,
`--served-model-name Qwen/Qwen3-8B`. Two clients are built per run
(`run_alfworld.py::build_llms`): an act LLM at `temperature=0` with ALFWorld stop tokens,
and a reflect LLM at `temperature=0, max_tokens=300` (only Reflexion uses the second one).

A **fresh vLLM server per repeat**, on a port derived from the array job id and repeat index,
with a guard that refuses to start if that port is already serving — an accidental attach to
another job's server would silently contaminate both.

### 3.5 "n seeds" means n repeats, not n seeds

`ae/runners/run_alfworld.py:90` calls `random.seed(args.seed)` and `random.` appears
**nowhere else** in `ae/`. Both LLMs are `temperature=0`. Varying `--seed` therefore changes
nothing. Wherever this project says "10 seeds" it means **10 independent repeats, each with
its own freshly launched vLLM server**, `--seed 42` throughout.

**Where the variance actually comes from:** vLLM is nondeterministic at `temperature=0`
because prefix caching and CUDA-graph kernel selection change float reduction order, which
flips argmax on near-tied logits.

That is also why servers are not shared. **Sharing one vLLM across repeats warms the prefix
cache and biases later repeats upward** — measured, Cochran Q *p* = 0.0034. If a fully
deterministic run is ever needed the flags are `--enforce-eager --no-enable-prefix-caching
--max-num-seqs 1`, at a large throughput cost.

### 3.6 The comparison matrix

Every arm runs the same 134 tasks, same model, same one-shot demo mapping, same horizon.

| Arm | Config | Repeats | State |
|---|---|---|---|
| **AE-full** | `ae_full_final_nopir_v2_5.yaml` | 20 | done |
| **ReAct** | same backbone, `controller=None` | 5 planned | 4 done, rep 4 needs re-run |
| **Reflexion** | `--max-trials 4` | 10 planned | smoke test in flight |
| **ADaPT** | one-shot `react` executor | 10 planned | smoke test in flight |
| ReflAct / ReflexGrad | published-anchor reproductions | — | run earlier, on the unsplit 134 (`get_all_tasks`) |

**ReAct is the load-bearing comparison** — it is AE with the controller removed and nothing
else changed, so the difference isolates the controller. Reflexion and ADaPT are the
literature comparison: Reflexion buys its gains with *retries* (up to 4 trials, i.e. up to 4×
the episodes) and extra reflection LLM calls; ADaPT buys its gains with planner calls. AE's
argument is that it spends **neither** — zero extra LLM calls, one episode, no retry (§1).
That cost asymmetry is the point, so report calls-per-task alongside accuracy.

> Anchor reproductions use `get_all_tasks()` (unsplit 134) deliberately — they reproduce
> published results and must not inherit this project's practice/exam protocol, which is not
> part of any published method.

### 3.7 Analysis: paired, never by totals

Across repeats, of the 134 tasks roughly **87 always succeed and 26 always fail** regardless
of method or repeat. All statistical power lives in the ~**21 tasks (15.7%) that flip**.

- Use **McNemar on paired per-task outcomes**. Compare task-by-task, not score-by-score.
- A 2-point aggregate delta is inside the noise band (AE's own sd is ≈2.5) and means nothing
  on its own.
- Report the per-repeat distribution, not just the mean, and never quote a single lucky run
  as the headline (this already happened once — see §6).

### 3.8 Integrity checks built into the runs

These exist because each one caught a real problem. Keep them when you write new scripts.

- **sha256 pin on controller code.** Every job re-hashes `intervention_renderer.py` and
  `stateful_controller.py` and aborts on mismatch, so a run can never silently execute
  edited controller code from a stale scratch copy.
- **Result-directory collision guard.** Aborts if the output dir for this repeat already
  exists, so a resubmit can never half-overwrite an earlier repeat.
- **Port-occupied guard.** See §3.4.
- **Demo-config assertion.** Confirms all three configs are one-shot with the expected
  indices before any GPU work starts.
- **ntfy on exit.** Every job posts its result (or its failure tail) to
  `https://ntfy.sh/AE_new_alert` from an `EXIT` trap, so results arrive without the
  assistant session staying open.

### 3.9 Exact invocation

```bash
python -m ae.runners.run_alfworld \
    --baseline ae_full --split practice \
    --task-types put,examine \
    --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --demo-config configs/demos/one_shot_v1.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    --run-name <name> --seed 42
```

Repeat for `--split exam`, and for groups B (`heat,cool,clean`, `one_shot_v1_index2`) and C
(`puttwo`, `one_shot_v1_index1`). ReAct is the same line with `--baseline react` and no
`--ae-config`.

Per-episode output is JSONL with the full merged step log: the ReAct generation fields and
the AE fields (all signals, affect state before/after, the state signature, the candidate vs.
fired intervention, and the intervention's outcome) advance in lockstep, one record per step.
That log is what any per-step analysis should read.

---

## 4. Decisions already made — do not re-litigate

Each of these cost GPU-days. They are recorded in `audit_reports/` with per-task JSON.

| Decision | Evidence |
|---|---|
| **PIR disabled**; `nopir_v2.5` is the official AE-full config | option-C variant is exploratory only; PIR deprecated and frozen |
| **`fixed_horizon` rejected**, keep `legacy_early_stop` | the "can only help" argument was wrong for independent runs; measured 57 vs 58 |
| **Sustained-only REPLAN rejected** | 3× repeat-verified regression (`examine` collapses). REPLAN stays as specified in §2.4 |
| **REFLECT-v2 prompt rewrite rejected** | the "Nothing happens" fix helped nothing and broke `heat`. Do not retry unconditional or task-type-conditional REFLECT prompt fixes |
| **Demo mapping is measured, not tuned** | 80→87/134 improvement traced entirely to demos; every *controller* change tested on top of it died on repeats |

> **Standing constraint on the numbers.** The author explicitly overrode the
> no-test-set-tuning rule ("带着 exam 一起调"). Every tuned figure below is **fitted on all
> 134 tasks (practice 100 + exam 34)** and is **not** a generalization result. It must be
> labelled that way in the thesis. Any *re*-tuning from here happens on practice only.

---

## 5. Traps that have already burned time

**5.1 `admissible_commands` describes the state the action was chosen *from*.** Validity must
be checked against the list captured **before** the action. `extract()` takes
`admissible_before` and `admissible_after` as separate parameters for exactly this reason.

**5.2 `info` has only three keys.** Reconfirmed empirically 2026-07-25:
`['extra.gamefile', 'won', 'admissible_commands']`. There is **no** inventory, facts, or
structured-state field. Never write a signal that assumes one. True inventory tracking would
need an extra `env.step(['inventory'])` — a real extra environment turn, not a free signal.

**5.3 REFLECT cannot be fully disabled via YAML.** Its `invalid_action` branch reads the raw
signal, not a hysteresis band. Disabling it needs a code change.

**5.4 `/vol/gpudata` is CEPH and its throughput collapses under contention.** Both smoke
tests on 2026-08-20 died with "vLLM not ready" — the first of five 3 GB shards took 6m40s
(~8 MB/s). Measured 235 MB/s the next day. The readiness wait is now **45 minutes** in all
active sbatch scripts. If a job dies during startup, check `vllm_server.log` for shard load
timings before assuming a code fault.

**5.5 HOME has a hard *inode* quota**, and it bites long before the space quota does.
Limits are 12696M space / **61000 files**. The symptom is `Disk quota exceeded` on writing
even a 1 KB file, and **`df` will not show it** — check with `quota -s`. This blocked all
work once, at 61000/61000 files with a 6-day grace. As of 2026-08-21 it has eased to
5795M / 46617 files, but the usual suspects are still there: `.vscode-server/` holds ~42k
regenerable files, and `ae/runners/runs/` holds 300 MB of gitignored run output that belongs
on scratch.

**5.6 Things that live outside the repo and will be lost on a cluster move:**
`/vol/gpudata/jy625-ae-data/scratch_ae3/sbatch_scripts/` (all run scripts), the ADaPT
one-shot patch (`external/` is gitignored), and the WebShop setup files
(`webshop_env.sh`, `smoke_text_env.py`).

---

## 6. Where the numbers stand

**AE-full**, `nopir_v2.5`, one-shot with per-type demo mapping, n = 20 independent repeats,
fresh server each, 134 tasks:

```
95 93 98 92 94 93 92 94 96 92 90 93 90 97 97 95 96 94 94 99
mean 94.2 / 134  (70.3%),  sd ≈ 2.5,  range 90–99
```

An earlier headline of 98.0 came from a single lucky repeat and **has been withdrawn**.

**ReAct**, identical demo mapping, n = 4 (rep 4 died on infrastructure and has not been
re-run):

```
89  92  89  90     mean 90.0 / 134
```

Both figures are fitted-on-134 per §4. The honest current statement is: *AE is ahead of
matched-demo ReAct by roughly 4 points on 134 tasks, on n=20 vs n=4, and the paired
per-task test has not yet been run on the final pair.* Do not upgrade that sentence without
the McNemar result.

Note the old ReAct n=3 numbers used a **different** demo mapping (`clean` on index 0). The
n=5 series **supersedes** them; it does not extend them. Do not pool.

---

## 7. What is in flight right now

| Item | State |
|---|---|
| Reflexion smoke, 12 tasks | job **276240**, running on gpuvm33 since 2026-08-21 01:44 |
| ADaPT smoke, 12 tasks | job **276241**, pending (waiting on a GPU) |
| ReAct rep 4 re-run | `run_react_n5_redo4.sbatch` written, **not submitted** |
| Text-only-signals ablation | config + script written and self-checking, **not submitted** |
| Push to GitHub | commit `12fafc1` made locally; **push blocked, no GitHub credentials on this machine** (`~/.ssh/id_rsa.pub` is not registered on the account) |
| WebShop | env works end-to-end on the 1000-product subset; full 5.48 GB catalogue downloaded and sha-verified; **Lucene index not built** |

**Why these two smokes exist.** Reflexion and ADaPT are being added as literature baselines:
both on ALFWorld, both Qwen3-8B, both one-shot with the §3.3 demo mapping, **10 repeats
each**. Committing 10 × 134 tasks to an untested code path is the expensive way to find a
bug, so 12 tasks run first. Once they pass, the full runs go out as **5 array tasks × 2
repeats** (the 6-job QOS cap forbids 10 in parallel).

Both smokes report to ntfy on exit and write a `RESULT.txt`. Neither needs anyone watching.

### 7.1 Reflexion smoke — job 276240

- Script: `scripts/slurm/ae3_runs/run_reflexion_smoke.sbatch`
- Logs: `/vol/gpudata/jy625-ae-data/logs/reflex_smoke_276240/` → `batch.log`, `run.log`,
  `vllm_server.log`, `RESULT.txt`
- Episode log: `<COPY>/ae/runners/runs/reflexion_smoke_276240/episode_log.jsonl`

`--baseline reflexion` is wired into `ae/runners/run_alfworld.py` and
`ae/baselines/reflexion.py` exists, but **no sbatch in this project had ever used it** —
every ALFWorld run so far was `react` or `ae_full`. That path is what is being tested.

Runs `--limit 12 --max-trials 4 --demo-config configs/demos/one_shot_v1.yaml`. Demo handling
is shared with ReAct (`agents.py::_build_base_prompt` pulls `react_{task_type}_{idx}` driven
by `--demo-config`, identically for both), so passing the same config gives byte-identical
one-shot examples — nothing extra to align.

**Three assertions, any of which fails the job:**

1. exactly 12 episodes,
2. `any(trials_used > 1)` — otherwise the reflexion retry loop never engaged at all,
3. at least one episode produced reflection text.

Pass looks like `SMOKE TEST PASSED` in `RESULT.txt`. **The score itself is not an assertion**
— 12 tasks says nothing about accuracy. This test only answers "does the machinery run."

### 7.2 ADaPT smoke — job 276241

- Script: `scripts/slurm/ae3_runs/run_adapt_smoke.sbatch`
- Logs: `/vol/gpudata/jy625-ae-data/logs/adapt_smoke_276241/`
- Result JSON: `external/ADaPT/results/comparison/Qwen/Qwen3-8B/smoke_oneshot_react_276241_Qwen/Qwen3-8B.json`

Runs `--executor react --react-type oneshot --num-task-samples 2` (2 per type × 6 types = 12;
ADaPT does `random.seed(0)` before sampling, so the 12 are deterministic).

**Context that makes this test necessary.** ADaPT has already been run here at full scale and
scored **8/134 (6.0%)** — all 8 wins in `examine`, and 0 in each of the other five types. The
cause was diagnosed: `plan_llm()` never once emitted the required `Step N:` /
`Execution Order:` format, so `plan_to_args()` always parsed an empty plan and every
multi-stage task failed at depth 1 with **no decomposition happening at all**. The 8 wins were
single-shot atomic-executor hits, not ADaPT recursion.

That run used the default `atomic` executor. This one uses the one-shot `react` executor,
which is a materially different configuration, so the 6% figure does not predict it.

**How to read the result — this is the part that matters.** The script prints:

```
tasks with a non-empty plan: N/12
```

- **N = 0** → the planner is *still* emitting nothing. The old bug is not fixed and the
  number that follows is meaningless as a measurement of ADaPT. Do not run 10 repeats.
- **N ≈ 12** → decomposition is genuinely happening. Whatever score comes out is then a
  real measurement of the method at this model scale, low or not, and the 10 repeats are
  worth spending.

This is the whole point of the smoke: **distinguish "our port is broken" from "the method is
genuinely weak with an 8B model."** Only the second is publishable as a baseline. Note the
planner-format fix described in `external/_provenance/ADaPT/SOURCE.md` reached 18/18
non-empty plans in an isolated planner-only A/B, so N=0 here would be a surprise worth
investigating rather than an expected outcome.

### 7.3 If a smoke dies during startup

Check `vllm_server.log` for shard load timings **before** suspecting the code. Both of these
jobs already died once (276093 / 276094 on 2026-08-20) purely because `/vol/gpudata` was
thrashing — the first of five 3 GB shards took 6m40s, about 8 MB/s, against 235 MB/s
measured the next day. Nothing was wrong with either script. The readiness wait was raised
from 20 to **45 minutes** in response; that is the current value in all four active scripts.

### 7.4 Demo-pool provenance for ADaPT

ADaPT's bundled ALFWorld demos were checked against ours: all 18 are our demos **verbatim**,
plus one trailing `'\n> think: Task completed!'` line its executor protocol requires. So
"demos consistent with ReAct" is a checked statement, not an assumption. The one-shot
mapping affects the **executor** only — ADaPT's planner prompts are intrinsic to the method,
the same way Reflexion's reflection template is, and are deliberately left alone.

---

## 8. Direction

**Immediate:** land the two smoke tests, then the 10-repeat Reflexion and ADaPT runs; re-run
ReAct rep 4 so the ReAct baseline is a clean n=5; run the paired McNemar of AE vs matched-demo
ReAct on the final numbers.

**Then: generality.** The strongest objection to AE is that hand-designed signals will not
transfer, making the whole thing an ALFWorld-specific hack. The code answer is that **only
signal extraction is environment-coupled — roughly 150 of ~1550 controller lines.** Layers
2–6 (affect update, hysteresis, rule tree, gating, outcome tracking) touch nothing
environment-specific. `extract()` already receives admissible actions as parameters, so a
WebShop port is an adapter plus wiring `env.get_available_actions()`, not a redesign.

Two pieces of evidence are planned for that argument: the WebShop port itself, and a
text-only-signals variant that zeroes the five environment-coupled weights
(`configs/controllers/ae_ablation_text_only_signals.yaml`) to show how much of AE's gain
survives on env-agnostic signals alone.

> Honest caveat already written into that config's header, and it should stay in the
> write-up: `progress_signal` remains internally gated on `invalid_action`
> (`signals.py:315`), so the claim is "affect state driven only by env-agnostic signals,"
> **not** "no environment-specific code executed."

**WebShop specifics** when you get there: 1.18M products, 12,087 instructions, episodes
capped at **15 steps**, reward is **continuous 0–1** (Reflexion counts only `reward == 1.0`
as success). The published methods subset *instructions* (100 for ADaPT and Reflexion, 500
for ReAct), not products. Source `/vol/gpudata/jy625-ae-data/webshop_official/webshop_env.sh`
before touching anything — it needs a private conda 3.8 env and a JVM on `PATH`. `gdown` is
dead for the data; use the HF mirror `YWZBrandon/webshop-data`.

---

## 9. Working conventions

- **Reply to the author in 中文.** Code, commits, and documents stay English.
- The author frequently asks for plain-language explanations. Prefer concrete analogies over
  jargon; "没太看懂" means back up and re-explain, not add detail.
- Do not use the structured question widget — ask decisions as plain chat text.
- Long-running jobs must self-report to ntfy. The author's session window closes on its own,
  so anything that only reports back through the assistant is effectively lost.
