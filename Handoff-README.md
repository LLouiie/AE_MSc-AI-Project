# AE (Intra-Episode Appraisal Controller) — Handoff README

Written 2026-07-25 by Claude Code, for whichever assistant (human or AI) picks
this up next. The working copy of this repo currently lives at
`~/projects/reflexion-intra-episode` on an Imperial College HPC cluster, but
is being moved elsewhere because that cluster's shared storage (RDS) has
started intermittently failing writes (`ENOSPC`) even though the user's own
quota has plenty of headroom — see "Known infrastructure issue" below. If
you're reading this from a fresh clone/new GitHub repo, that move already
happened; treat this file as the map.

## 1. What this project actually is (read this first)

This is a thesis project. It went through a **direction pivot mid-project**:

**Old direction (now frozen, not deleted):** cross-task experience learning —
run an agent on "practice" tasks, extract rules from successful/failed
trajectories into a Rule Library (ExpeL-style ADD/EDIT/REMOVE/AGREE
consolidation), retrieve rules on later tasks, and study whether an "affect
signal" should gate when rules get consolidated. This is **no longer the
thesis's core contribution** but the code is preserved (see section 4).

**New direction (what you should be working on):** within a *single* episode,
can a multidimensional appraisal/affect state — computed from observable
trajectory signals, not LLM self-report — detect *what kind* of problem the
agent has hit, and route to a different intervention accordingly?

Research question:
> Can a multidimensional appraisal controller improve within-episode
> intervention timing and selection compared with scalar progress gating,
> fixed rules, and LLM self-reflection?

Core hypothesis: **progress is not equivalent to epistemic reliability.** Two
distinct failure modes need distinguishing:
- **Strategic stagnation** ("frustration"): no progress, repeated actions,
  tool failures, plan clearly not working.
- **Epistemic unreliability** ("uncertainty"): agent *looks* like it's making
  progress (new observations, more steps) but the evidence is thin,
  conflicting, or pointing at the wrong entity — confidently wrong, not
  visibly stuck.

State variables (names may evolve): `frustration`, `uncertainty`, `surprise`
(prediction/observation mismatch), `confidence`/`recovery`. Each accumulates
and decays over steps (not a per-step independent yes/no judgment):

```python
frustration_t = decay_f * frustration_prev
    + w_repeat * repeated_action + w_failure * tool_failure
    + w_no_progress * no_progress - w_recovery * recovered_progress
uncertainty_t = decay_u * uncertainty_prev
    + w_conflict * evidence_conflict + w_disagreement * answer_disagreement
    + w_gap * evidence_gap - w_consistency * evidence_consistency
```

All weights/thresholds/decays/cooldowns go in config files, never hardcoded,
**never tuned on the test set.**

Controller output — action space:
- `CONTINUE`: no extra cost.
- `VERIFY`: re-check evidence / verify a key assumption (this must exist in
  the final version — a `CONTINUE`/`REPLAN`-only pilot is fine as a first
  smoke test, but shipping without `VERIFY` collapses this into "just a
  fancier scalar progress gate," which is explicitly *not* the contribution).
- `REPLAN`: abandon the current plan, regenerate one.

Routing sketch (not yet implemented in code):
```
high frustration + sustained low progress -> REPLAN
high uncertainty or evidence conflict     -> VERIFY
high surprise                              -> VERIFY, escalate to REPLAN if needed
otherwise                                  -> CONTINUE
```
Plus: cooldown, hysteresis, partial reset after intervention, fast decay
after recovery, per-episode intervention budget.

**Explicit anti-goal:** do not let this collapse into "ReflexGrad but with
extra steps." ReflexGrad (baseline #5 below) already does scalar-progress ->
replan. The AE contribution has to be the *multidimensional, distinguishing
strategic-stagnation-from-epistemic-unreliability* part, expressed as a claim
like:

> We introduce a multidimensional appraisal controller that distinguishes
> strategic stagnation from epistemic uncertainty and routes agents to
> different within-episode interventions.

not:

> We use artificial emotion to decide when to reflect.

## 2. Baseline set (final, do not expand without the user's say-so)

Implementing exactly six things, in this priority order:

1. **ReAct** (Yao et al. 2022, arXiv:2210.03629) — no meta-controller.
2. **Reflexion** (Shinn et al. 2023, arXiv:2303.11366) — post-episode
   reflect-and-retry.
3. **ReflexGrad** (arXiv:2511.14584 **v4** — must be v4, not earlier
   versions) — the closest/most important comparator. Four mechanisms:
   hierarchical TODO planning, scalar progress evaluator (`E(o,a,o',τ) ∈
   [0,10]`, LLM-judged), fast TextGrad-style refinement every `k=3` steps,
   slow causal reflection after `m=5` consecutive scores below `θ_low=4`
   (then `cooldown=5` steps where both fast/slow are suppressed). Fully
   episode-local, paper states no cross-episode learning. Official repo:
   github.com/qpiai/reflexgrad, entry `main.py`. ALFWorld full-system result
   on Qwen-3-8B: 75.4%±2.2 (134 tasks, 10 seeds).
4. **ADaPT** (Prasad et al., arXiv:2311.05772, NAACL 2024 Findings) —
   failure-triggered recursive decomposition: executor self-reports
   success/failure, planner decomposes into 3-5 subtasks with And/Or logic on
   failure, recursion capped at `d_max=3` for ALFWorld. Official repo:
   github.com/archiki/ADaPT (`run_alfworld.py`), docs incomplete per the
   author, read the code not just the README.
5. **ReflAct** (Kim et al., EMNLP 2025, arXiv:2505.15182) — replaces ReAct's
   `Thought` (predict next action) with `Reflection` (state-vs-goal gap,
   *then* decide action), every step, not just on failure. **No official
   code repository exists** — confirmed by direct search, and the
   WeiminXiong/MPO repo the paper borrows its ALFWorld environment/baseline
   code from is *not* an official ReflAct implementation; don't call it one.
   Any implementation here must be labeled "paper-based reproduction," not
   "official repository reproduction." Example contrast from the paper:
   ```
   ReAct:    Thought: Now I find a spraybottle 2. Next, I need to take it.
   ReflAct:  Reflection: Currently, I am at cabinet 2 and have found a
             spraybottle 2, which brings me closer to completing the task of
             placing it on the toilet.
   ```
   Full ICL demonstrations are in the paper's Appendix K (not reprinted in
   the HTML body I could fetch — go get the appendix before implementing).
6. **AE** (this project's own method, section 1).

Explicitly **out of scope** (do not implement, do not scaffold elaborate
abstractions in anticipation of them): DuSAR, LATS, ExpeL-as-baseline, and
every "controlled controller baseline" from the original spec (Fixed
Interval, Event Trigger, Random Budget-Matched, Uncertainty-Only, Scalar
Progress Gate as a standalone baseline, LLM Self-Trigger). The user cut these
explicitly to avoid over-engineering before the core 6 exist and work.

Full research findings (paper summaries, repo structure, dependency/effort
table, recommended implementation order, exact-vs-adapted-reproduction
classification for each) are in `BASELINE_RESEARCH.md`, which should travel
with this repo — if it's not already in the new location, it was last written
to
`/rds/general/ephemeral/.../scratchpad/BASELINE_RESEARCH.md` on the old
machine and should be recovered/re-sent to the user.

Recommended implementation order (my recommendation, given to the user, they
have not overridden it): ReAct/Reflexion (verify+wire, cheap) -> ReflexGrad
(highest priority new baseline, most work) -> ADaPT (official repo exists,
lower risk) -> ReflAct (no code at all, highest uncertainty, do last).

## 3. Repo/branch/tag state (as of the storage failure)

Three separate physical locations exist. Two are on the HPC filesystem that's
currently having storage problems — **do not assume these paths still exist
or are still healthy** once you're working from a new location:

| Path | GitHub remote | Status |
|---|---|---|
| `~/projects/reflexion` | `github.com/LLouiie/AE` | **Original repo. Its `.git` started throwing `ENOSPC` on internal metadata writes (append to `.git/logs/HEAD`, rewrite `.git/config`) even though general home-directory writes worked. Treated as a read-only backup, not touched further. Do not try to repair its `.git` by hand.** |
| `~/projects/reflexion-intra-episode` | `github.com/LLouiie/AE` (same remote as above — it's a fresh `git clone` of the same repo) | **The actual working copy for this project.** Branch `feature/intra-episode-ae`, created from tag `legacy/rule-library-2026-07-25` (which is pushed to `LLouiie/AE`). This is the repo the user is about to move to a brand-new GitHub repo, since it's still logically the same remote as the confusing old one. |
| `~/projects/AE` | `github.com/LLouiie/AE2` | Separate, unrelated-by-name-collision-only repo. Contains the **official** ExpeL/ReAct/Reflexion implementations that produced the thesis's *existing, already-reported* HotpotQA/FEVER numbers. Not part of the new AE direction's codebase, but its result data must not be deleted (see section 6). |

Tag `legacy/rule-library-2026-07-25` marks the last commit before the pivot —
it's the frozen snapshot of the Rule Library / consolidation-scheduling
scaffolding, including a scheduler comment that literally anticipated this
pivot:
```python
# schedulers.py, Scheduler.decide() docstring:
#   confidence, disagreement, frustration   <- "# Phase 2 预留" (reserved),
#   never implemented — direct ancestor of the new appraisal controller idea,
#   just intended for "when to consolidate rules" rather than "when to
#   intervene within an episode."
```

**When you set up the new GitHub repo:** push both the tag and the
`feature/intra-episode-ae` branch (or just push everything and rename
`feature/intra-episode-ae` to `main` there if the old `main`/legacy history
isn't wanted in the new repo — that's the user's call, not decided yet).

## 4. What's implemented in `ae/` so far (commit `3c2bc7a` on
`feature/intra-episode-ae`)

```
ae/
  core.py             StepContext, AppraisalState, InterventionType,
                       InterventionDecision, MetaController Protocol.
                       Every baseline and the AE controller should speak
                       this interface so cost/intervention accounting is
                       directly comparable across baselines.
  logging_utils.py     snapshot_config() / load_done_ids() / JsonlLogger —
                       same JSONL-append + config.json-snapshot pattern as
                       the pre-existing hotpotqa_runs/run_episode.py, reused
                       so every ae/ run directory looks the same.
  llm_client.py         Re-exports hotpotqa_runs/llm.py's AnyOpenAILLM +
                        call_counter via sys.path, so every ae/ module shares
                        one vLLM client instead of each file doing its own
                        sys.path hack (which is what the pre-pivot
                        alfworld_runs_ae/agents.py does).
  baselines/
    react.py            Thin wrapper around alfworld_runs_ae/agents.py's
                        ALFWorldAgent (unmodified). IMPLEMENTED, unit-level
                        verified, not yet run against live vLLM.
    reflexion.py         Thin wrapper around ALFWorldReflectAgent, with
                        rules_text pinned to "" so it never imports
                        consolidation.py/schedulers.py. IMPLEMENTED, same
                        verification status as react.py.
  runners/
    run_alfworld.py      CLI entry: `python -m ae.runners.run_alfworld
                        --baseline react|reflexion --split practice|exam
                        --limit N --run-name X [--seed --max-trials --model
                        --base-url --output-dir]`. Dispatches to
                        ae/baselines/*; raises a clear SystemExit for
                        adapt/reflact/reflexgrad/ae (scaffolded as CLI
                        choices, not implemented — see NotImplementedError
                        message pointing at BASELINE_RESEARCH.md).
```

Adapt/reflact/reflexgrad/ae baselines: **not started.** Only the CLI choice
strings exist as placeholders in `run_alfworld.py`.

Two bugfixes landed in *existing* (non-new-direction) code as part of
getting this far, both documented in the commit message of `3c2bc7a`:

1. `alfworld_runs_ae/environment.py::make_alfworld_env` — alfworld>=0.4 (the
   `reflexion_hotpot` conda env) no longer re-exports `Alfred*Env` classes at
   the `environment` package level the way alfworld==0.3.5 (`alfworld035`
   conda env) does. Added a fallback that imports the per-class submodule
   directly (`alfworld.agents.environment.alfred_tw_env.AlfredTWEnv` etc.),
   so `make_alfworld_env()` now works unpinned across both conda envs.
2. **Latent bug found in the legacy (frozen) `run_practice.py`**: ALFWorld
   task rows in `alfworld_tasks_suffix.json` only have `{"goal", "gamefile"}`
   — there is no `"env_name"` key. The old `run_practice.py`'s
   `task.get("env_name", f"task_{qi}")` fallback was *silently* firing on
   every single task, which meant `get_task_type()` never matched any of the
   `PREFIXES` dict entries and every task defaulted to `task_type="put"` —
   i.e. **the few-shot prompt selection has been wrong for every non-"put"
   task this entire pipeline's history.** Not fixed in the frozen legacy
   file (intentionally, it's frozen), but fixed in `ae/runners/run_alfworld.py`
   by deriving the task id from `gamefile.split("/")[-3]` instead.

## 5. ALFWorld dataset — provenance, already fact-checked

`AE/data/alfworld/alfworld_tasks_suffix.json`, 134 tasks, split
first-100=practice / last-34=exam (this practice/exam split is itself a
holdover from the old direction's protocol and may not matter for the new
one, which doesn't do cross-task learning — worth reconsidering whether
practice/exam splitting even makes sense now, nobody has revisited this).

I verified by direct set comparison (not just trusting a docstring) that
**these 134 tasks are exactly the full official ALFWorld `valid_unseen` test
split** (Shridhar et al., ICLR 2021) — not a hand-curated "confirmed
solvable" subset as `environment.py`'s docstring claims. The docstring's
"134 solvable valid_unseen tasks" comment is inaccurate; correct it if you
touch that file. Every game file under
`$ALFWORLD_DATA/json_2.1.1/valid_unseen/` (134 of them) is present in the
json, and vice versa — confirmed as an exact set equality, not just a
matching count. This also happens to be the same 134-task set ReflexGrad's
paper reports numbers on, and effectively every ALFWorld LLM-agent paper
(ReAct, Reflexion, ExpeL, ADaPT if it uses ALFWorld's held-out split,
ReflexGrad, ReflAct) uses this same standard split, because it's the
benchmark's own fixed test set, not something any of these papers curated
themselves. **Correct citation for the dataset section is Shridhar et al.
2021 (the ALFWorld paper), not ExpeL and not ReflexGrad** — they're all just
using the same standard benchmark.

## 6. Existing thesis results — do not touch, do not regenerate

These already exist, are correct/final, and must not be deleted or
overwritten by anything in the new direction's work:

- `AE/baselines/reflexion/hotpotqa_runs/runs/reflexion_wiki_qwen32b_sample100/`
  — the actual Reflexion HotpotQA result the thesis currently reports.
  5 trials, Qwen2.5-32B-Instruct, final EM=51% (trial 5: correct=51,
  incorrect=29, halted=20). This is the **official langchain-based Reflexion
  repo fork**, not the same code as `ae/baselines/reflexion.py` above (which
  wraps a from-scratch reimplementation living in the reflexion-intra-episode
  repo). If a new-direction Reflexion ALFWorld/HotpotQA number gets reported
  later, it must be labeled as coming from a *different* implementation than
  this existing HotpotQA number — don't imply they're the same codebase.
- `AE/baselines/ReAct/runs/react_wiki_qwen32b_sample100/`,
  `react_fever_qwen32b_sample100/` — existing ReAct HotpotQA/FEVER results,
  Qwen2.5-32B.
- `AE/baselines/ExpeL/logs/{hotpotqa,fever}/expel/` — a partially-completed
  multi-seed ExpeL variance study (seeds 42/1/7/123/2024). **Abandoned by
  explicit user instruction** ("之前那个实验没跑的就算了") — do not resume
  it. What exists: HotpotQA seeds 42/1/7/123 fully evaluated (~39-40% EM
  each, consistent), seed 2024 only 78/100 tasks done (walltime-killed,
  incomplete, do not treat as valid). FEVER: never actually ran for any of
  the new seeds (a PBS job dependency bug meant it silently never started).
  This whole multi-seed thread is closed; the ExpeL vs Reflexion HotpotQA
  comparison already written up found ExpeL's ~40% single-pass score is
  actually *consistent* with the ExpeL paper's own reported comparison point
  (paper: ExpeL 39% vs Reflexion-R3 40%, i.e. "roughly matches," not "ExpeL
  loses badly") — the apparent gap against Reflexion's 51% is because that
  51% is Reflexion's *5-trial final* number, not a same-methodology
  comparison. If this needs restating in the thesis, that reasoning is
  already worked out, just needs writing up.

## 7. Known infrastructure issue (may or may not follow you to the new
location)

Starting ~2026-07-25 07:00 BST, the original HPC's shared RDS filesystem
began throwing `No space left on device` on write operations — first
isolated to `~/projects/reflexion/.git`'s internal files (`logs/HEAD`,
`config`), then spreading to completely unrelated paths (a conda env's
`pip install`, a plain text file in `$HOME`). Confirmed **not** a personal
quota problem: `quota -s` showed only 68% of a 1TB home quota and 9% of a
10M-file quota in use throughout. The `rds` filesystem itself
(`df -h`) was at 92% (13PB/14PB) — most likely a cluster-wide near-capacity
situation (shared GPFS storage pool, not per-user), where individual quota
having headroom doesn't help if the underlying physical pool/OST is out of
allocatable blocks. This is why the user is moving off this machine's
storage rather than waiting for HPC support. If you're on a fresh
environment now, this problem is probably not relevant to you, but if
`ENOSPC` errors show up again unexpectedly, don't assume it's a code bug —
check `df -h`, `df -i`, and `quota -s` (or local equivalents) before
debugging application logic.

Consequence of hitting this mid-work: a PBS smoke-test job (compute job ID
`3427094`, 3-task react+reflexion ALFWorld run on Qwen2.5-32B/2×L40S) was
submitted, then cancelled to switch to a faster/smaller Qwen2.5-7B-Instruct
single-GPU config for quicker queue turnaround — but the PBS script edit for
that swap failed mid-way due to this storage issue and was **not
completed or resubmitted**. `~/ae_alfworld_smoke.pbs` on the old machine is
in a half-edited state (original 32B/2-GPU version, since the edit that would
have changed it to 7B/1-GPU failed and left the original untouched — Edit
tools use atomic temp-file+rename, so no corruption, just "the edit never
landed"). **Nothing has been smoke-tested against a live vLLM server yet.**
This is the actual next step once you're set up somewhere stable: get
`ae.runners.run_alfworld --baseline react` and `--baseline reflexion` running
end-to-end on ~3 ALFWorld tasks with a real model before writing any more
baseline code.

Cached local models available on the old HPC (for reference, may not be
relevant on new infra): Qwen2.5-32B-Instruct (used for all existing thesis
results), Qwen2.5-14B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct,
Qwen3-32B. Two relevant conda envs existed: `reflexion_hotpot` (alfworld
0.4.2 + modern openai>=1.0 — the one that actually works with this repo's
`llm.py`) and `alfworld035` (alfworld 0.3.5, but stuck on ancient
openai==0.28.1 which is incompatible with `llm.py`'s `from openai import
OpenAI` — an attempted `pip install -U openai` in this env failed partway
through due to the storage issue; it's not confirmed broken, just not
upgraded, so treat `alfworld035` as **not currently usable** for this
project until someone re-attempts that upgrade under working storage.

## 8. Immediate next steps, in order

1. Get this repo onto stable storage / the new GitHub repo (in progress, the
   reason this file exists).
2. Re-verify the environment on wherever you land: alfworld installed, a
   conda/venv with modern `openai>=1.0`, `$ALFWORLD_DATA` pointing at
   downloaded ALFWorld game data (`alfworld-download`), and
   `AE/data/alfworld/alfworld_tasks_suffix.json` present (or re-download it
   from github.com/LeapLabTHU/ExpeL/blob/main/data/alfworld/alfworld_tasks_suffix.json
   — see section 5, it's just the standard valid_unseen split, nothing
   proprietary).
3. Stand up a local/served LLM endpoint (vLLM or otherwise) and actually run
   the smoke test: `python -m ae.runners.run_alfworld --baseline react
   --split practice --limit 3 --run-name smoke3_react`, then the same with
   `--baseline reflexion --max-trials 2`. Confirm both produce sane
   `episode_log.jsonl` output before writing more code.
4. Once react/reflexion are confirmed working end-to-end, move to ReflexGrad
   (highest priority remaining baseline) — read `reflexgrad_core_v12.py`,
   `reflexgrad_trial.py`, `dynamic_prompting.py`, `generate_reflections.py`
   from github.com/qpiai/reflexgrad in full before writing any adapter code,
   per the user's explicit "read papers/repos first, don't guess" standing
   instruction.
5. Then ADaPT, then ReflAct, then finally the AE controller itself (section 1).
6. Do not implement DuSAR/LATS/controlled-baselines/ExpeL-as-new-baseline
   unless the user explicitly reopens that scope.

## 9. Standing constraints from the user (apply throughout)

- Never tune AE's weights/thresholds on the test set.
- Every baseline must share the same `VERIFY`/`REPLAN` operator
  implementation and the same prompt — don't let each baseline invent its own
  wording, or comparisons become meaningless.
- Every trigger/intervention must log its reason and the state values that
  caused it (`InterventionDecision.reason` / `.scores` in `ae/core.py` exist
  for this).
- All experiments need fixed-seed + resume support (the JSONL-append +
  `done`-set checkpoint pattern in `logging_utils.py` already does this,
  reuse it, don't reinvent per-baseline).
- Adapted-vs-exact reproduction must be labeled honestly per baseline (see
  section 2's classification) — never describe an adaptation as an "official"
  reproduction.
- Git commit messages / PR text: **English**, even though day-to-day
  conversation with the user is in Chinese (their standing preference).
- Don't ask the user to re-explain context that's already in this file or in
  `BASELINE_RESEARCH.md` — read those first.
