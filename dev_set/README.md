# dev60 — independent tuning dev set (valid_seen, 60 tasks)

Built 2026-08-05. Purpose: parameter-sensitivity tuning dev set, strictly
separate from the official 134-task valid_unseen evaluation set. See
[[project-ae-pir-ablation-decision]]-adjacent memory / this session for
context — this is a NEW artifact, not related to the PIR ablation round.

## Files

- `dev60_valid_seen_manifest.json` — the 60-task manifest actually used for
  the 7-config sweep. Fields per task: `dev_idx` (1-60, fixed order),
  `task_type`, `split` (always `"valid_seen"`), `task_dir`, `trial_dir`,
  `gamefile` (absolute path), `gamefile_sha256`, `goal_text` (exact string
  shown to the agent, extracted from the actual `env.reset()` observation,
  not reconstructed).
- `dev60_candidate_pool_and_backups.json` — full provenance: the 134 used
  task ids (for reference), the 13 reserved-but-unused extra valid_unseen
  solvable trials (NOT used this round), the full 168-trial valid_seen
  solvable candidate pool, the per-type selection, and the per-type unused
  backup list (fixed order, for any future preflight substitution needs).

## Data provenance and scope

- **Official 134-task set** (`data/alfworld/alfworld_tasks_suffix.json`):
  all from `json_2.1.1/valid_unseen`. Remains the frozen final evaluation
  set — untouched, not read from for sampling beyond exclusion-checking.
- **13 extra valid_unseen solvable trials**: found while auditing
  valid_unseen for spare capacity (see conversation). Reserved, unused this
  round per explicit instruction.
- **valid_seen candidate pool (168 solvable trials)**: `json_2.1.1/valid_seen`
  has 251 total trial directories; 186 were run through
  `alfworld-generate --expert_type planner` (65 were filtered out upstream
  by the tool itself — `movable`/`Sliced` trajectories or task types outside
  the 6 in scope); of those 186, 168 were confirmed solvable by the PDDL
  planner. This 168-trial pool is the dev60 sampling universe.
- `valid_train` was checked (199 solvable trials found) but **not used** —
  explicit instruction this round is valid_seen only.

All newly-compiled `game.tw-pddl` files live under
`/vol/gpudata/jy625-ae-data/alfworld_data_devset_gen/` — a directory
entirely separate from `/vol/gpudata/jy625-ae-data/alfworld_data/` (the
original, untouched data root). The official 134 task's source files were
never written to, overwritten, or regenerated in place.

## Compilation vs. task modification

`alfworld-generate` repackages each trial's existing `initial_state.pddl`
(the actual task: object placement/init state) and `traj_data.json`
(task_type/target object/receptacle metadata) into `game.tw-pddl`, and runs
a PDDL planner to verify solvability + record a walkthrough. Verified by
regenerating one of the official 134's own game.tw-pddl into a scratch
location and diffing fields against the original: `pddl_domain`,
`pddl_problem` (= the task/initial state itself), `solvable`, and
`walkthrough` were byte-identical; only the `grammar` field differed, and
only in its goal-sentence template choice (a same-task, same-object
different paraphrase, e.g. "look at bowl under the desklamp" vs "examine
the bowl with the desklamp" — both ExpeL/ALFRED's own human-annotated
synonyms for the identical `look_at_obj_in_light` task on the identical
object) plus a cosmetic reformatting of the (never-invoked) `help` command's
grammar rule. No task goal, initial state, or solvability was altered by
this tooling.

## Sampling procedure (seed=42, fully deterministic, no manual curation)

For each of the 6 task types (processed in this fixed alphabetical order:
clean, cool, examine, heat, put, puttwo), against ONE shared
`random.Random(42)` instance:

1. Take that type's full valid_seen solvable candidate pool, sorted
   canonically by `(task_dir, trial_dir)`.
2. `rng.shuffle()` it in place (consumes the shared RNG stream — this is
   why type-processing order is fixed and documented, since it determines
   how the RNG stream is partitioned across types).
3. Walk the shuffled list, taking each trial only if its `task_dir` (task
   family — e.g. `pick_clean_then_place_in_recep-SoapBar-None-Drawer-423`)
   hasn't already been selected for this type, until 10 distinct-family
   trials are collected (all 6 types had ≥13 distinct families available,
   so 10-distinct-families was always reachable without needing the
   same-family fallback pass).
4. The remaining shuffled trials (not selected) become that type's
   fixed-order backup queue, for preflight-failure substitution.

No difficulty/performance-based selection at any step — the only
non-random filter is the family-dedup constraint in step 3, which is
structural (diversity), not outcome-based.

## Preflight

All 60 selected tasks were loaded via the actual production loading path
(`environment.make_single_task_env` + `env.reset()`, same call used by
`ae/runners/run_alfworld.py`) and checked for a non-empty observation +
non-empty admissible-commands list. **0/60 failed** — no backup
substitution was needed this round. (The backup queues are still saved and
documented in case a future re-run under different infra ever needs them.)

## Verification performed

- Trial-level, game-ID-level (gamefile path), and content-level (SHA256)
  dedup: all 60 selected tasks are pairwise distinct on every axis.
- Task-family-level dedup: 0 duplicate `task_dir` within any type (10/10
  distinct families in every type).
- Cross-check against the official 134-task set: 0 overlap.
- `goal_text` for a sample of tasks spot-checked against `task_dir` to
  confirm the loader returned the specific requested game, not a fallback
  default.

## Freeze

`dev60_valid_seen_manifest.json` SHA256:
`8e96ddfaaee64c65c16df42b542ef71d3681922418f7f9b6c5b28527fff635eb`

This manifest and task order are what the 7-config parameter sweep uses.
Do not regenerate or resample this manifest for the sweep — any future
need for a different dev set is a new, explicitly-requested artifact, not
a silent replacement of this one.
