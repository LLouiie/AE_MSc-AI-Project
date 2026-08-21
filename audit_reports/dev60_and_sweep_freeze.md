# dev60 tuning round — freeze record

Frozen 2026-08-05, immediately before starting the 7-config single-parameter
sensitivity sweep. Baseline code is `ae_full_nopir_v2_5_frozen` (see
`audit_reports/ae_full_nopir_v2_5_frozen.md` — same git HEAD, same code
SHA256s, unchanged since that freeze).

## Git state (unchanged since the nopir_v2.5 freeze)

- Branch: `feature/intra-episode-ae`
- HEAD: `1d4bb66b22eeb26bad0cc9741f0997cd8a663536`
- Full patch of tracked-file changes:
  `/vol/gpudata/jy625-ae-data/logs/freeze_devset_round_tracked.patch`
  (2237 lines, identical to the prior freeze's patch — no tracked-file
  content changed between the two freezes, only new untracked `dev_set/`
  and `audit_reports/` files were added)

## dev_set/ files (new this round)

```
8e96ddfaaee64c65c16df42b542ef71d3681922418f7f9b6c5b28527fff635eb  dev_set/dev60_valid_seen_manifest.json
19a9e01d2c01fd74cca7343a670ceba50388e41ac92a1a3103146253c512cedb  dev_set/dev60_candidate_pool_and_backups.json
a6ff273525cdde7563270ddf46156fb4c1fb4e4b9dc54cb061c869bb7076b6f6  dev_set/README.md
```

- All 60 `game.tw-pddl` files referenced by the manifest were re-hashed and
  matched their recorded `gamefile_sha256` exactly (0 mismatches).
- Aggregate SHA256 of all 60 game.tw-pddl contents concatenated in
  `dev_idx` order: `60f013f52ea85cd72e3b2e5494bc48ce1f035d48ac97af83ca3686a9f62af815`

## Code/config SHA256 (baseline for the sweep — identical to nopir_v2.5 freeze)

```
24a4d76f8bcffaa5dc0bfa27873958c6a3a78597c0ec01240c29a88d90726030  configs/controllers/ae_full_final_nopir_v2_5.yaml
35249c6949ab5e8fcd2d0292f34c4c68be78c78c57f15a92a088177dbd6f2f3f  ae/controllers/stateful_controller.py
cca07bb06ed6c6edda7693b20e1aed9062bc6cca70c6c26abfee2701c7760bcf  ae/controllers/config.py
dd24c6343dbf483e443d58a9744bfb5800e3453a2685da38ba577e12d08dacb6  ae/controllers/signals.py
5d059168480cb6829abf0bf97001a13bf98baad9ffa180873e8b6cba483b10ff  ae/controllers/canonical_action.py
811e2d5f8d417c6b756109a7f5d8e0eb4a5fc6600e459971dd192ef688915c05  alfworld_runs_ae/agents.py
```

## Substitution rule if a frozen game file is ever lost/corrupted

If a manifest entry's `game.tw-pddl` becomes unreadable in the future, its
replacement must come from that task type's `unused_backups_per_type` queue
in `dev60_candidate_pool_and_backups.json`, walked **in the saved fixed
order**, and must still result in 10 distinct `task_dir` families for that
type (skip any backup whose family already appears among the type's other
9 kept tasks). No family-repeat substitution, no reordering, no
performance-based selection. As of this freeze, 60/60 tasks pass
production preflight (load + reset) — no substitution is needed or has
been performed.

## Sweep configs (all derived from the frozen nopir_v2.5 base, one variable
## each, see `configs/controllers/ae_tune_dev_*.yaml`)

| config | warmup | grace | patch | max_int | cooldown |
|---|---|---|---|---|---|
| ae_tune_dev_baseline | 3 | 3 | 3 | 3 | 3 |
| ae_tune_dev_grace2 | 3 | **2** | 3 | 3 | 3 |
| ae_tune_dev_grace5 | 3 | **5** | 3 | 3 | 3 |
| ae_tune_dev_patch2 | 3 | 3 | **2** | 3 | 3 |
| ae_tune_dev_patch5 | 3 | 3 | **5** | 3 | 3 |
| ae_tune_dev_cooldown1 | 3 | 3 | 3 | 3 | **1** |
| ae_tune_dev_cooldown5 | 3 | 3 | 3 | 3 | **5** |

All 7 have `post_intervention_repeat_enabled: false` (frozen nopir_v2.5
base), `termination_policy=legacy_early_stop`, `MAX_STEPS=50`, same
`dev60_valid_seen_manifest.json` and task order, same A40/vLLM/Qwen3-8B
endpoint and generation params as the nopir_v2.5/C freeze.
