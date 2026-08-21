# ae_full_nopir_v2_5_frozen — freeze record

Frozen 2026-08-05, prior to starting parameter-sensitivity tuning. This
document is the reference snapshot; nothing in this list should change
while the tuning round (dev-set sweep) is in progress. If any of these
values ever need to change, that is itself a signal the tuning round's
isolation from the 134-task frozen baseline has been broken.

## Git state

- Branch: `feature/intra-episode-ae`
- HEAD: `1d4bb66b22eeb26bad0cc9741f0997cd8a663536`
- Working tree: not committed (matches project convention throughout this
  session — see `feedback_ae_standing_constraints`). Full `git status
  --short` and `git diff` captured below/alongside.
- Full patch of tracked-file changes saved at
  `/vol/gpudata/jy625-ae-data/logs/freeze_nopir_v2_5_tracked.patch`
  (2237 lines, `git diff` output against HEAD).
- Untracked new files as of this freeze (relevant to AE, not the
  ReflexGrad/anchor side-work also present in the tree):
  `ae/controllers/canonical_action.py`,
  `ae/controllers/tests/test_canonical_action.py`,
  `ae/controllers/tests/test_signal_semantics.py`,
  `ae/controllers/tests/test_intervention_outcome.py`,
  `ae/controllers/tests/test_post_intervention_exact_repeat.py`,
  `ae/controllers/tests/test_pir_ablation_modes.py`,
  `ae/controllers/tests/test_ae_full_production_termination_gate.py`,
  `configs/controllers/ae_full_final_nopir_v2_5.yaml`,
  `configs/controllers/ae_full_reflect2repeat_v2_6_c.yaml`.

## SHA256 of the code/config that defines this version's behavior

```
24a4d76f8bcffaa5dc0bfa27873958c6a3a78597c0ec01240c29a88d90726030  configs/controllers/ae_full_final_nopir_v2_5.yaml
bbe963f9d9222ada666cf2b209c85bd1295b0e08b23e38f4de7d0be5fc351ef8  configs/controllers/ae_full.yaml
35249c6949ab5e8fcd2d0292f34c4c68be78c78c57f15a92a088177dbd6f2f3f  ae/controllers/stateful_controller.py
cca07bb06ed6c6edda7693b20e1aed9062bc6cca70c6c26abfee2701c7760bcf  ae/controllers/config.py
dd24c6343dbf483e443d58a9744bfb5800e3453a2685da38ba577e12d08dacb6  ae/controllers/signals.py
5d059168480cb6829abf0bf97001a13bf98baad9ffa180873e8b6cba483b10ff  ae/controllers/canonical_action.py
811e2d5f8d417c6b756109a7f5d8e0eb4a5fc6600e459971dd192ef688915c05  alfworld_runs_ae/agents.py
```

Note: `configs/controllers/ae_full.yaml` (the repo's nominal "default"
config, sha256 above) is NOT what nopir_v2.5 uses — nopir_v2.5 uses
`ae_full_final_nopir_v2_5.yaml` specifically
(`post_intervention_repeat_enabled: false`). `ae_full.yaml` itself was
never modified this round and still defaults to
`post_intervention_repeat_enabled: true` (the deprecated original PIR) —
it is listed here only as a reference point, not as part of the frozen
nopir_v2.5 definition.

## Behavioral contents of this frozen version

- put→move canonicalization: present (`ae/controllers/canonical_action.py`)
- warmup boundary + edge-detection fix: present (`agents.py`'s
  `controller_in_warmup` grace-suppression path; `stateful_controller.py`'s
  frozen-baseline-during-warmup edge detection)
- frustration_medium gate fix: present (`consecutive_exact_action_repeat`
  signal replacing `repeated_action`/`repeated_observation` in
  `_select_intervention`)
- PIR / reflect2repeat: **disabled**
  (`post_intervention_repeat_enabled: false` in the yaml;
  `post_intervention_repeat_reflect_double` is moot when disabled)
- No loop-detection or other new algorithmic rule beyond the above three
  fixes.

## Current parameter values (must stay fixed through the tuning round's
## baseline arm; only the sweep's 6 variant arms deviate one at a time)

```
warmup_steps          = 3
recovery_grace_steps  = 3
patch_duration_steps  = 3
max_interventions     = 3
cooldown_steps        = 3
frustration_medium    = 0.45
termination_policy    = legacy_early_stop
MAX_STEPS             = 50
```

## Inference environment

- GPU: 1x NVIDIA A40 (node `gpuvm35.doc.ic.ac.uk`, SLURM job 270382,
  partition `a40`)
- Model: `Qwen/Qwen3-8B` (`/vol/gpudata/jy625-ae-data/models/Qwen3-8B`),
  served via vLLM, `max_model_len=40960`
- vLLM: 0.25.1
- PyTorch: 2.11.0+cu130 (CUDA 13.0)
- Client generation params: `temperature=0`, `max_tokens=ACT_MAX_TOKENS`
  (256, from `alfworld_runs_ae/agents.py`), stop sequences per
  `agents.ACT_STOP`

## Reference result (100/134, the number this freeze certifies)

- `ae/runners/runs/ae_full_final_nopir_v2_5_a40_practice/` — 100 episodes,
  78 successes
- `ae/runners/runs/ae_full_final_nopir_v2_5_a40_exam/` — 34 episodes,
  22 successes
- Overall: 100/134 (74.6%)
- These two directories are the certified reference and must not be
  overwritten, appended to, or reused as a dev/tuning set.
