# Vendored source: MPO (ReflAct v2 Appendix G's ALFWorld ReAct anchor)

Repo: https://github.com/WeiminXiong/MPO
Commit SHA (pinned, fetched 2026-07-29, `main` at fetch time): `5529eca70ab352eb7e56533ca44cb60fbf34bef1`
Commit date: 2025-08-20T17:40:27Z
License: Apache-2.0 (see `LICENSE` in this directory, copied verbatim from the same commit)

Fetched via `raw.githubusercontent.com/WeiminXiong/MPO/<sha>/<path>` — a one-time
vendoring at implementation time, not a runtime dependency. `ae/baselines/
react_reflact_anchor.py` and `alfworld_runs_ae/mpo_prompts.py` import only from
the files below; nothing in this repo fetches MPO at run time.

## Reproducing this directory from a fresh clone

`external/` is entirely gitignored in this repo (`.gitignore` line 155),
same as `external/reflexgrad` and `external/ADaPT` — so these files do not
travel with `git clone`. Run:

```
python3 external/mpo_reflact/fetch_and_verify.py
```

This re-downloads each file from the pinned commit above and refuses to
write anything that doesn't match the SHA-256 hashes react_reflact_anchor
was actually implemented and tested against (hashes are hardcoded in the
script itself, not read from anything fetched). `--verify-only` checks
what's on disk without any network access; `--force` re-fetches even if
already correct. Non-zero exit and an explicit MISMATCH/FAIL line on any
integrity failure — never a silent partial vendor.

## Files, exact upstream path -> local path

| Upstream path | Local path | Used at runtime? |
|---|---|---|
| `prompt/instructions/alfworld_inst.txt` | `prompt/instructions/alfworld_inst.txt` | Yes — loaded verbatim as the instruction text |
| `prompt/icl_examples/alfworld_icl.json` | `prompt/icl_examples/alfworld_icl.json` | Yes — loaded verbatim; category keys (`pick_and_place`, `pick_clean_then_place`, `pick_heat_then_place`, `pick_cool_then_place`, `pick_two_obj`, `look_at_obj`) match `alfworld_runs_ae/environment.py::PREFIXES`'s keys exactly |
| `prompt/templates.py` | `prompt/templates.py` | Yes — `prompt_with_icl()` imported and called with `icl_num=1, workflow=None` (no modification) |
| `envs/alfworld_env.py` | `envs/alfworld_env_reference.py` | No — kept as a citable reference only. Depends on MPO's own `envs.BaseEnv`/`tasks.AlfWorldTask`/`utils.datatypes.State`, which are not vendored (out of scope: we reuse this repo's own environment/success-determination code instead, per explicit instruction — see `alfworld_runs_ae/mpo_prompts.py` module docstring for the specific behaviors mirrored from this file: `parse_action`'s regex and the `put X in/on Y` normalization). |

## Known upstream behaviors intentionally NOT replicated (see audit gaps)

- `AlfWorldEnv.step()` determines `self.state.success` from `done` alone, not
  explicitly from `info["won"]`. This repo's AE-project convention (spec
  section 1.1) requires success to come from `info["won"]` specifically; we
  keep that convention rather than copying MPO's looser one. Disclosed, not
  hidden.
- `max_bad_steps=50` (a separate cap on parse-failure steps, distinct from
  `max_steps`) is not replicated. `react_reflact_anchor` uses a single
  `agent_steps` budget (30, per the reproduction spec) that already counts
  parse failures; a second independent bad-steps cap was not requested by the
  spec and was not added.
- MPO's own step counter (`self.state.steps`) increments on parse-failure
  steps exactly like a real step, with no separate "did env.step() actually
  run" distinction. This repo's `agent_steps` / `env_actions` / `llm_calls`
  split (reproduction spec section 五) is a deliberate refinement beyond
  what MPO itself logs, not a bug-for-bug replication.
