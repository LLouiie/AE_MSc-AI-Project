# Legacy — old Rule Library direction (frozen)

Moved here 2026-07-25 per `AE_MIGRATION_AUDIT.md` §4. This is the pre-pivot
ExpeL-style Rule Library / affect-scheduler scaffolding: cross-task rule
extraction (`consolidation.py`), when-to-consolidate scheduling
(`schedulers.py`), and their driver scripts (`run_practice.py`,
`run_exam.py`, `test_rulepool.py`).

Not deleted, not maintained. The new intra-episode direction lives in `ae/`
at the repo root — see `Handoff-README.md`. `llm.py`, `environment.py`,
`agents.py` stayed in the parent `hotpotqa_runs/` directory (still reused by
the new direction); these files' sys.path setup was updated to reach them
across the extra directory level.
