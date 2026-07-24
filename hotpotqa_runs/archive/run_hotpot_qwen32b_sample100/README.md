# Archived: exact code that produced `reflexion_wiki_qwen32b_sample100`

This directory is a frozen snapshot of the code that actually generated the
Reflexion HotpotQA result used in the thesis's baseline comparison. It is
**not** live code — nothing here is imported by anything else in this repo.
It exists purely so the result is traceable to the exact files that produced
it, since this snapshot was run from a separate working copy
(`~/projects/AE/baselines/reflexion/hotpotqa_runs/`) that had already
diverged from this repo's `main` branch (that copy predates this repo's
2026-07-12 LangChain-removal refactor, so `run_hotpot.py` still depends on
`langchain.Wikipedia`).

- **Result directory** (not committed, lives on the HPC filesystem):
  `~/projects/AE/baselines/reflexion/hotpotqa_runs/runs/reflexion_wiki_qwen32b_sample100/`
  (`results.jsonl`, `checkpoint.joblib`, `trial_log.txt`, run logs)
- **Exact run config** (copied here as `result_config.json`):
  ```json
  {
    "data": "data/hotpot-qa-distractor-sample.joblib",
    "run_name": "reflexion_wiki_qwen32b_sample100",
    "strategy": "reflexion",
    "trials": 5,
    "max_steps": 6,
    "limit": null,
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "base_url": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "api": "completions",
    "user_agent": "AE-thesis-project/1.0 (jy625@imperial.ac.uk)",
    "started": "2026-07-16 18:01:13"
  }
  ```
- **Command** (per `run_hotpot.py`'s own usage docstring, matching the config above):
  ```bash
  export WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (jy625@imperial.ac.uk)'
  python run_hotpot.py \
      --data data/hotpot-qa-distractor-sample.joblib \
      --run-name reflexion_wiki_qwen32b_sample100 \
      --strategy reflexion --trials 5 --max-steps 6 \
      --model Qwen/Qwen2.5-32B-Instruct \
      --base-url http://localhost:8000/v1 --api completions
  ```

Files here (`agents.py`, `environment.py`, `fewshots.py`, `llm.py`, `mocks.py`,
`prompts.py`, `react.py`, `run_hotpot.py`, `tests.py`, `util.py`) are copied
byte-for-byte from that working copy as of 2026-07-24, unmodified.
