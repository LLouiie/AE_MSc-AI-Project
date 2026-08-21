# WebShop setup — the parts that are ours

The WebShop clone itself lives outside this repo at
`/vol/gpudata/jy625-ae-data/webshop_official` (upstream:
https://github.com/princeton-nlp/WebShop). These four files are the local
additions needed to make it run here, kept in the repo so a cluster move
doesn't lose them.

| File | What it is |
|---|---|
| `webshop_env.sh` | **Source this before touching WebShop.** Points at the private conda 3.8 env and puts the JVM on `PATH`/`JVM_PATH` (pyserini needs it). Nothing works without it. |
| `requirements_trimmed.txt` | upstream `requirements.txt` minus the packages that don't resolve on this cluster |
| `smoke_text_env.py` | non-interactive gym-env smoke test: `reset` → `search[shirt]` → `click[...]`, asserting each step. Passes on the 1000-product subset. |
| `EXPECTED_SHA256.txt` | checksums for the two large data files, so a truncated download is caught rather than silently producing a broken index |

## Traps

- **`gdown` is dead** for the WebShop data ("too many accesses"). Use the HuggingFace
  mirror `YWZBrandon/webshop-data` instead. `items_shuffle.json` is 5.48 GB — download with
  `curl -C -` in a retry loop; it truncated repeatedly otherwise.
- Full catalogue is **1.18M products / 12,087 instructions**, but `web_agent_site/utils.py`
  still points at the **1000-product subset**. The full Lucene index has not been built yet.
- Episodes are capped at **15 steps** (`webshop_runs/webshop_trial.py`) and reward is
  **continuous 0–1**; Reflexion counts only `reward == 1.0` as success.
- Published methods subset **instructions**, not products: 100 for ADaPT and Reflexion,
  500 for ReAct.
