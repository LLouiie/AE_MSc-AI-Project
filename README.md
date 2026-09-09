# Artificial Emotion for Language Agents

Artificial Emotion (AE) is a lightweight controller for ReAct agents. It monitors the current trajectory, maintains four internal states, and temporarily adds a `VERIFY`, `REFLECT`, or `REPLAN` directive when recovery is needed. It requires no training and no separate LLM call.

This repository contains the AE implementation and evaluation runners for ALFWorld and WebShop, together with ReAct, Reflexion, and ADaPT baselines.

## Quickstart

Clone the repository and create an environment:

```bash
git clone git@github.com:LLouiie/AE_MSc-AI-Project.git
cd AE_MSc-AI-Project
python -m venv .venv
source .venv/bin/activate
pip install -r alfworld_runs/requirements.txt
pip install alfworld
```

Download ALFWorld and set its data directory:

```bash
alfworld-download
export ALFWORLD_DATA=/path/to/alfworld_data
```

Start an OpenAI-compatible server for `Qwen/Qwen3-8B`, then run a five-task AE smoke test:

```bash
python -m ae.runners.run_alfworld \
  --baseline ae_full \
  --split practice \
  --limit 5 \
  --run-name quickstart \
  --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
  --demo-config configs/demos/one_shot_v1.yaml \
  --termination-policy legacy_early_stop \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000/v1
```

Outputs are written to `ae/runners/runs/quickstart/`.

## Full experiments

Production Slurm scripts are in `scripts/slurm/`. The main ALFWorld evaluation covers all 134 tasks:

```bash
sbatch scripts/slurm/ae3_runs/run_ae_n20.sbatch
```

Available AE modes are `full`, `no_trigger`, `random_trigger`, `verify_only`, `reflect_only`, and `replan_only`. See `scripts/slurm/ae3_runs/run_ae_ablation_table2.sbatch` for the ablation setup.

## WebShop

WebShop requires its web application and search index to be running. Setup assets are in `webshop_runs/setup/`. With WebShop and the model server available:

```bash
export WEBSHOP_URL=http://127.0.0.1:3000
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
cd webshop_runs
python run_ae.py \
  --num-envs 100 \
  --ae-config ../configs/controllers/ae_full_final_nopir_v2_5.yaml \
  --output webshop_ae.json
```

## Project layout

- `ae/` — controller, signals, affect-state updates, and runners
- `alfworld_runs_ae/` — ALFWorld agent and environment integration
- `webshop_runs/` — WebShop adapter and evaluation runners
- `configs/` — controller and one-shot demonstration settings
- `scripts/slurm/` — smoke tests and production jobs
- `external/ADaPT/` — ADaPT baseline integration

Generated logs, model files, and result artifacts are not tracked by Git.

## Acknowledgements

This project builds on [Reflexion](https://github.com/noahshinn/reflexion), [ReAct](https://github.com/ysymyth/ReAct), [ALFWorld](https://github.com/alfworld/alfworld), and [WebShop](https://github.com/princeton-nlp/WebShop).
