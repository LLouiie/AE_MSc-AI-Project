# Artificial Emotion for Language Agents

This repository evaluates an Artificial Emotion (AE) controller for language agents in interactive environments. AE observes an agent trajectory, maintains four lightweight states (uncertainty, frustration, surprise, and confidence), and transiently injects a `VERIFY`, `REFLECT`, or `REPLAN` directive into the next ReAct prompt. The controller does not require training or an additional LLM call.

The current experiments use ALFWorld and WebShop with ReAct, Reflexion, and ADaPT baselines. This project is built on the original [Reflexion](https://github.com/noahshinn/reflexion) repository.

## Repository structure

- `ae/`: controller, signals, affect-state updates, baselines, and ALFWorld runner.
- `alfworld_runs_ae/`: ALFWorld agent loop and environment integration.
- `webshop_runs/`: WebShop environment adapter and evaluation runners.
- `configs/controllers/`: AE controller and ablation configurations.
- `configs/demos/`: one-shot demonstration selections.
- `scripts/slurm/`: reproducible cluster job scripts.

## ALFWorld

Start an OpenAI-compatible model server, then run:

```bash
python -m ae.runners.run_alfworld \
  --baseline ae_full \
  --split practice \
  --run-name ae_example \
  --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
  --demo-config configs/demos/one_shot_v1.yaml \
  --termination-policy legacy_early_stop \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000/v1
```

The full 134-task evaluation is split by task type and dataset partition. See `scripts/slurm/ae3_runs/run_ae_n20.sbatch` for the production launch procedure.

## WebShop

Start the WebShop environment and an OpenAI-compatible model server, then run:

```bash
cd webshop_runs
python run_ae.py \
  --num-envs 100 \
  --ae-config ../configs/controllers/ae_full_final_nopir_v2_5.yaml \
  --output webshop_ae.json
```

The environment setup assets are under `webshop_runs/setup/`.

## Ablations

The ALFWorld and WebShop runners share the same AE implementation and expose the following controller modes:

- full AE
- no trigger
- random trigger
- VERIFY only
- REFLECT only
- REPLAN only

Generated runs, logs, and result files are intentionally excluded from Git. Production outputs are stored separately from the source checkout.
