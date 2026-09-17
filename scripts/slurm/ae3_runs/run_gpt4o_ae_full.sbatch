#!/bin/bash
#SBATCH --partition=t4
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --job-name=gpt4o_ae_full
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/gpt4o_ae_full-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/gpt4o_ae_full-%j.err

set -u
. /homes/jy625/.config/ae/openai.env
export OPENAI_MODEL=gpt-4o-2024-11-20
export OPENAI_BASE_URL=https://api.openai.com/v1

ROOT=/homes/jy625/projects/AE3
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
OUT=/vol/gpudata/jy625-ae-data/results/gpt4o_ae_full_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$ROOT" || exit 1

notify() {
    curl -s -m 20 -H "Title: $1" -d "$2" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
}

run_group() {
    split=$1
    types=$2
    demo=$3
    group=$4
    "$PY" -m ae.runners.run_alfworld \
        --baseline ae_full --split "$split" --task-types "$types" \
        --run-name "gpt4o_ae_${group}_${split}_${SLURM_JOB_ID}" \
        --seed 42 \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
        --demo-config "configs/demos/${demo}.yaml" \
        --termination-policy legacy_early_stop \
        --model gpt-4o-2024-11-20 --base-url https://api.openai.com/v1 \
        --output-dir "$OUT"
}

run_group practice put,examine one_shot_v1 A || { notify "GPT-4o AE failed" "Job ${SLURM_JOB_ID}: group A practice failed"; exit 1; }
run_group exam put,examine one_shot_v1 A || { notify "GPT-4o AE failed" "Job ${SLURM_JOB_ID}: group A exam failed"; exit 1; }
run_group practice heat,cool,clean one_shot_v1_index2 B || { notify "GPT-4o AE failed" "Job ${SLURM_JOB_ID}: group B practice failed"; exit 1; }
run_group exam heat,cool,clean one_shot_v1_index2 B || { notify "GPT-4o AE failed" "Job ${SLURM_JOB_ID}: group B exam failed"; exit 1; }
run_group practice puttwo one_shot_v1_index1 C || { notify "GPT-4o AE failed" "Job ${SLURM_JOB_ID}: group C practice failed"; exit 1; }
run_group exam puttwo one_shot_v1_index1 C || { notify "GPT-4o AE failed" "Job ${SLURM_JOB_ID}: group C exam failed"; exit 1; }

notify "GPT-4o AE complete" "Job ${SLURM_JOB_ID}: all 134 ALFWorld tasks finished"
