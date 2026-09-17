#!/bin/bash
#SBATCH --partition=t4
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --job-name=gpt4turbo_match
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/gpt4turbo_match-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/gpt4turbo_match-%j.err

set -u
. /homes/jy625/.config/ae/openai.env
ROOT=/homes/jy625/projects/AE3
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
OUT=/vol/gpudata/jy625-ae-data/results/model_smokes/gpt4turbo_matched_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$ROOT" || exit 1

notify() {
    curl -s -m 20 -H "Title: $1" -d "$2" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
}

run_group() {
    types=$1
    demo=$2
    group=$3
    "$PY" -m ae.runners.run_alfworld \
        --baseline ae_full --split practice --task-types "$types" --limit 2 \
        --run-name "gpt4turbo_match_${group}_${SLURM_JOB_ID}" --seed 42 \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
        --demo-config "configs/demos/${demo}.yaml" \
        --termination-policy legacy_early_stop \
        --model gpt-4-turbo-2024-04-09 --base-url https://api.openai.com/v1 \
        --output-dir "$OUT"
}

run_group put,examine one_shot_v1 A || { notify "GPT-4 Turbo matched smoke failed" "Job ${SLURM_JOB_ID}: group A failed"; exit 1; }
run_group heat,cool,clean one_shot_v1_index2 B || { notify "GPT-4 Turbo matched smoke failed" "Job ${SLURM_JOB_ID}: group B failed"; exit 1; }
run_group puttwo one_shot_v1_index1 C || { notify "GPT-4 Turbo matched smoke failed" "Job ${SLURM_JOB_ID}: group C failed"; exit 1; }
notify "GPT-4 Turbo matched smoke complete" "Job ${SLURM_JOB_ID}: 6 matched-demo tasks finished"
