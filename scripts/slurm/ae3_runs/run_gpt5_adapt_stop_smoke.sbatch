#!/bin/bash
#SBATCH --job-name=gpt5_adapt_stop
#SBATCH --partition=t4
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/gpt5_adapt_stop_smoke-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/gpt5_adapt_stop_smoke-%j.err

set -uo pipefail

ADAPT=/homes/jy625/projects/AE3/external/ADaPT
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/gpt5_adapt_stop_smoke_${SLURM_JOB_ID}
RESULTS_ROOT=/vol/gpudata/jy625-ae-data/results/adapt_hosted_stop_smoke
RUN_MODE=${1:-smoke}
MODEL=${2:-gpt-5}
MODEL_LABEL=$MODEL
if [ "$RUN_MODE" = full ]; then
    FNAME=${MODEL_LABEL}_adapt_counted_full_${SLURM_JOB_ID}
    TASK_ARGS=(--eval-all True)
    EXPECTED_TASKS=134
else
    FNAME=${MODEL_LABEL}_adapt_counted_smoke_${SLURM_JOB_ID}
    TASK_ARGS=(--num-task-samples 2)
    EXPECTED_TASKS=12
fi
mkdir -p "$LOGDIR" "$RESULTS_ROOT"

notify() {
    curl -s -m 20 -H "Title: $1" -d "$2" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
}
on_exit() {
    rc=$?
    if [ "$rc" -eq 0 ]; then
        notify "OK - ADaPT $MODEL $RUN_MODE" "job $SLURM_JOB_ID completed; see $LOGDIR"
    else
        notify "FAILED - ADaPT $MODEL $RUN_MODE" "job $SLURM_JOB_ID rc=$rc; see $LOGDIR"
    fi
}
trap on_exit EXIT

source /homes/jy625/.config/ae/openai.env
export OPENAI_BASE_URL=https://api.openai.com/v1
export ALFWORLD_DATA=/vol/gpudata/jy625-ae-data/alfworld_data

cd "$ADAPT" || exit 1
"$PY" run_alfworld.py \
    --LM "$MODEL" --exec-LM "$MODEL" \
    --eval-type ood "${TASK_ARGS[@]}" \
    --executor hybrid --react-type oneshot \
    --max-depth 3 --max-runs 20 \
    --fname "$FNAME" --results-root "$RESULTS_ROOT" \
    --verbose True > "$LOGDIR/run.log" 2>&1

"$PY" - "$RESULTS_ROOT" "$FNAME" "$EXPECTED_TASKS" <<'PY' > "$LOGDIR/check.txt"
import glob
import json
import os
import sys

root, fname, expected = sys.argv[1:]
paths = glob.glob(os.path.join(root, "**", fname + "_*.json"), recursive=True)
assert paths, "result JSON not found"
data = json.load(open(paths[0]))
tasks = [task for group, rows in data.items() if group != "overall" for task in rows.values()]
assert len(tasks) == int(expected), f"expected {expected} tasks, got {len(tasks)}"
overall = data["overall"]
success = sum(overall["success"])
count = sum(overall["count"])
plans = sum(bool(task.get("plans")) for task in tasks)
print(f"success={success}/{count}; tasks_with_plans={plans}/{count}")
print(paths[0])
PY

