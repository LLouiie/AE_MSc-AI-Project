#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%4
#SBATCH --job-name=llama31_suite
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/llama31_suite-%A_%a.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/llama31_suite-%A_%a.err

set -uo pipefail
METHODS=(react reflexion full no_trigger random_trigger verify_only reflect_only replan_only)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
REPO=/homes/jy625/projects/AE3
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
SOURCE_MODEL=/vol/gpudata/jy625-ae-data/models/Llama-3.1-8B-Instruct
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
ROOT=/vol/gpudata/jy625-ae-data/results/llama31_suite_${SLURM_ARRAY_JOB_ID}/${METHOD}
NODE_TMP=${SLURM_TMPDIR:-/data/jy625-ae/llama_suite_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
# Load weights from the shared model snapshot. Copying a full snapshot per
# array task can exhaust local disk when tasks land on the same node.
LOCAL_MODEL=$SOURCE_MODEL
PORT=$((20000 + SLURM_ARRAY_JOB_ID % 500 + SLURM_ARRAY_TASK_ID))
mkdir -p "$ROOT"
mkdir -p "$NODE_TMP"

notify() { curl -s -m 20 -H "Title: $1" -d "$2" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true; }
cleanup() {
  rc=$?; set +e
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
  if [ "$rc" -eq 0 ]; then notify "Llama suite $METHOD complete" "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; else notify "Llama suite $METHOD failed" "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, exit $rc"; fi
  exit "$rc"
}
trap cleanup EXIT

cd "$REPO" || exit 1
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0 AE_USE_CHAT_COMPLETIONS=1
export ALFWORLD_TOKENIZER_NAME_OR_PATH="$LOCAL_MODEL"
export XDG_CACHE_HOME=${NODE_TMP}/cache VLLM_CACHE_ROOT=${NODE_TMP}/vllm_cache TORCHINDUCTOR_CACHE_DIR=${NODE_TMP}/torch_cache
vllm serve "$LOCAL_MODEL" --served-model-name "$MODEL_NAME" --host 127.0.0.1 --port "$PORT" \
  --dtype half --max-model-len 40960 >"$ROOT/vllm.log" 2>&1 &
VLLM_PID=$!
for _ in $(seq 1 360); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && READY=1 && break
  kill -0 "$VLLM_PID" 2>/dev/null || exit 1
  sleep 5
done
[ "${READY:-0}" = 1 ] || exit 1

BASELINE=ae_full; EXTRA=()
case "$METHOD" in
  react) BASELINE=react ;;
  reflexion) BASELINE=reflexion; EXTRA=(--max-trials 10) ;;
  full) EXTRA=(--ae-ablation full) ;;
  *) EXTRA=(--ae-ablation "$METHOD") ;;
esac

run_group() {
  split=$1 types=$2 demo=$3 group=$4
  "$PY" -m ae.runners.run_alfworld --baseline "$BASELINE" "${EXTRA[@]}" \
    --split "$split" --task-types "$types" --run-name "${METHOD}_${group}_${split}_${SLURM_ARRAY_JOB_ID}" \
    --seed 42 --ae-random-seed 42 --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --demo-config "configs/demos/${demo}.yaml" --termination-policy legacy_early_stop \
    --model "$MODEL_NAME" --base-url "http://127.0.0.1:${PORT}/v1" --output-dir "$ROOT"
}
run_group practice put,examine one_shot_v1 A || exit 1
run_group exam put,examine one_shot_v1 A || exit 1
run_group practice heat,cool,clean one_shot_v1_index2 B || exit 1
run_group exam heat,cool,clean one_shot_v1_index2 B || exit 1
run_group practice puttwo one_shot_v1_index1 C || exit 1
run_group exam puttwo one_shot_v1_index1 C || exit 1

"$PY" - "$ROOT" <<'PY'
import json, pathlib, sys
r=pathlib.Path(sys.argv[1]); rows=[json.loads(x) for p in r.glob('*/episode_log.jsonl') for x in p.open() if x.strip()]
s={'tasks':len(rows),'successes':sum(x['success'] for x in rows),'success_rate':sum(x['success'] for x in rows)/len(rows),'llm_calls':sum(x['llm_calls'] for x in rows),'steps':sum((x.get('ae_summary') or {}).get('environment_steps',0) for x in rows)}
(r/'summary.json').write_text(json.dumps(s,indent=2)+'\n'); print(s)
assert len(rows)==134, len(rows)
PY
