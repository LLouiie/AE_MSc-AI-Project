#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=llama31_ae_full
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/llama31_ae_full-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/llama31_ae_full-%j.err

set -uo pipefail
REPO=/homes/jy625/projects/AE3
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
MODEL_PATH=/vol/gpudata/jy625-ae-data/models/Llama-3.1-8B-Instruct
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
OUT=/vol/gpudata/jy625-ae-data/results/llama31_ae_full_${SLURM_JOB_ID}
PORT=$((19000 + SLURM_JOB_ID % 1000))
mkdir -p "$OUT"

cleanup() {
  rc=$?
  set +e
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
  if [ "$rc" -eq 0 ]; then title="Llama 3.1 8B AE full complete"; else title="Llama 3.1 8B AE full failed"; fi
  curl -s -m 20 -H "Title: $title" -d "Job ${SLURM_JOB_ID}, exit ${rc}" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
  exit "$rc"
}
trap cleanup EXIT

cd "$REPO" || exit 1
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
export AE_USE_CHAT_COMPLETIONS=1
vllm serve "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 --port "$PORT" --dtype half --max-model-len 40960 \
  >"$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && READY=1 && break
  kill -0 "$VLLM_PID" 2>/dev/null || exit 1
  sleep 5
done
[ "${READY:-0}" = 1 ] || exit 1

run_group() {
  split=$1 types=$2 demo=$3 group=$4
  "$PY" -m ae.runners.run_alfworld --baseline ae_full --split "$split" \
    --task-types "$types" --run-name "llama31_ae_full_${group}_${split}_${SLURM_JOB_ID}" \
    --seed 42 --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --demo-config "configs/demos/${demo}.yaml" --termination-policy legacy_early_stop \
    --model "$MODEL_NAME" --base-url "http://127.0.0.1:${PORT}/v1" --output-dir "$OUT"
}

run_group practice put,examine one_shot_v1 A || exit 1
run_group exam put,examine one_shot_v1 A || exit 1
run_group practice heat,cool,clean one_shot_v1_index2 B || exit 1
run_group exam heat,cool,clean one_shot_v1_index2 B || exit 1
run_group practice puttwo one_shot_v1_index1 C || exit 1
run_group exam puttwo one_shot_v1_index1 C || exit 1

"$PY" - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for p in root.glob('*/episode_log.jsonl') for line in p.open() if line.strip()]
summary = {
    'tasks': len(rows),
    'successes': sum(int(r['success']) for r in rows),
    'success_rate': sum(int(r['success']) for r in rows) / len(rows),
    'llm_calls': sum(int(r['llm_calls']) for r in rows),
    'environment_steps': sum(len(r.get('step_log', [])) for r in rows),
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(summary)
if len(rows) != 134:
    raise SystemExit(f'expected 134 tasks, got {len(rows)}')
PY
