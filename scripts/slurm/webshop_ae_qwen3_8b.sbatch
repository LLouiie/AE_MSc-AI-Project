#!/bin/bash
#SBATCH --job-name=webshop_ae
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/webshop_ae-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/webshop_ae-%j.err

set -euo pipefail
REPO=/homes/jy625/projects/AE3
WS_ROOT=/vol/gpudata/jy625-ae-data/webshop_official
WS_PY=/vol/gpudata/jy625-ae-data/miniconda3/envs/webshop/bin/python
AE_PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
MODEL=/vol/gpudata/jy625-ae-data/models/Qwen3-8B
NUM_ENVS=${1:-${NUM_ENVS:-3}}
SHOPPING_GUIDANCE=${2:-${SHOPPING_GUIDANCE:-on}}
AE_ABLATION=${3:-${AE_ABLATION:-full}}
AE_RANDOM_SEED=${4:-${AE_RANDOM_SEED:-42}}
AE_RANDOM_TRIGGER_PROBABILITY=${5:-${AE_RANDOM_TRIGGER_PROBABILITY:-0.5}}
LOGDIR=/vol/gpudata/jy625-ae-data/logs/webshop_ae_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
PORT=$((32000 + SLURM_JOB_ID % 700))

cleanup() {
  rc=$?
  set +e
  [ -n "${WS_PID:-}" ] && kill "$WS_PID" 2>/dev/null
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
  if [ "$rc" -eq 0 ]; then
    body=$("$AE_PY" -c 'import json,sys; x=json.load(open(sys.argv[1]))["overall"]; print("reward=%.3f success=%.3f n=%d triggered=%d interventions=%d"%(x["average_reward"],x["success_rate"],x["count"],x["triggered_episodes"],x["interventions"]))' "$LOGDIR/result.json")
    title="OK - WebShop AE $AE_ABLATION Qwen3-8B one-shot"
  else
    body="job $SLURM_JOB_ID failed; see $LOGDIR"
    title="FAILED rc=$rc - WebShop AE $AE_ABLATION Qwen3-8B one-shot"
  fi
  curl -s -m 20 -H "Title: $title" -d "$body" https://ntfy.sh/AE_new_alert >/dev/null 2>&1
  exit "$rc"
}
trap cleanup EXIT

export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export JAVA_HOME=/vol/gpudata/jy625-ae-data/miniconda3/envs/webshop/lib/jvm
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"

cd "$WS_ROOT"
PYTHONPATH="$WS_ROOT" "$WS_PY" "$REPO/webshop_runs/run_frozen_webshop.py" \
  --log --attrs --goal-seed 233 >"$LOGDIR/webshop.log" 2>&1 &
WS_PID=$!
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
vllm serve "$MODEL" --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port "$PORT" >"$LOGDIR/vllm.log" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 180); do
  curl -fsS http://127.0.0.1:3000/fixed_0 >/dev/null 2>&1 && A=1
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && B=1
  [ "${A:-0}" = 1 ] && [ "${B:-0}" = 1 ] && break
  sleep 10
done
[ "${A:-0}" = 1 ] && [ "${B:-0}" = 1 ]

export WEBSHOP_URL=http://127.0.0.1:3000
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=EMPTY
export OPENAI_MODEL=Qwen/Qwen3-8B
cd "$REPO/webshop_runs"
"$AE_PY" run_ae.py --num-envs "$NUM_ENVS" \
  --ae-config "$REPO/configs/controllers/ae_full_final_nopir_v2_5.yaml" \
  --shopping-guidance "$SHOPPING_GUIDANCE" \
  --ae-ablation "$AE_ABLATION" \
  --ae-random-seed "$AE_RANDOM_SEED" \
  --ae-random-trigger-probability "$AE_RANDOM_TRIGGER_PROBABILITY" \
  --output "$LOGDIR/result.json" >"$LOGDIR/run.log" 2>&1
"$AE_PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["overall"]["count"]==int(sys.argv[2]); assert all(len(e["actions"])>0 for e in x["episodes"])' "$LOGDIR/result.json" "$NUM_ENVS"
LLM_CALLS=$(grep -c 'POST /v1/chat/completions HTTP/1.1' "$LOGDIR/vllm.log" || true)
"$AE_PY" -c 'import json,sys; n=int(sys.argv[2]); calls=int(sys.argv[3]); json.dump({"total_llm_calls":calls,"average_llm_calls_per_episode":calls/n},open(sys.argv[1],"w"),indent=2)' "$LOGDIR/llm_calls.json" "$NUM_ENVS" "$LLM_CALLS"
