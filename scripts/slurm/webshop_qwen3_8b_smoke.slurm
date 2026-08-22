#!/bin/bash
#SBATCH --job-name=webshop_q3_smoke
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/webshop_q3_smoke-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/webshop_q3_smoke-%j.err

set -euo pipefail

REPO=/homes/jy625/projects/AE3
WS_ROOT=/vol/gpudata/jy625-ae-data/webshop_official
WS_PY=/vol/gpudata/jy625-ae-data/miniconda3/envs/webshop/bin/python
AE_PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
MODEL_PATH=/vol/gpudata/jy625-ae-data/models/Qwen3-8B
MODEL_NAME=Qwen/Qwen3-8B
LOGDIR=/vol/gpudata/jy625-ae-data/logs/webshop_q3_smoke_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
NODE_TMP=${SLURM_TMPDIR:-/tmp/ae3_${USER}_${SLURM_JOB_ID}}
LOCAL_MODEL=$NODE_TMP/Qwen3-8B

export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export JAVA_HOME=/vol/gpudata/jy625-ae-data/miniconda3/envs/webshop/lib/jvm
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"

WS_PORT=3000
LLM_PORT=$((31000 + SLURM_JOB_ID % 1000))
cleanup() {
    rc=$?
    [ -n "${WS_PID:-}" ] && kill "$WS_PID" 2>/dev/null || true
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true
    if [ -z "${SLURM_TMPDIR:-}" ]; then
        rm -rf "$NODE_TMP"
    fi
    if [ "$rc" -eq 0 ]; then
        title="OK - WebShop Qwen3-8B smoke"
        body="job $SLURM_JOB_ID passed: 3 episodes completed"
    else
        title="FAILED rc=$rc - WebShop Qwen3-8B smoke"
        body="job $SLURM_JOB_ID failed: $(tail -5 "$LOGDIR/run.log" 2>/dev/null | head -c 700)"
    fi
    curl -s -m 20 -H "Title: $title" -d "$body" \
        https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
    trap - EXIT
    exit "$rc"
}
trap cleanup EXIT

# Fail before allocating startup time if the agent environment is incomplete.
cd "$REPO/webshop_runs"
"$AE_PY" -c "import bs4, openai, requests; import webshop_trial"

# Keep slow CEPH transfer outside the vLLM readiness deadline, then load all
# checkpoint shards from node-local storage.
mkdir -p "$NODE_TMP"
avail_kb=$(df -Pk "$NODE_TMP" | awk 'NR==2 {print $4}')
if [ "${avail_kb:-0}" -ge 20971520 ]; then
    cp -a "$MODEL_PATH" "$LOCAL_MODEL"
else
    LOCAL_MODEL=$MODEL_PATH
fi

cd "$WS_ROOT" || exit 1
"$WS_PY" -m web_agent_site.app --log --attrs >"$LOGDIR/webshop.log" 2>&1 &
WS_PID=$!

source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
vllm serve "$LOCAL_MODEL" --served-model-name "$MODEL_NAME" --host 127.0.0.1 --port "$LLM_PORT" >"$LOGDIR/vllm.log" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$WS_PORT/fixed_0" >/dev/null 2>&1 && WS_READY=1
    curl -fsS "http://127.0.0.1:$LLM_PORT/v1/models" >/dev/null 2>&1 && LLM_READY=1
    [ "${WS_READY:-0}" = 1 ] && [ "${LLM_READY:-0}" = 1 ] && break
    sleep 10
done
[ "${WS_READY:-0}" = 1 ] && [ "${LLM_READY:-0}" = 1 ] || exit 2

cd "$REPO/webshop_runs" || exit 1
export WEBSHOP_URL="http://127.0.0.1:$WS_PORT"
export OPENAI_BASE_URL="http://127.0.0.1:$LLM_PORT/v1"
export OPENAI_API_KEY=EMPTY
export OPENAI_MODEL="$MODEL_NAME"

"$AE_PY" main.py --num_trials 1 --num_envs 3 --run_name "$LOGDIR/react" >"$LOGDIR/run.log" 2>&1
test -s "$LOGDIR/react/world.log"
grep -q 'TOTAL: 3' "$LOGDIR/react/world.log"
test "$(find "$LOGDIR/react" -maxdepth 1 -name 'env_results_trial_*.json' | wc -l)" -eq 1
# A completed loop is not enough: prove the model produced at least one
# command that the environment accepted beyond reset/think.
grep -Eq '^Action: (search|click)\[' "$LOGDIR/run.log"
echo "WEBSHOP QWEN3-8B SMOKE PASSED"
