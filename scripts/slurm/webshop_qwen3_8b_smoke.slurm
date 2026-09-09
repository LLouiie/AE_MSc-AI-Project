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
NUM_ENVS=${NUM_ENVS:-${2:-3}}
NUM_TRIALS=${NUM_TRIALS:-${4:-1}}
ADAPT_WEBSHOP=${ADAPT_WEBSHOP:-${1:-0}}
STRICT_SMOKE=${STRICT_SMOKE:-${3:-1}}
USE_MEMORY=${USE_MEMORY:-${5:-0}}
WEBSHOP_MAX_STEPS=${WEBSHOP_MAX_STEPS:-${6:-15}}
export WEBSHOP_MAX_STEPS
METHOD=ReAct
[ "$USE_MEMORY" = "1" ] && METHOD=Reflexion
LOGDIR=/vol/gpudata/jy625-ae-data/logs/webshop_q3_smoke_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
NODE_TMP=${SLURM_TMPDIR:-/tmp/ae3_${USER:-jy625}_${SLURM_JOB_ID}}
LOCAL_MODEL=$NODE_TMP/Qwen3-8B

mkdir -p "$NODE_TMP/tmp" "$NODE_TMP/cache/vllm" "$NODE_TMP/cache/torchinductor" "$NODE_TMP/cache/triton"
export TMPDIR="$NODE_TMP/tmp"
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export XDG_CACHE_HOME="$NODE_TMP/cache"
export VLLM_CACHE_ROOT="$NODE_TMP/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$NODE_TMP/cache/torchinductor"
export TRITON_CACHE_DIR="$NODE_TMP/cache/triton"
export VLLM_USE_FLASHINFER_SAMPLER=0
export JAVA_HOME=/vol/gpudata/jy625-ae-data/miniconda3/envs/webshop/lib/jvm
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"

WS_PORT=3000
LLM_PORT=$((31000 + SLURM_JOB_ID % 1000))
cleanup() {
    rc=$?
    # Notification is best-effort cleanup. Do not let `set -e` or pipefail
    # abort the trap before curl (ADaPT has no react/world.log).
    set +e
    [ -n "${WS_PID:-}" ] && kill "$WS_PID" 2>/dev/null || true
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true
    if [ -z "${SLURM_TMPDIR:-}" ]; then
        rm -rf "$NODE_TMP"
    fi
    if [ "$rc" -eq 0 ]; then
        title="OK - WebShop Qwen3-8B $METHOD one-shot steps=$WEBSHOP_MAX_STEPS"
        if [ "$METHOD" = "ADaPT" ]; then
            body="job $SLURM_JOB_ID passed: $("$AE_PY" -c 'import json,sys; x=json.load(open(sys.argv[1]))["overall"]; print("reward=%.3f success=%.3f n=%d" % (x["score"],x["success_rate"],x["count"]))' "$LOGDIR/result.json" 2>/dev/null)"
        else
            body="job $SLURM_JOB_ID passed: $(tail -8 "$LOGDIR/react/world.log" 2>/dev/null | head -c 700)"
        fi
    else
        title="FAILED rc=$rc - WebShop Qwen3-8B $METHOD one-shot steps=$WEBSHOP_MAX_STEPS"
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
PYTHONPATH="$WS_ROOT" "$WS_PY" "$REPO/webshop_runs/run_frozen_webshop.py" \
  --log --attrs --goal-seed 233 >"$LOGDIR/webshop.log" 2>&1 &
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

export WEBSHOP_URL="http://127.0.0.1:$WS_PORT"
export OPENAI_BASE_URL="http://127.0.0.1:$LLM_PORT/v1"
export OPENAI_API_KEY=EMPTY
export OPENAI_MODEL="$MODEL_NAME"

cd "$REPO/webshop_runs" || exit 1
"$AE_PY" capture_goal_manifest.py --num-envs "$NUM_ENVS" \
    --output "$LOGDIR/goal_manifest.json"

if [ "$ADAPT_WEBSHOP" = "1" ]; then
    METHOD=ADaPT
    cd "$REPO" || exit 1
    export NUM_SESS="$NUM_ENVS"
    "$AE_PY" external/ADaPT/run_webshop.py >"$LOGDIR/run.log" 2>&1
    RESULT="$REPO/results/webshop/ADaPT_maxd3_pgdisp3_sess${NUM_ENVS}.json"
    test -s "$RESULT"
    cp "$RESULT" "$LOGDIR/result.json"
    "$AE_PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["overall"]["count"] == int(sys.argv[2])' "$LOGDIR/result.json" "$NUM_ENVS"
    # Failed module attempts are intentionally absent from action_history, so
    if [ "$STRICT_SMOKE" = "1" ]; then
    # validate the complete plan trace instead of replay checkpoints.
    "$AE_PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); eps=[(k,v) for k,v in x.items() if k != "overall"]; bad=[(k, "exception" if v.get("error") else "no post-search module") for k,v in eps if v.get("error") or not any(("SimpleMatch[" in p or "DetailMatch[" in p or "Buy[" in p) and " at depth " in p for p in v.get("plans", [])[1:])]; assert eps and not bad, "ADaPT incomplete episodes: %s" % bad' "$LOGDIR/result.json"
    fi
else
    cd "$REPO/webshop_runs" || exit 1
    if [ "$USE_MEMORY" = "1" ]; then
        "$AE_PY" main.py --num_trials "$NUM_TRIALS" --num_envs "$NUM_ENVS" --run_name "$LOGDIR/react" --use_memory >"$LOGDIR/run.log" 2>&1
    else
        "$AE_PY" main.py --num_trials "$NUM_TRIALS" --num_envs "$NUM_ENVS" --run_name "$LOGDIR/react" >"$LOGDIR/run.log" 2>&1
    fi
    test -s "$LOGDIR/react/world.log"
    grep -q "TOTAL: $NUM_ENVS" "$LOGDIR/react/world.log"
    test "$(find "$LOGDIR/react" -maxdepth 1 -name 'env_results_trial_*.json' | wc -l)" -eq "$NUM_TRIALS"
    grep -Eq '^Action: (search|click)\[' "$LOGDIR/run.log"
fi
LLM_CALLS=$(grep -c 'POST /v1/chat/completions HTTP/1.1' "$LOGDIR/vllm.log" || true)
"$AE_PY" -c 'import json,sys; total=int(sys.argv[1]); n=int(sys.argv[2]); json.dump({"total_llm_calls":total,"average_llm_calls_per_episode":total/n,"count":n},open(sys.argv[3],"w"),indent=2)' \
    "$LLM_CALLS" "$NUM_ENVS" "$LOGDIR/llm_calls.json"
echo "WEBSHOP QWEN3-8B ${METHOD^^} ONE-SHOT PASSED: $NUM_ENVS episodes"
