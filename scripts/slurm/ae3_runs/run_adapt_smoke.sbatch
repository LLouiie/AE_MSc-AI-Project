#!/bin/bash
#SBATCH --job-name=adapt_smoke
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/adapt_smoke-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/adapt_smoke-%j.err

# ============================================================================
# SMOKE TEST: ADaPT on ALFWorld, ReAct executor with a ONE-SHOT demo matching
# this project's ReAct/AE/Reflexion runs. 12 tasks (2 per task type).
#
# Why a smoke test rather than going straight to 10 repeats: ADaPT has already
# been run here once at full scale and scored 8/134 (6.0%), with all 8 wins in
# `examine` and 0/24 0/31 0/23 0/21 0/17 everywhere else -- see
# results/comparison/Qwen/Qwen3-8B/full_adapt_alfworld_v2_*.json. The cause is
# documented in run_alfworld.py's AE3 note: plan_llm() never once emitted the
# "Step N:"/"Execution Order:" format, so plan_to_args() always parsed an empty
# plan and every multi-stage task failed at depth 1 with no decomposition at
# all. The 8 wins were single-shot atomic-executor hits, not ADaPT recursion.
#
# THAT run used the DEFAULT `atomic` executor. This one uses `--executor react`
# with the new `--react-type oneshot`, which is a materially different
# configuration, so the 6% figure does not predict it. 12 tasks settles which
# regime we are in before committing 10 repeats x 134 tasks to it.
#
# --react-type oneshot is an AE3 addition; upstream's std/cross/common are
# hardcoded two-shot and are left byte-identical. Its per-type demo indices
# (put/examine 0, heat/cool/clean 2, puttwo 1) are the same mapping the ReAct
# and AE runs use, and ADaPT's demo file was verified to be our demos verbatim
# plus one trailing "> think: Task completed!" line that its executor/planner
# protocol requires.
# ============================================================================

set -uo pipefail

ADAPT=/homes/jy625/projects/AE3/external/ADaPT
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/adapt_smoke_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export ALFWORLD_DATA=/vol/gpudata/jy625-ae-data/alfworld_data

MODEL_SOURCE=${1:-/vol/gpudata/jy625-ae-data/models/Qwen3-8B}
NODE_TMP=${SLURM_TMPDIR:-/tmp/ae3_${USER}_${SLURM_JOB_ID}}
MODEL=${2:-Qwen/Qwen3-8B}
USE_SHARED_MODEL=${3:-0}
if [ "$USE_SHARED_MODEL" = "1" ]; then
    LOCAL_MODEL=$MODEL_SOURCE
else
    LOCAL_MODEL=$NODE_TMP/${MODEL##*/}
fi

EXECUTOR=${EXECUTOR:-hybrid}
RUN_FULL=${RUN_FULL:-0}
case "${SLURM_JOB_NAME:-}" in
    adapt_full*) RUN_FULL=1 ;;
esac
if [ "$RUN_FULL" = "1" ]; then
    FNAME=full_oneshot_react_${SLURM_JOB_ID}
    TASK_ARGS=(--eval-all True)
    EXPECTED_TASKS=134
    RUN_LABEL="full repeat"
else
    FNAME=smoke_oneshot_react_${SLURM_JOB_ID}
    TASK_ARGS=(--num-task-samples 2)
    EXPECTED_TASKS=12
    RUN_LABEL="12-task smoke"
fi

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() { echo "[$(date -Is)] FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/failure.log"
         [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; exit 1; }

# --- push notification -------------------------------------------------
# The job reports its own outcome to ntfy so results arrive without anyone
# watching the terminal. -m 20 and `|| true` so a slow or unreachable ntfy
# can never wedge or fail the job it is reporting on.
JOBLABEL="ADaPT ${RUN_LABEL} (one-shot react exec)"
notify() { curl -s -m 20 -H "Title: $1" -d "$2" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true; }
on_exit() {
    rc=$?
    log "job ending, exit $rc"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    [ -z "${SLURM_TMPDIR:-}" ] && rm -rf "$NODE_TMP"
    if [ "$rc" -eq 0 ] && [ -s "$LOGDIR/RESULT.txt" ]; then
        notify "OK - $JOBLABEL" "$(head -c 900 "$LOGDIR/RESULT.txt")"
    elif [ "$rc" -eq 0 ]; then
        notify "OK - $JOBLABEL" "job ${SLURM_JOB_ID:-$SLURM_ARRAY_JOB_ID} finished, exit 0. $(tail -2 "$LOGDIR/batch.log" 2>/dev/null | head -c 700)"
    else
        notify "FAILED rc=$rc - $JOBLABEL" "job ${SLURM_JOB_ID:-$SLURM_ARRAY_JOB_ID}: $(tail -4 "$LOGDIR/batch.log" 2>/dev/null | head -c 800)"
    fi
}
trap on_exit EXIT

[ -d "$ALFWORLD_DATA/json_2.1.1/valid_unseen" ] || fail "ALFWORLD_DATA has no json_2.1.1/valid_unseen"
log "ALFWORLD_DATA=$ALFWORLD_DATA"

cd "$ADAPT" || fail "cannot cd into $ADAPT"

# Keep persistent outputs off the quota-limited home filesystem.
RESULTS_ROOT="/vol/gpudata/jy625-ae-data/results/adapt_smoke_${SLURM_JOB_ID}"
DEST_DIR="$RESULTS_ROOT/${MODEL%%/*}/${MODEL##*/}/${FNAME}_${MODEL%%/*}"
mkdir -p "$DEST_DIR"
log "results will land in $DEST_DIR"

source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate

# Stage the checkpoint once from congested CEPH, then let vLLM perform all
# shard reads and mmap activity against node-local storage. Staging is kept
# outside the readiness deadline because CEPH throughput can fall below 10 MB/s.
mkdir -p "$NODE_TMP" || fail "cannot create node-local staging dir $NODE_TMP"
export XDG_CACHE_HOME="$NODE_TMP/cache"
export VLLM_CACHE_ROOT="$NODE_TMP/vllm_cache"
export TORCHINDUCTOR_CACHE_DIR="$NODE_TMP/torchinductor_cache"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" || fail "cannot create node-local compile caches"
if [ "$USE_SHARED_MODEL" != "1" ]; then
    avail_kb=$(df -Pk "$NODE_TMP" | awk 'NR==2 {print $4}')
    [ "${avail_kb:-0}" -ge 20971520 ] || fail "less than 20 GiB free in $NODE_TMP"
fi
if [ "$USE_SHARED_MODEL" = "1" ]; then
    log "using shared model snapshot $LOCAL_MODEL"
else
    log "staging model from CEPH to $LOCAL_MODEL"
    stage_start=$(date +%s)
    cp -a "$MODEL_SOURCE" "$LOCAL_MODEL" || fail "model staging failed"
    stage_seconds=$(( $(date +%s) - stage_start ))
    log "model staged locally in ${stage_seconds}s"
fi

PORT=$(( 8200 + (SLURM_JOB_ID % 600) ))
curl -s -o /dev/null -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null && fail "port ${PORT} busy"
log "starting vLLM on ${PORT}"
vllm serve "$LOCAL_MODEL" \
    --served-model-name "$MODEL" --port "${PORT}" --host 0.0.0.0 \
    --dtype half --max-model-len 40960 \
    > "$LOGDIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
READY=0
for t in $(seq 1 270); do
    curl -s -o /dev/null -w "" "$OPENAI_BASE_URL/models" 2>/dev/null && { READY=1; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM died during startup"
    sleep 10
done
[ "$READY" -eq 1 ] || fail "vLLM not ready within 45 minutes"
log "vLLM ready, OPENAI_BASE_URL=$OPENAI_BASE_URL"

# Smoke mode samples 2 per type; RUN_FULL=1 passes --eval-all True and runs
# the complete 134-task ExpeL subset through the exact same code path.
"$PY" run_alfworld.py \
    --LM "$MODEL" --exec-LM "$MODEL" \
    --eval-type ood \
    "${TASK_ARGS[@]}" \
    --executor "$EXECUTOR" --react-type oneshot \
    --max-depth 3 --max-runs 20 \
    --fname "$FNAME" \
    --results-root "$RESULTS_ROOT" \
    --verbose True \
    > "$LOGDIR/run.log" 2>&1
rc=$?
kill "$VLLM_PID" 2>/dev/null; VLLM_PID=""
[ "$rc" -eq 0 ] || fail "run_alfworld.py exited rc=$rc -- see $LOGDIR/run.log"

"$PY" - <<PYEOF 2>&1 | tee "$LOGDIR/RESULT.txt"
import json, glob
hits = glob.glob("$RESULTS_ROOT/**/*${FNAME}*/**/*.json", recursive=True)
if not hits:
    hits = glob.glob("$RESULTS_ROOT/**/${FNAME}_*.json", recursive=True)
assert hits, "no result file written"
d = json.load(open(hits[0]))
o = d["overall"]
names = ["put","clean","heat","cool","examine","puttwo"]
print("ADaPT one-shot react executor, $RUN_LABEL")
for n, s, c in zip(names, o["success"], o["count"]):
    print(f"  {n:8s} {s}/{c}")
print(f"  TOTAL    {sum(o['success'])}/{sum(o['count'])} = {100*sum(o['success'])/sum(o['count']):.1f}%")
print()
print("reference: prior full run, atomic executor = 8/134 (6.0%), all 8 in examine")
# Did decomposition actually happen this time? That was the failure last time.
plans = [t.get("plans") for k in d if k != "overall" for t in d[k].values()]
nonempty = [p for p in plans if p]
print(f"tasks with a non-empty plan: {len(nonempty)}/{len(plans)}"
      "   <-- 0 means the planner is STILL emitting nothing and this is the old bug")
assert len(plans) == $EXPECTED_TASKS, f"expected $EXPECTED_TASKS tasks, got {len(plans)}"
if nonempty:
    print("sample plan:", str(nonempty[0])[:300])
print("RUN COMPLETED")
PYEOF
grep -q "RUN COMPLETED" "$LOGDIR/RESULT.txt" || fail "result parsing failed -- see $LOGDIR/RESULT.txt"
log "done"
