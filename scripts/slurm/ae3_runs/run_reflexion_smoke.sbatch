#!/bin/bash
#SBATCH --job-name=reflex_smoke
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/reflex_smoke-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/reflex_smoke-%j.err

# ============================================================================
# SMOKE TEST for --baseline reflexion on ALFWorld with Qwen3-8B.
#
# `--baseline reflexion` is wired into ae/runners/run_alfworld.py (line 140)
# and ae/baselines/reflexion.py exists, but NO sbatch in this project has ever
# used it -- every ALFWorld run so far was react or ae_full. So this path is
# untested at scale. Burning 10 repeats x 134 tasks on an untested code path
# would be the expensive way to discover a bug, hence 12 tasks first.
#
# Demo handling is shared with ReAct: agents.py::_build_base_prompt() pulls
# `react_{task_type}_{idx}` from alfworld_3prompts.json driven by --demo-config,
# for reflexion exactly as for react. So passing the same demo config gives
# byte-identical one-shot examples -- nothing extra to align.
#
# What this must prove: reflexion runs to completion, the reflect step fires on
# a failure and its text lands in the log, and trials_used > 1 somewhere.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/reflex_smoke_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL_SOURCE=/vol/gpudata/jy625-ae-data/models/Qwen3-8B
NODE_TMP=${SLURM_TMPDIR:-/tmp/ae3_${USER}_${SLURM_JOB_ID}}
LOCAL_MODEL=$NODE_TMP/Qwen3-8B
RUN_FULL=0
case "${SLURM_JOB_NAME:-}" in
    reflex_full*) RUN_FULL=1 ;;
esac

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() { echo "[$(date -Is)] FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/failure.log"
         [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; exit 1; }

# --- push notification -------------------------------------------------
# The job reports its own outcome to ntfy so results arrive without anyone
# watching the terminal. -m 20 and `|| true` so a slow or unreachable ntfy
# can never wedge or fail the job it is reporting on.
if [ "$RUN_FULL" = "1" ]; then
    JOBLABEL="Reflexion full repeat (134 tasks, ALFWorld)"
else
    JOBLABEL="Reflexion smoke (12 tasks, ALFWorld)"
fi
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

cd "$COPY" || fail "cannot cd into $COPY"
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate

# Stage the checkpoint once from congested CEPH, then let vLLM perform all
# shard reads and mmap activity against node-local storage. Staging is kept
# outside the readiness deadline because CEPH throughput can fall below 10 MB/s.
mkdir -p "$NODE_TMP" || fail "cannot create node-local staging dir $NODE_TMP"
avail_kb=$(df -Pk "$NODE_TMP" | awk 'NR==2 {print $4}')
[ "${avail_kb:-0}" -ge 20971520 ] || fail "less than 20 GiB free in $NODE_TMP"
log "staging model from CEPH to $LOCAL_MODEL"
stage_start=$(date +%s)
cp -a "$MODEL_SOURCE" "$LOCAL_MODEL" || fail "model staging failed"
stage_seconds=$(( $(date +%s) - stage_start ))
log "model staged locally in ${stage_seconds}s"

PORT=$(( 8800 + (SLURM_JOB_ID % 600) ))
curl -s -o /dev/null -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null && fail "port ${PORT} busy"
log "starting vLLM on ${PORT}"
vllm serve "$LOCAL_MODEL" \
    --served-model-name Qwen/Qwen3-8B --port "${PORT}" --host 0.0.0.0 \
    > "$LOGDIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
BASE_URL="http://localhost:${PORT}/v1"
READY=0
for t in $(seq 1 270); do
    curl -s -o /dev/null -w "" "$BASE_URL/models" 2>/dev/null && { READY=1; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM died during startup"
    sleep 10
done
[ "$READY" -eq 1 ] || fail "vLLM not ready within 45 minutes"
log "vLLM ready"

if [ "$RUN_FULL" = "1" ]; then
    groups=("put,examine" "heat,cool,clean" "puttwo")
    group_names=("a" "b" "c")
    demo_configs=("configs/demos/one_shot_v1.yaml"
                  "configs/demos/one_shot_v1_index2.yaml"
                  "configs/demos/one_shot_v1_index1.yaml")
    : > "$LOGDIR/run.log"
    for i in 0 1 2; do
        for split in practice exam; do
            run_name="reflexion_full_${SLURM_JOB_ID}_${group_names[$i]}_${split}"
            log "running $run_name"
            "$PY" -m ae.runners.run_alfworld \
                --baseline reflexion --split "$split" \
                --task-types "${groups[$i]}" \
                --run-name "$run_name" \
                --max-trials 4 \
                --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
                --demo-config "${demo_configs[$i]}" \
                --termination-policy legacy_early_stop \
                --seed 42 \
                --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
                --output-dir "$COPY/ae/runners/runs" \
                >> "$LOGDIR/run.log" 2>&1
            rc=$?
            [ "$rc" -eq 0 ] || fail "$run_name exited rc=$rc -- see $LOGDIR/run.log"
        done
    done
else
    "$PY" -m ae.runners.run_alfworld \
        --baseline reflexion --split practice \
        --run-name reflexion_smoke_${SLURM_JOB_ID} \
        --limit 12 \
        --max-trials 4 \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
        --demo-config configs/demos/one_shot_v1.yaml \
        --termination-policy legacy_early_stop \
        --seed 42 \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$COPY/ae/runners/runs" \
        > "$LOGDIR/run.log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || fail "runner exited rc=$rc -- see $LOGDIR/run.log"
fi
kill "$VLLM_PID" 2>/dev/null; VLLM_PID=""

"$PY" - <<PYEOF 2>&1 | tee "$LOGDIR/RESULT.txt"
import glob, json
if $RUN_FULL:
    paths = glob.glob("$COPY/ae/runners/runs/reflexion_full_${SLURM_JOB_ID}_*/episode_log.jsonl")
    expected = 134
else:
    paths = ["$COPY/ae/runners/runs/reflexion_smoke_${SLURM_JOB_ID}/episode_log.jsonl"]
    expected = 12
rows = [json.loads(line) for p in paths for line in open(p)]
print(f"sub-runs          : {len(paths)}")
print(f"episodes           : {len(rows)}")
print(f"score              : {sum(r['success'] for r in rows)}/{len(rows)}")
tu = [r.get('trials_used') for r in rows]
print(f"mean trials/task   : {sum(tu)/len(tu):.3f}")
nrefl = sum(1 for r in rows if r.get('reflections'))
print(f"episodes w/ reflect: {nrefl}")
ex = next((r for r in rows if r.get('reflections')), None)
if ex:
    print("sample reflection  :", str(ex['reflections'][0])[:300])
assert len(rows) == expected, f"expected {expected} episodes"
assert any(t and t > 1 for t in tu), "NO episode ever retried -- reflexion loop never engaged"
assert nrefl > 0, "NO reflection text produced"
print("RUN COMPLETED")
PYEOF
grep -q "RUN COMPLETED" "$LOGDIR/RESULT.txt" || fail "result assertions failed -- see $LOGDIR/RESULT.txt"
log "RUN COMPLETED"
