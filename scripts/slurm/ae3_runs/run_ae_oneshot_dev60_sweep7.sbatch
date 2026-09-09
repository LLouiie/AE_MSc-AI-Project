#!/bin/bash
#SBATCH --job-name=ae_1shot_sweep
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_1shot_sweep-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_1shot_sweep-%j.err

# ============================================================================
# One-shot controller hyperparameter resweep on dev60 (task #83, follow-up to
# the noise-quantification job #82, which found stdev=2.08/range=4 successes
# across 3 identical-config repeat runs on this same 60-task set -- that
# noise floor is the yardstick every config difference below must be judged
# against).
#
# Same 7-config single-variable design as the original two-shot dev60 sweep
# (task #59, see audit_reports/dev60_and_sweep_freeze.md): baseline +
# grace{2,5} + patch{2,5} + cooldown{1,5}, configs copied byte-identical
# from the live repo (configs/controllers/ae_tune_dev_*.yaml, diff-verified
# single-variable-only vs baseline). ONLY new variable vs the original
# sweep: --demo-config configs/demos/one_shot_v1.yaml (uniform _0,
# one-shot) instead of the two-shot default, and this line's controller
# code (old REPLAN-unconditional + REPLAN prompt-v2,
# stateful_controller.py sha=
# 6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484,
# intervention_renderer.py sha=
# d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f)
# instead of the original sweep's frozen nopir_v2.5 controller sha
# (35249c69...) -- i.e. this sweep answers "what's the best
# cooldown/patch/grace for ONE-SHOT AE with this session's controller+
# prompt", not a re-verification of the original two-shot sweep.
#
# Single run per config (matches the original sweep's first pass -- see
# task #65 for how repeat-verification was added afterward for promising
# configs there). Runs in 4 sequential pairs/singleton against one shared
# vLLM server to bound wall-clock time.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_1shot_sweep_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

CONFIGS="ae_tune_dev_baseline ae_tune_dev_grace2 ae_tune_dev_grace5 ae_tune_dev_patch2 ae_tune_dev_patch5 ae_tune_dev_cooldown1 ae_tune_dev_cooldown5"
RUN_PREFIX=ae_oneshot_dev60_sweep

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() {
    echo "[$(date -Is)] PREFLIGHT FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/preflight_failure.log"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}
trap 'log "job ending, exit code $?"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null' EXIT

log "job started on $(hostname), SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

# ---------------------------------------------------------------------------
# Pre-flight 1: none of the 7 result dirs already exist.
# ---------------------------------------------------------------------------
for c in $CONFIGS; do
    d="${RUN_PREFIX}_${c#ae_tune_dev_}"
    [ -e "$COPY/ae/runners/runs/$d" ] && fail "result directory already exists, refusing to overwrite: $COPY/ae/runners/runs/$d"
done
log "check 1 OK: all 7 result dirs free"

# ---------------------------------------------------------------------------
# Pre-flight 2: sha256 of code + each of the 7 configs (single-variable-diff
# confirmed already at authoring time; recorded here for the record).
# ---------------------------------------------------------------------------
{
    echo "=== COPY ==="
    sha256sum "$COPY"/alfworld_runs_ae/agents.py \
              "$COPY"/alfworld_runs_ae/demo_config.py \
              "$COPY"/ae/controllers/intervention_renderer.py \
              "$COPY"/ae/controllers/stateful_controller.py \
              "$COPY"/ae/baselines/ae_full.py \
              "$COPY"/ae/runners/run_alfworld_devset.py \
              "$COPY"/configs/demos/one_shot_v1.yaml
    for c in $CONFIGS; do
        sha256sum "$COPY/configs/controllers/${c}.yaml"
    done
} > "$LOGDIR/sha256.log" 2>&1
grep -q "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch -- see $LOGDIR/sha256.log"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch -- see $LOGDIR/sha256.log"
log "check 2 OK: sha256 recorded (see $LOGDIR/sha256.log)"

# ---------------------------------------------------------------------------
# Pre-flight 3: regression suite.
# ---------------------------------------------------------------------------
cd "$COPY" || fail "cannot cd into $COPY"
for f in alfworld_runs_ae/tests/test_demo_config.py \
         alfworld_runs_ae/tests/test_alfworld_action_normalize.py \
         alfworld_runs_ae/tests/test_output_parser.py \
         alfworld_runs_ae/tests/test_prompt_budget.py \
         alfworld_runs_ae/tests/test_put_move_compat_integration.py \
         alfworld_runs_ae/tests/test_environment_anchor.py; do
    out=$("$PY" "$f" 2>&1)
    { echo "=== $f ==="; echo "$out"; } >> "$LOGDIR/regression.log"
    line=$(echo "$out" | tail -1)
    echo "$line" | grep -qE '^[0-9]+ passed, 0 failed$' \
        || fail "regression: $f did not fully pass ($line) -- see $LOGDIR/regression.log"
done
log "check 3 OK: regression suite passed (see $LOGDIR/regression.log)"

log "ALL PRE-FLIGHT CHECKS PASSED. Starting vLLM server."

source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /vol/gpudata/jy625-ae-data/models/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --port 8000 --host 0.0.0.0 \
    > "$LOGDIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
log "vLLM server starting, pid=$VLLM_PID, log=$LOGDIR/vllm_server.log"

BASE_URL="http://localhost:8000/v1"
READY=0
for i in $(seq 1 60); do
    if curl -s -o /dev/null -w "" "$BASE_URL/models" 2>/dev/null; then
        READY=1
        break
    fi
    kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM server process died during startup (pid=$VLLM_PID) -- see $LOGDIR/vllm_server.log"
    sleep 10
done
[ "$READY" -eq 1 ] || fail "vLLM server did not become ready within 10 minutes -- see $LOGDIR/vllm_server.log"
log "vLLM server ready at $BASE_URL"

run_one() {
    local cfg=$1
    local run_name="${RUN_PREFIX}_${cfg#ae_tune_dev_}"
    "$PY" -m ae.runners.run_alfworld_devset \
        --run-name "$run_name" \
        --ae-config "configs/controllers/${cfg}.yaml" \
        --demo-config configs/demos/one_shot_v1.yaml \
        --termination-policy legacy_early_stop \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$COPY/ae/runners/runs" \
        > "$LOGDIR/${run_name}.log" 2>&1
}

log "=== stage 1: baseline + grace2 (parallel) ==="
run_one ae_tune_dev_baseline & p1=$!
run_one ae_tune_dev_grace2 & p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
log "stage 1 done: baseline=$r1 grace2=$r2"
[ "$r1" -eq 0 ] || fail "ae_tune_dev_baseline run failed (rc=$r1)"
[ "$r2" -eq 0 ] || fail "ae_tune_dev_grace2 run failed (rc=$r2)"

log "=== stage 2: grace5 + patch2 (parallel) ==="
run_one ae_tune_dev_grace5 & p1=$!
run_one ae_tune_dev_patch2 & p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
log "stage 2 done: grace5=$r1 patch2=$r2"
[ "$r1" -eq 0 ] || fail "ae_tune_dev_grace5 run failed (rc=$r1)"
[ "$r2" -eq 0 ] || fail "ae_tune_dev_patch2 run failed (rc=$r2)"

log "=== stage 3: patch5 + cooldown1 (parallel) ==="
run_one ae_tune_dev_patch5 & p1=$!
run_one ae_tune_dev_cooldown1 & p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
log "stage 3 done: patch5=$r1 cooldown1=$r2"
[ "$r1" -eq 0 ] || fail "ae_tune_dev_patch5 run failed (rc=$r1)"
[ "$r2" -eq 0 ] || fail "ae_tune_dev_cooldown1 run failed (rc=$r2)"

log "=== stage 4: cooldown5 (solo) ==="
run_one ae_tune_dev_cooldown5; r1=$?
log "stage 4 done: cooldown5=$r1"
[ "$r1" -eq 0 ] || fail "ae_tune_dev_cooldown5 run failed (rc=$r1)"

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
VLLM_PID=""
log "vLLM server stopped."
log "JOB COMPLETE."
