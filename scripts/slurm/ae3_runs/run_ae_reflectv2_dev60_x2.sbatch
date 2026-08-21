#!/bin/bash
#SBATCH --job-name=ae_reflectv2
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_reflectv2-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_reflectv2-%j.err

# ============================================================================
# task #84: REFLECT prompt-v2 ("Nothing happens" fix) evaluation on dev60.
#
# Root cause (this session's chat-log case study, examine-type failures):
# real-trajectory replay found ALL 18/18 one-shot examine failures (both AE
# and ReAct baselines) hit an inadmissible action ("Nothing happens.") very
# early, usually from re-issuing "go to <receptacle N>" for a receptacle
# already reachable from the current position (ALFWorld groups several
# numbered receptacles, e.g. multiple drawers, under one physical "go to"
# location). REFLECT's v1 directive never told the model what "Nothing
# happens." usually means or what to do about it. This run tests ONE
# unconditional added line in REFLECT's directive (isolated workdir
# reflect_v2_workdir, ae/controllers/intervention_renderer.py, sha256=
# 13cb177ab4bfc4b08cfa24cd90f5c9506aadfe98f8d59b34582e6a65a8e78cc3) --
# REPLAN and VERIFY are untouched (diff-verified against the prompt-v2
# baseline: only the REFLECT dict entry + module docstring changed).
#
# Same base as task #82's noise run: ae_tune_dev_baseline.yaml controller
# config (== ae_full_final_nopir_v2_5.yaml), one_shot_v1.yaml demo config
# (uniform _0), seed irrelevant here (run_alfworld_devset.py has no --seed,
# dev60 order is fixed), same dev60 manifest, same model/generation params.
# 2 repeat runs (not 3, to bound GPU time) -- compared against task #82's
# existing 3x REFLECT-v1 baseline (29/25/28, mean=27.33, stdev=2.08) as the
# noise yardstick.
# ============================================================================

set -uo pipefail

WD=/vol/gpudata/jy625-ae-data/scratch_ae3/reflect_v2_workdir
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_reflectv2_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

RUN1=ae_reflectv2_dev60_rep1
RUN2=ae_reflectv2_dev60_rep2

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
# Pre-flight 1: result dirs free.
# ---------------------------------------------------------------------------
for d in "$RUN1" "$RUN2"; do
    [ -e "$WD/ae/runners/runs/$d" ] && fail "result directory already exists, refusing to overwrite: $WD/ae/runners/runs/$d"
done
log "check 1 OK: 2 result dirs free"

# ---------------------------------------------------------------------------
# Pre-flight 2: sha256 + confirm ONLY intervention_renderer.py differs from
# the prompt-v2 baseline workdir (REPLAN/VERIFY/controller/demo_config/
# agents.py all byte-identical).
# ---------------------------------------------------------------------------
BASE=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
{
    echo "=== reflect_v2_workdir ==="
    sha256sum "$WD"/alfworld_runs_ae/agents.py \
              "$WD"/alfworld_runs_ae/demo_config.py \
              "$WD"/ae/controllers/intervention_renderer.py \
              "$WD"/ae/controllers/stateful_controller.py \
              "$WD"/ae/baselines/ae_full.py \
              "$WD"/ae/runners/run_alfworld_devset.py \
              "$WD"/configs/controllers/ae_tune_dev_baseline.yaml \
              "$WD"/configs/demos/one_shot_v1.yaml \
              "$WD"/dev_set/dev60_valid_seen_manifest.json
    echo ""
    echo "=== diff vs prompt-v2 baseline (must be ONLY intervention_renderer.py) ==="
    diff -rq --exclude=runs --exclude='__pycache__' --exclude='*.pyc' "$BASE" "$WD" || true
} > "$LOGDIR/sha256.log" 2>&1

grep -q "13cb177ab4bfc4b08cfa24cd90f5c9506aadfe98f8d59b34582e6a65a8e78cc3  $WD/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch vs expected REFLECT-v2 -- see $LOGDIR/sha256.log"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $WD/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch -- see $LOGDIR/sha256.log"
DIFF_LINES=$(grep -c "^Files .* differ$\|^Only in" "$LOGDIR/sha256.log" || true)
DIFF_NONRENDERER=$(grep "^Files .* differ$\|^Only in" "$LOGDIR/sha256.log" | grep -v "intervention_renderer.py" || true)
if [ -n "$DIFF_NONRENDERER" ]; then
    fail "unexpected diff outside intervention_renderer.py vs prompt-v2 baseline: $DIFF_NONRENDERER"
fi
log "check 2 OK: sha256 recorded, ONLY intervention_renderer.py differs from prompt-v2 baseline (see $LOGDIR/sha256.log)"

# ---------------------------------------------------------------------------
# Pre-flight 3: regression suite.
# ---------------------------------------------------------------------------
cd "$WD" || fail "cannot cd into $WD"
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

log "=== 2 sequential repeat runs (60 dev tasks each, REFLECT-v2) ==="
for RUN in "$RUN1" "$RUN2"; do
    log "--- starting $RUN ---"
    "$PY" -m ae.runners.run_alfworld_devset \
        --run-name "$RUN" \
        --ae-config configs/controllers/ae_tune_dev_baseline.yaml \
        --demo-config configs/demos/one_shot_v1.yaml \
        --termination-policy legacy_early_stop \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$WD/ae/runners/runs" \
        > "$LOGDIR/${RUN}.log" 2>&1
    rc=$?
    log "--- finished $RUN rc=$rc ---"
    [ "$rc" -eq 0 ] || fail "$RUN exited non-zero ($rc) -- see $LOGDIR/${RUN}.log"
done

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
VLLM_PID=""
log "vLLM server stopped."
log "JOB COMPLETE."
