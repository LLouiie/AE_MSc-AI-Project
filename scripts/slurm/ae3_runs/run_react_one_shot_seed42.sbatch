#!/bin/bash
#SBATCH --job-name=react_oneshot_seed42
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/react_oneshot_seed42-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/react_oneshot_seed42-%j.err

# ============================================================================
# Fully self-contained batch job: ReAct baseline, one-shot demo (demo_indices:
# [0], all six task types), seed=42, full 134-task list (100 practice + 34
# exam), termination_policy=legacy_early_stop -- matches the AE one-shot run
# (ae_full_nopir_v2_5_oldctrl_replanpromptv2_oneshot0_seed42) in every shared
# dimension (model, seed, task list/order, generation params, termination
# policy, one-shot demo config). ReAct has no controller, so it does not use
# ae_full_final_nopir_v2_5.yaml or intervention_renderer.py at all --
# --ae-config is simply not passed.
#
# Uses the SAME isolated code copy as the AE one-shot run
# (COPY = /vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42),
# which already carries the demo_config feature merged on top of the
# canonical nopir_v2.5 + REPLAN prompt-v2 code base. ReAct's own code path
# (ALFWorldAgent with controller=None) never touches the prompt-v2-specific
# additions (they are no-ops when self.controller is None, by design) --
# this guarantees ReAct and AE build byte-identical initial base prompts
# from the exact same demo_config.py + agents.py, per
# test_demo_config.py::test_react_and_ae_build_identical_initial_base_prompt
# (already verified passing in this copy before the AE run was launched).
#
# Starts and tears down its OWN vLLM server -- never touches job 270382
# (which is about to hit its 3-day SLURM time limit) or any other job.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
SEED=42
LOGDIR=/vol/gpudata/jy625-ae-data/logs/react_oneshot_seed42_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

RUN_PREFIX=react_oneshot0_seed42

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
# Pre-flight 1: result directories must not already exist.
# ---------------------------------------------------------------------------
for d in "${RUN_PREFIX}_a40_practice" "${RUN_PREFIX}_a40_exam"; do
    [ -e "$COPY/ae/runners/runs/$d" ] && fail "result directory already exists, refusing to overwrite: $COPY/ae/runners/runs/$d"
done
log "check 1 OK: result directories do not already exist"

# ---------------------------------------------------------------------------
# Pre-flight 2: record SHA256 of the code this run will actually execute.
# ---------------------------------------------------------------------------
{
    echo "=== COPY (shared with the AE one-shot run) ==="
    echo "path: $COPY"
    sha256sum "$COPY"/alfworld_runs_ae/agents.py \
              "$COPY"/alfworld_runs_ae/demo_config.py \
              "$COPY"/ae/baselines/react.py \
              "$COPY"/ae/runners/run_alfworld.py \
              "$COPY"/configs/demos/one_shot_v1.yaml
} > "$LOGDIR/sha256.log" 2>&1
log "check 2 OK: sha256 recorded to $LOGDIR/sha256.log"

# ---------------------------------------------------------------------------
# Pre-flight 3: regression suite relevant to react + demo_config.
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

# ---------------------------------------------------------------------------
# Start this job's OWN vLLM server.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Smoke test: 2 tasks, not counted toward the formal result, deleted after.
# ---------------------------------------------------------------------------
log "=== starting smoke test (2 tasks, practice split) ==="
"$PY" -m ae.runners.run_alfworld \
    --baseline react --split practice \
    --run-name "smoke_${RUN_PREFIX}" \
    --limit 2 --seed "$SEED" \
    --demo-config configs/demos/one_shot_v1.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/smoke.log" 2>&1
rc_smoke=$?
[ "$rc_smoke" -eq 0 ] || fail "smoke test exited non-zero ($rc_smoke) -- see $LOGDIR/smoke.log"
grep -q "PILOT DONE" "$LOGDIR/smoke.log" || fail "smoke test did not report PILOT DONE -- see $LOGDIR/smoke.log"
rm -rf "$COPY/ae/runners/runs/smoke_${RUN_PREFIX}"
log "smoke test OK (2/2 tasks completed, rc=0), smoke result dir removed"

# ---------------------------------------------------------------------------
# Full 134-task run: practice + exam in parallel against the same server.
# ---------------------------------------------------------------------------
log "=== starting full run: practice(100) + exam(34) ==="
"$PY" -m ae.runners.run_alfworld \
    --baseline react --split practice \
    --run-name "${RUN_PREFIX}_a40_practice" \
    --seed "$SEED" \
    --demo-config configs/demos/one_shot_v1.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/practice.log" 2>&1 &
pid_p=$!
"$PY" -m ae.runners.run_alfworld \
    --baseline react --split exam \
    --run-name "${RUN_PREFIX}_a40_exam" \
    --seed "$SEED" \
    --demo-config configs/demos/one_shot_v1.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/exam.log" 2>&1 &
pid_e=$!
wait "$pid_p"; rc_p=$?
wait "$pid_e"; rc_e=$?
log "=== finished full run: rc_practice=$rc_p rc_exam=$rc_e ==="
[ "$rc_p" -eq 0 ] || fail "practice run exited non-zero ($rc_p) -- see $LOGDIR/practice.log"
[ "$rc_e" -eq 0 ] || fail "exam run exited non-zero ($rc_e) -- see $LOGDIR/exam.log"

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
VLLM_PID=""
log "vLLM server stopped."
log "JOB COMPLETE."
