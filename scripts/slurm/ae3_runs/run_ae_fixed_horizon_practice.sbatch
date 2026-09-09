#!/bin/bash
#SBATCH --job-name=ae_fixedhorizon
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_fixedhorizon-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_fixedhorizon-%j.err

# ============================================================================
# task: one-shot AE, practice split (100 tasks), --termination-policy
# fixed_horizon instead of the official run's legacy_early_stop. No code
# change -- this is a pre-existing CLI flag in run_alfworld.py, so this runs
# directly against the official copy (one_shot_oldcontroller_promptv2_seed42),
# same controller/demo config/seed as the frozen 80/134 result.
#
# Rationale: legacy_early_stop force-ends an episode the moment
# consecutive_exact_action_repeat fires past its grace window, with
# termination_reason="exhausted_repeated" and success=0 by construction.
# fixed_horizon removes that early exit -- the episode keeps running (same
# controller/REFLECT/REPLAN/VERIFY logic, unaffected) up to MAX_STEPS=50 or
# a real win/env-terminal. Checked in the frozen practice log
# (ae_full_nopir_v2_5_oldctrl_replanpromptv2_oneshot0_seed42_a40_practice):
# 20/100 practice episodes (6/34 exam) currently end via exhausted_repeated,
# i.e. guaranteed failures under legacy_early_stop. Since early-stopping can
# only end an already-failing trajectory sooner, never turn a would-be
# success into a failure, fixed_horizon should weakly dominate on success
# rate (same or better, never worse) -- the only cost is more steps/LLM
# calls for those episodes. This is a structural argument, not a prompt-
# wording gamble like REFLECT-v2, so a real (not noise) improvement is
# plausible even from a single run if it recovers a nontrivial fraction of
# those 20 practice cases.
#
# Per the practice-first/exam-once discipline (see
# audit_reports/ae_oneshot_demo_selection_ablation_report.md and this
# session's agreement to simplify by using the 134-task practice/exam split
# directly instead of the separate dev60 sandbox): PRACTICE ONLY this run.
# Exam stays untouched until/unless this shows a real, above-noise gain on
# practice -- do not run exam from this script.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_fixedhorizon_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

RUN_NAME=ae_fixedhorizon_oneshot0_seed42_practice

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() {
    echo "[$(date -Is)] PREFLIGHT FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/preflight_failure.log"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}
# Notification fires from the compute node itself, inside this script's own
# EXIT trap -- deliberately NOT from an external polling loop, which dies
# whenever the driving session restarts (happened three times on this
# project: jobs 271479, 273214/273236, 273526/273531/273532).
notify() {
    curl -s -m 20 -d "$1" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
}
trap 'rc=$?; log "job ending, exit code $rc"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; notify "fixed_horizon practice (job ${SLURM_JOB_ID}) ended, exit=$rc, logdir=$LOGDIR"' EXIT

log "job started on $(hostname), SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

# ---------------------------------------------------------------------------
# Pre-flight 1: result dir free.
# ---------------------------------------------------------------------------
[ -e "$COPY/ae/runners/runs/$RUN_NAME" ] && fail "result directory already exists, refusing to overwrite: $COPY/ae/runners/runs/$RUN_NAME"
log "check 1 OK: result dir free"

# ---------------------------------------------------------------------------
# Pre-flight 2: sha256 -- confirm controller/renderer/demo-config are the
# exact same as the frozen 80/134 official run (only the CLI flag differs,
# no code path may have changed since).
# ---------------------------------------------------------------------------
{
    echo "=== COPY ==="
    sha256sum "$COPY"/alfworld_runs_ae/agents.py \
              "$COPY"/alfworld_runs_ae/demo_config.py \
              "$COPY"/ae/controllers/intervention_renderer.py \
              "$COPY"/ae/controllers/stateful_controller.py \
              "$COPY"/ae/baselines/ae_full.py \
              "$COPY"/ae/runners/run_alfworld.py \
              "$COPY"/configs/controllers/ae_full_final_nopir_v2_5.yaml \
              "$COPY"/configs/demos/one_shot_v1.yaml
} > "$LOGDIR/sha256.log" 2>&1
grep -q "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch -- see $LOGDIR/sha256.log"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch -- see $LOGDIR/sha256.log"
log "check 2 OK: sha256 recorded, controller/renderer confirmed byte-identical to the frozen 80/134 run (see $LOGDIR/sha256.log)"

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

log "=== practice split (100 tasks), fixed_horizon ==="
"$PY" -m ae.runners.run_alfworld \
    --baseline ae_full --split practice \
    --run-name "$RUN_NAME" \
    --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --demo-config configs/demos/one_shot_v1.yaml \
    --termination-policy fixed_horizon \
    --seed 42 \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/${RUN_NAME}.log" 2>&1
rc=$?
log "--- finished $RUN_NAME rc=$rc ---"
[ "$rc" -eq 0 ] || fail "$RUN_NAME exited non-zero ($rc) -- see $LOGDIR/${RUN_NAME}.log"

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
VLLM_PID=""
log "vLLM server stopped."
log "JOB COMPLETE."
