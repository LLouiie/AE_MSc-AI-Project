#!/bin/bash
#SBATCH --job-name=ae_putclean1
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_putclean1-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_putclean1-%j.err

# ============================================================================
# User-requested sanity check (task #79, 2026-08-12), NOT part of the
# original pre-registered demo-selection ablation (task #74-78): also
# empirically check _0 vs _1 for put/clean on practice, even though these
# two types were originally excluded from selection (fixed at _0 due to
# ReflAct Appendix K verbatim-match evidence -- see
# alfworld_runs_ae/demo_config.py's module docstring). This run produces
# ONLY the _1 side (42 tasks: put=18, clean=24); the _0 side is the
# existing uniform-_0 baseline practice run
# (ae_full_nopir_v2_5_oldctrl_replanpromptv2_oneshot0_seed42_a40_practice).
#
# Exploratory/confirmatory only -- does NOT change the already-frozen
# mixed-demo mapping (configs/demos/frozen_mixed_demo_ablation_seed42.yaml,
# put=0 clean=0) used by the concurrently-running task #77 GPU job
# (ae_mixed_demo_group{A,B}_*), which was frozen before this run started
# and is not revised by it. Same code base, controller, prompt-v2, seed,
# termination policy as every other run this session.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
SEED=42
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_putclean1_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

RUN_NAME=ae_practice_putclean_index1_seed42_a40

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
# Pre-flight 1: result dir free; pairing baseline dir present.
# ---------------------------------------------------------------------------
[ -e "$COPY/ae/runners/runs/$RUN_NAME" ] && fail "result directory already exists, refusing to overwrite: $COPY/ae/runners/runs/$RUN_NAME"
BASELINE_DIR="$COPY/ae/runners/runs/ae_full_nopir_v2_5_oldctrl_replanpromptv2_oneshot0_seed42_a40_practice"
[ -d "$BASELINE_DIR" ] || fail "pairing baseline dir missing, cannot pair: $BASELINE_DIR"
log "check 1 OK: result dir free, uniform-_0 baseline dir present for pairing at $BASELINE_DIR"

# ---------------------------------------------------------------------------
# Pre-flight 2: sha256 + confirm controller/renderer byte-identical to the
# baseline run (only variable is demo index for put/clean).
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
              "$COPY"/configs/demos/one_shot_v1_index1.yaml
} > "$LOGDIR/sha256.log" 2>&1
grep -q "c4824c0606a87eedc32aa96564c9ddcedafc1fd11a7b212a3131382fe43367ba  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch -- see $LOGDIR/sha256.log"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch -- see $LOGDIR/sha256.log"
log "check 2 OK: sha256 recorded, controller/renderer confirmed byte-identical to baseline run"

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

# ---------------------------------------------------------------------------
# Pre-flight 4: confirm task-type filter selects exactly 42 practice tasks
# (put=18, clean=24).
# ---------------------------------------------------------------------------
FILTER_CHECK=$("$PY" -c "
import sys
sys.path.insert(0, 'alfworld_runs_ae')
from environment import get_practice_tasks, get_task_type
from collections import Counter
tasks = get_practice_tasks()
allowed = {'put','clean'}
c = Counter()
for t in tasks:
    env_name = t.get('env_name') or t['gamefile'].split('/')[-3]
    tt = get_task_type(env_name)
    if tt in allowed:
        c[tt] += 1
total = sum(c.values())
print(f'put={c[\"put\"]} clean={c[\"clean\"]} total={total}')
assert c['put'] == 18 and c['clean'] == 24 and total == 42, 'COUNT MISMATCH'
print('OK')
" 2>&1)
echo "$FILTER_CHECK" >> "$LOGDIR/batch.log"
echo "$FILTER_CHECK" | grep -q "^OK$" || fail "task-type filter count mismatch -- expected put=18 clean=24 total=42, got: $FILTER_CHECK"
log "check 4 OK: task-type filter selects exactly 42 practice tasks (put=18 clean=24)"

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

log "=== starting smoke test (2 tasks, put/clean, practice split) ==="
"$PY" -m ae.runners.run_alfworld \
    --baseline ae_full --split practice \
    --run-name "smoke_${RUN_NAME}" \
    --task-types put,clean \
    --limit 2 --seed "$SEED" \
    --demo-config configs/demos/one_shot_v1_index1.yaml \
    --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/smoke.log" 2>&1
rc_smoke=$?
[ "$rc_smoke" -eq 0 ] || fail "smoke test exited non-zero ($rc_smoke) -- see $LOGDIR/smoke.log"
grep -q "PILOT DONE" "$LOGDIR/smoke.log" || fail "smoke test did not report PILOT DONE -- see $LOGDIR/smoke.log"
rm -rf "$COPY/ae/runners/runs/smoke_${RUN_NAME}"
log "smoke test OK (2/2 tasks completed, rc=0), smoke result dir removed"

log "=== starting full run: 42 practice tasks (put/clean, demo index 1) ==="
"$PY" -m ae.runners.run_alfworld \
    --baseline ae_full --split practice \
    --run-name "$RUN_NAME" \
    --task-types put,clean \
    --seed "$SEED" \
    --demo-config configs/demos/one_shot_v1_index1.yaml \
    --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/run.log" 2>&1
rc=$?
log "=== finished full run: rc=$rc ==="
[ "$rc" -eq 0 ] || fail "run exited non-zero ($rc) -- see $LOGDIR/run.log"

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
VLLM_PID=""
log "vLLM server stopped."
log "JOB COMPLETE."
