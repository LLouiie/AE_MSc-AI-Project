#!/bin/bash
#SBATCH --job-name=ae_ablation_hcpe1
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_ablation_hcpe1-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_ablation_hcpe1-%j.err

# ============================================================================
# Demo-selection ablation, task #75: practice-only paired experiment.
#
# Same code/config base as the pre-registered uniform-_0 baseline run
# (ae_full_nopir_v2_5_oldctrl_replanpromptv2_oneshot0_seed42_a40_practice,
# 58/100 for heat/cool/puttwo/examine subset -- see that run's own
# episode_log.jsonl for the _0 side of the pairing): old REPLAN-unconditional
# controller (ae_full_final_nopir_v2_5.yaml), REPLAN prompt-v2
# (intervention_renderer.py, unchanged from that run), same model/seed/
# termination-policy/task-list-order. THE ONLY VARIABLE CHANGED IS THE DEMO
# INDEX (configs/demos/one_shot_v1_index1.yaml, demo_indices=[1] instead of
# [0]), and only for heat/cool/puttwo/examine's practice-split tasks via the
# new --task-types filter (ae/runners/run_alfworld.py) -- a pure
# task-selection restriction added to the runner CLI, not a change to
# demo_config.py, agents.py, or any controller/prompt logic. put/clean are
# never run under this config so their Appendix-K-confirmed _0 demo is
# never displaced.
#
# This run does NOT touch or overwrite the uniform-_0 baseline result dirs.
# Results here are for practice-only, exploratory demo-selection PAIRING --
# per the user's explicit protocol, practice results here are descriptive/
# for-selection only, not a generalization claim (exam is not run in this
# step; task #77 will re-validate the frozen selection on the full 134).
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
SEED=42
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_ablation_hcpe1_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

RUN_NAME=ae_practice_ablation_hcpe_index1_seed42_a40

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
# Pre-flight 1: result directory must not already exist; baseline _0 result
# dirs (this run's pairing counterpart) must exist and must NOT be touched.
# ---------------------------------------------------------------------------
[ -e "$COPY/ae/runners/runs/$RUN_NAME" ] && fail "result directory already exists, refusing to overwrite: $COPY/ae/runners/runs/$RUN_NAME"
BASELINE_DIR="$COPY/ae/runners/runs/ae_full_nopir_v2_5_oldctrl_replanpromptv2_oneshot0_seed42_a40_practice"
[ -d "$BASELINE_DIR" ] || fail "pairing baseline dir missing, cannot pair: $BASELINE_DIR"
log "check 1 OK: result dir free, uniform-_0 baseline dir present for pairing at $BASELINE_DIR"

# ---------------------------------------------------------------------------
# Pre-flight 2: record SHA256 of the code this run will actually execute,
# including confirming the controller/renderer are UNCHANGED from the
# baseline run (only the runner CLI's task-selection filter is new).
# ---------------------------------------------------------------------------
{
    echo "=== COPY (shared with the AE 80/134 baseline and ReAct one-shot runs) ==="
    echo "path: $COPY"
    sha256sum "$COPY"/alfworld_runs_ae/agents.py \
              "$COPY"/alfworld_runs_ae/demo_config.py \
              "$COPY"/ae/controllers/intervention_renderer.py \
              "$COPY"/ae/controllers/stateful_controller.py \
              "$COPY"/ae/baselines/ae_full.py \
              "$COPY"/ae/runners/run_alfworld.py \
              "$COPY"/configs/controllers/ae_full_final_nopir_v2_5.yaml \
              "$COPY"/configs/demos/one_shot_v1_index1.yaml
    echo ""
    echo "=== expected intervention_renderer.py sha (prompt-v2, from replan_v2_workdir) ==="
    echo "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f"
    echo ""
    echo "=== expected stateful_controller.py sha (old REPLAN-unconditional, matches baseline run) ==="
    echo "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484"
} > "$LOGDIR/sha256.log" 2>&1

grep -q "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch vs baseline run -- see $LOGDIR/sha256.log"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch vs baseline run -- see $LOGDIR/sha256.log"
log "check 2 OK: sha256 recorded and controller/renderer confirmed byte-identical to baseline run (see $LOGDIR/sha256.log)"

# ---------------------------------------------------------------------------
# Pre-flight 3: regression suite (same set used for the baseline run).
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
# Pre-flight 4: confirm the task-type filter selects exactly the expected
# 58 practice tasks (heat=19, cool=16, puttwo=9, examine=14) before any GPU
# spend -- catches any accidental scope drift immediately.
# ---------------------------------------------------------------------------
FILTER_CHECK=$("$PY" -c "
import sys
sys.path.insert(0, 'alfworld_runs_ae')
from environment import get_practice_tasks, get_task_type
from collections import Counter
tasks = get_practice_tasks()
allowed = {'heat','cool','puttwo','examine'}
c = Counter()
for t in tasks:
    env_name = t.get('env_name') or t['gamefile'].split('/')[-3]
    tt = get_task_type(env_name)
    if tt in allowed:
        c[tt] += 1
total = sum(c.values())
print(f'heat={c[\"heat\"]} cool={c[\"cool\"]} puttwo={c[\"puttwo\"]} examine={c[\"examine\"]} total={total}')
assert c['heat'] == 19 and c['cool'] == 16 and c['puttwo'] == 9 and c['examine'] == 14 and total == 58, 'COUNT MISMATCH'
print('OK')
" 2>&1)
echo "$FILTER_CHECK" >> "$LOGDIR/batch.log"
echo "$FILTER_CHECK" | grep -q "^OK$" || fail "task-type filter count mismatch -- expected heat=19 cool=16 puttwo=9 examine=14 total=58, got: $FILTER_CHECK"
log "check 4 OK: task-type filter selects exactly 58 practice tasks (heat=19 cool=16 puttwo=9 examine=14)"

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
# Smoke test: 2 tasks (within the filtered set), not counted toward the
# formal result, deleted after.
# ---------------------------------------------------------------------------
log "=== starting smoke test (2 tasks, practice split, task-types filter) ==="
"$PY" -m ae.runners.run_alfworld \
    --baseline ae_full --split practice \
    --run-name "smoke_${RUN_NAME}" \
    --task-types heat,cool,puttwo,examine \
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

# ---------------------------------------------------------------------------
# Full practice-only ablation run: 58 tasks (heat/cool/puttwo/examine only).
# ---------------------------------------------------------------------------
log "=== starting full run: 58 practice tasks (heat/cool/puttwo/examine, demo index 1) ==="
"$PY" -m ae.runners.run_alfworld \
    --baseline ae_full --split practice \
    --run-name "$RUN_NAME" \
    --task-types heat,cool,puttwo,examine \
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
