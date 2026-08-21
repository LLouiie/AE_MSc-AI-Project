#!/bin/bash
#SBATCH --job-name=ae_sustained
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_sustained-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_sustained-%j.err

# ============================================================================
# task #85: sustained-only REPLAN routing (see
# audit_reports/ae_full_sustained_v1_134_report.md, section 0 for the code
# diff) evaluated on the one-shot dev60 set, 3x repeat.
#
# That prior report tested this same single-variable routing change (REPLAN's
# frustration_high+confidence_low branch in _select_intervention() only
# arms once steps_since_meaningful_change >= patch_duration_steps, otherwise
# falls through to REFLECT/VERIFY) under a DIFFERENT config (cooldown5,
# two-shot demos, 134-task practice+exam), single run only, and explicitly
# said it needed a repeat-verification pass (like dev60 3x) before any
# real vs noise conclusion could be drawn -- never followed up. This run
# does that follow-up under the one-shot line specifically.
#
# Isolated workdir sustained_only_workdir/ (rsync copy of
# one_shot_oldcontroller_promptv2_seed42), single file changed:
# ae/controllers/stateful_controller.py, sha256=
# f0b53c9bf5180f2714c122c8abc49dde7726b708cbcd8d47fa2d73f57f94908d
# (diff-verified: this is the ONLY file differing from the one-shot
# baseline copy). Same controller config (ae_full_final_nopir_v2_5.yaml)
# and demo config (one_shot_v1.yaml, uniform _0) as task #82's noise run
# and the sweep7 baseline arm, so results are directly comparable to the
# now-established REFLECT-v1 dev60 baseline distribution: 29, 25, 28, 33
# (mean=28.75, stdev=3.30, n=4). Per the newly-added standing constraint
# (>=3 repeats before accepting/rejecting a dev60 config), this run does
# 3 repeats.
# ============================================================================

set -uo pipefail

WD=/vol/gpudata/jy625-ae-data/scratch_ae3/sustained_only_workdir
BASE=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_sustained_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

RUN1=ae_sustained_dev60_rep1
RUN2=ae_sustained_dev60_rep2
RUN3=ae_sustained_dev60_rep3

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
for d in "$RUN1" "$RUN2" "$RUN3"; do
    [ -e "$WD/ae/runners/runs/$d" ] && fail "result directory already exists, refusing to overwrite: $WD/ae/runners/runs/$d"
done
log "check 1 OK: 3 result dirs free"

# ---------------------------------------------------------------------------
# Pre-flight 2: sha256 + confirm ONLY stateful_controller.py differs from the
# one-shot baseline workdir (REPLAN/REFLECT/VERIFY renderer, controller
# config, demo_config, agents.py all byte-identical).
# ---------------------------------------------------------------------------
{
    echo "=== sustained_only_workdir ==="
    sha256sum "$WD"/alfworld_runs_ae/agents.py \
              "$WD"/alfworld_runs_ae/demo_config.py \
              "$WD"/ae/controllers/intervention_renderer.py \
              "$WD"/ae/controllers/stateful_controller.py \
              "$WD"/ae/baselines/ae_full.py \
              "$WD"/ae/runners/run_alfworld_devset.py \
              "$WD"/configs/controllers/ae_full_final_nopir_v2_5.yaml \
              "$WD"/configs/demos/one_shot_v1.yaml \
              "$WD"/dev_set/dev60_valid_seen_manifest.json
    echo ""
    echo "=== diff vs one-shot baseline (must be ONLY stateful_controller.py) ==="
    diff -rq --exclude=runs --exclude='__pycache__' --exclude='*.pyc' "$BASE" "$WD" || true
} > "$LOGDIR/sha256.log" 2>&1

grep -q "f0b53c9bf5180f2714c122c8abc49dde7726b708cbcd8d47fa2d73f57f94908d  $WD/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch vs expected sustained-only version -- see $LOGDIR/sha256.log"
grep -q "c4824c0606a87eedc32aa96564c9ddcedafc1fd11a7b212a3131382fe43367ba  $WD/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch -- see $LOGDIR/sha256.log"
DIFF_NONCONTROLLER=$(grep "^Files .* differ$\|^Only in" "$LOGDIR/sha256.log" | grep -v "stateful_controller.py" || true)
if [ -n "$DIFF_NONCONTROLLER" ]; then
    fail "unexpected diff outside stateful_controller.py vs one-shot baseline: $DIFF_NONCONTROLLER"
fi
log "check 2 OK: sha256 recorded, ONLY stateful_controller.py differs from one-shot baseline (see $LOGDIR/sha256.log)"

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

log "=== 3 sequential repeat runs (60 dev tasks each, sustained-only REPLAN) ==="
for RUN in "$RUN1" "$RUN2" "$RUN3"; do
    log "--- starting $RUN ---"
    "$PY" -m ae.runners.run_alfworld_devset \
        --run-name "$RUN" \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
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
