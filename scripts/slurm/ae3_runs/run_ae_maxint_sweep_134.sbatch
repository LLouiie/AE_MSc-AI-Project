#!/bin/bash
#SBATCH --job-name=ae_maxint
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_maxint-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_maxint-%j.err

# ============================================================================
# max_interventions sweep (6 and 12 vs the frozen baseline's 3), full 134.
#
# NO CODE CHANGE -- runs against the official one-shot copy with the frozen v1
# renderer; the only thing that varies is one integer in the controller yaml.
# Single-variable verified: `diff` of the two new configs against
# ae_full_final_nopir_v2_5.yaml shows exactly one changed line each
# (`max_interventions: 3` -> 6 / 12).
#
# WHY THIS AXIS (practice-log analysis of the frozen 80/134 run):
#   - 38 of the 42 practice failures EXHAUST the 3-intervention budget, and
#     then run a median of 19 further steps (max 38) with NO directive active.
#   - Across all steps inside failing episodes, only 22.1% have any directive
#     active; 74.5% of the 591 inadmissible actions in those episodes happen
#     while nothing is active at all.
#   - 38 of the 58 successes use ZERO interventions, so raising the cap should
#     barely perturb episodes that are already going well -- the change is
#     concentrated on exactly the episodes currently being abandoned.
#   - max_interventions was NOT in the one-shot dev60 sweep (task #83 varied
#     grace/patch/cooldown only), so this is an untested axis on this line.
#
# Interpretation limits, fixed BEFORE seeing any result:
#   - Per-type run-to-run noise on 134 is large: the demo ablation's heat arm
#     scored 14 then 8 on two runs of an IDENTICAL config (23 tasks). A single
#     draw per config here can identify a promising direction but cannot on its
#     own establish a real effect; any config that looks good needs repeats.
#   - More interventions is not automatically better: directives can mislead
#     (see audit_reports/ae_reflect_prompt_v2_rejected.md), so a null or
#     negative result here is a genuine possibility, not a bug.
#   - Splits: practice AND exam, per the user's explicit instruction to tune
#     against all 134. Resulting numbers are fitted-on-134, NOT generalization.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_maxint_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() {
    echo "[$(date -Is)] PREFLIGHT FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/preflight_failure.log"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}
notify() {
    curl -s -m 20 -d "$1" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
}
trap 'rc=$?; log "job ending, exit code $rc"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; notify "max_interventions sweep 134 (job ${SLURM_JOB_ID}) ended, exit=$rc, logdir=$LOGDIR"' EXIT

log "job started on $(hostname), SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

for N in 6 12; do
  for s in practice exam; do
    d="ae_maxint${N}_oneshot0_seed42_${s}"
    [ -e "$COPY/ae/runners/runs/$d" ] && fail "result dir already exists: $d"
  done
done
log "check 1 OK: 4 result dirs free"

# sha256: code must be byte-identical to the frozen 80/134 run -- ONLY the
# controller yaml's max_interventions may differ.
{
    sha256sum "$COPY"/alfworld_runs_ae/agents.py \
              "$COPY"/alfworld_runs_ae/demo_config.py \
              "$COPY"/ae/controllers/intervention_renderer.py \
              "$COPY"/ae/controllers/stateful_controller.py \
              "$COPY"/ae/baselines/ae_full.py \
              "$COPY"/ae/runners/run_alfworld.py \
              "$COPY"/configs/demos/one_shot_v1.yaml \
              "$COPY"/configs/controllers/ae_maxint6.yaml \
              "$COPY"/configs/controllers/ae_maxint12.yaml
} > "$LOGDIR/sha256.log" 2>&1
grep -q "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch (must be the frozen v1 renderer, NOT v3)"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch"
log "check 2 OK: code byte-identical to the frozen run (v1 renderer confirmed)"

# Single-variable guard: each maxint config must differ from the frozen
# baseline in exactly one non-comment line, and that line must be
# max_interventions.
for N in 6 12; do
    D=$(diff <(grep -v '^#' "$COPY/configs/controllers/ae_full_final_nopir_v2_5.yaml") \
             <(grep -v '^#' "$COPY/configs/controllers/ae_maxint${N}.yaml") | grep '^[<>]' || true)
    NLINES=$(echo "$D" | grep -c '^[<>]' || true)
    [ "$NLINES" -eq 2 ] || fail "ae_maxint${N}.yaml differs in more than max_interventions: $D"
    echo "$D" | grep -q "max_interventions: ${N}" || fail "ae_maxint${N}.yaml does not set max_interventions=${N}: $D"
    echo "maxint${N} single-variable diff OK: $D" >> "$LOGDIR/batch.log"
done
log "check 3 OK: both configs are single-variable (max_interventions only)"

cd "$COPY" || fail "cannot cd into $COPY"
for f in alfworld_runs_ae/tests/test_demo_config.py \
         alfworld_runs_ae/tests/test_alfworld_action_normalize.py \
         alfworld_runs_ae/tests/test_output_parser.py \
         alfworld_runs_ae/tests/test_prompt_budget.py \
         alfworld_runs_ae/tests/test_put_move_compat_integration.py \
         alfworld_runs_ae/tests/test_environment_anchor.py; do
    out=$("$PY" "$f" 2>&1)
    { echo "=== $f ==="; echo "$out"; } >> "$LOGDIR/regression.log"
    echo "$out" | tail -1 | grep -qE '^[0-9]+ passed, 0 failed$' \
        || fail "regression: $f did not fully pass"
done
log "check 4 OK: regression suite passed"

log "ALL PRE-FLIGHT CHECKS PASSED. Starting vLLM server."

source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /vol/gpudata/jy625-ae-data/models/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --port 8000 --host 0.0.0.0 \
    > "$LOGDIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
log "vLLM server starting, pid=$VLLM_PID"

BASE_URL="http://localhost:8000/v1"
READY=0
for i in $(seq 1 60); do
    curl -s -o /dev/null -w "" "$BASE_URL/models" 2>/dev/null && { READY=1; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM died during startup"
    sleep 10
done
[ "$READY" -eq 1 ] || fail "vLLM not ready within 10 minutes"
log "vLLM server ready"

run_split() {
    local cfg=$1 split=$2 run_name=$3
    "$PY" -m ae.runners.run_alfworld \
        --baseline ae_full --split "$split" \
        --run-name "$run_name" \
        --ae-config "configs/controllers/${cfg}.yaml" \
        --demo-config configs/demos/one_shot_v1.yaml \
        --termination-policy legacy_early_stop \
        --seed 42 \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$COPY/ae/runners/runs" \
        > "$LOGDIR/${run_name}.log" 2>&1
}

for N in 6 12; do
    log "=== max_interventions=$N : practice + exam in parallel ==="
    run_split "ae_maxint${N}" practice "ae_maxint${N}_oneshot0_seed42_practice" & p1=$!
    run_split "ae_maxint${N}" exam     "ae_maxint${N}_oneshot0_seed42_exam"     & p2=$!
    wait $p1; r1=$?
    wait $p2; r2=$?
    log "maxint$N done: practice rc=$r1 exam rc=$r2"
    [ "$r1" -eq 0 ] || fail "maxint$N practice failed (rc=$r1)"
    [ "$r2" -eq 0 ] || fail "maxint$N exam failed (rc=$r2)"
done

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
VLLM_PID=""
log "JOB COMPLETE."
