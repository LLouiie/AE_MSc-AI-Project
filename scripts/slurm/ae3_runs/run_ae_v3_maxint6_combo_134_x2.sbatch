#!/bin/bash
#SBATCH --job-name=ae_combo
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_combo-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_combo-%j.err

# ============================================================================
# REFLECT-v3 (admissible-action grounding) + max_interventions=6, full 134,
# TWO independent repeats.
#
# Round-1 single-draw results on 134 (baseline = frozen 80/134):
#   fixed_horizon          76/134  (-4)  -> DROPPED, see note below
#   REFLECT-v3 alone       83/134  (+3), practice 62/100 (+4)
#   max_interventions=6    82/134  (+2)
#   max_interventions=12   80/134  (+0)
# Every one of those deltas is inside the noise band (dev60 repeats of an
# identical config gave stdev 3.30 on 60 tasks; per-type swings of 6 on 23
# tasks have been observed), so NONE of them is established yet. This job
# does two things at once: it repeats the two positive-direction changes as
# a combination, and -- because both components are independently logged --
# lets their joint effect be compared against each one's solo draw.
#
# Why these two combine cleanly: they act on different bottlenecks and were
# each measured against the same frozen baseline with the other held fixed.
# REFLECT-v3 improves WHAT a directive says; max_interventions changes HOW
# OFTEN one is available. The practice-log analysis that motivated both found
# 74.5% of the 591 inadmissible actions in failing episodes occur with NO
# directive active, and 38/42 failures exhaust the 3-intervention budget then
# run a median 19 further unsupervised steps -- i.e. REFLECT-v3's better
# wording can only reach ~14% of the problem while the budget stays at 3.
# The combination is the first configuration where the better directive is
# also available for most of the flailing.
#
# Note on fixed_horizon (dropped): its pre-registered rationale was that
# removing the exhausted_repeated early exit can only ever give a failing
# episode more steps, never turn a success into a failure -- that argument
# holds within a single fixed trajectory but NOT across runs, because
# trajectories diverge under vLLM non-determinism. It scored -4, which is
# inside noise, so this is "no evidence it helps", not "proven harmful".
# It is simply not carried forward.
#
# Splits: practice AND exam, per the user's explicit instruction to tune on
# all 134. Numbers are fitted-on-134, NOT generalization.
#
# Code: isolated workdir admissible_reflect_workdir/ (REFLECT-v3), exactly two
# files differ from the official copy plus the new v3 contract test:
#   stateful_controller.py   sha256=3127c243b5abcbd9a5115c3d01bd19098dc62527ece63ad099d207a25207164c
#   intervention_renderer.py sha256=d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f
# Controller config: ae_maxint6.yaml (single-variable vs the frozen baseline:
# max_interventions 3 -> 6, re-verified below).
# ============================================================================

set -uo pipefail

WD=/vol/gpudata/jy625-ae-data/scratch_ae3/admissible_reflect_workdir
BASE=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_combo_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() {
    echo "[$(date -Is)] PREFLIGHT FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/preflight_failure.log"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}
notify() { curl -s -m 20 -d "$1" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true; }
trap 'rc=$?; log "job ending, exit code $rc"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; notify "v3+maxint6 combo 134 x2 (job ${SLURM_JOB_ID}) ended, exit=$rc, logdir=$LOGDIR"' EXIT

log "job started on $(hostname), SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

for REP in 1 2; do
  for s in practice exam; do
    d="ae_combo_v3_maxint6_rep${REP}_${s}"
    [ -e "$WD/ae/runners/runs/$d" ] && fail "result dir already exists: $d"
  done
done
log "check 1 OK: 4 result dirs free"

{
    sha256sum "$WD"/alfworld_runs_ae/agents.py \
              "$WD"/alfworld_runs_ae/demo_config.py \
              "$WD"/ae/controllers/intervention_renderer.py \
              "$WD"/ae/controllers/stateful_controller.py \
              "$WD"/ae/runners/run_alfworld.py \
              "$WD"/configs/controllers/ae_maxint6.yaml \
              "$WD"/configs/demos/one_shot_v1.yaml
    echo ""
    echo "=== diff vs official copy ==="
    diff -rq --exclude=runs --exclude='__pycache__' --exclude='*.pyc' "$BASE" "$WD" || true
} > "$LOGDIR/sha256.log" 2>&1

grep -q "3127c243b5abcbd9a5115c3d01bd19098dc62527ece63ad099d207a25207164c  $WD/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch vs expected v3"
grep -q "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f  $WD/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch vs expected Appendix A renderer"
UNEXPECTED=$(grep "^Files .* differ$\|^Only in" "$LOGDIR/sha256.log" \
    | grep -v "intervention_renderer.py" | grep -v "stateful_controller.py" \
    | grep -v "test_reflect_admissible_v3.py" || true)
[ -n "$UNEXPECTED" ] && fail "unexpected diff outside the two v3 files: $UNEXPECTED"
log "check 2 OK: only the two v3 controller files (+ its test) differ from the official copy"

D=$(diff <(grep -v '^#' "$BASE/configs/controllers/ae_full_final_nopir_v2_5.yaml") \
         <(grep -v '^#' "$WD/configs/controllers/ae_maxint6.yaml") | grep '^[<>]' || true)
[ "$(echo "$D" | grep -c '^[<>]')" -eq 2 ] || fail "ae_maxint6.yaml is not single-variable: $D"
echo "$D" | grep -q "max_interventions: 6" || fail "ae_maxint6.yaml does not set max_interventions=6: $D"
log "check 3 OK: controller config is single-variable (max_interventions 3 -> 6)"

cd "$WD" || fail "cannot cd into $WD"
for f in alfworld_runs_ae/tests/test_demo_config.py \
         alfworld_runs_ae/tests/test_alfworld_action_normalize.py \
         alfworld_runs_ae/tests/test_output_parser.py \
         alfworld_runs_ae/tests/test_prompt_budget.py \
         alfworld_runs_ae/tests/test_put_move_compat_integration.py \
         alfworld_runs_ae/tests/test_environment_anchor.py \
         ae/controllers/tests/test_replan_prompt_v2.py \
         ae/controllers/tests/test_replan_progress_wording.py \
         ae/controllers/tests/test_reflect_admissible_v3.py; do
    out=$("$PY" "$f" 2>&1)
    { echo "=== $f ==="; echo "$out"; } >> "$LOGDIR/regression.log"
    echo "$out" | tail -1 | grep -qE '^[0-9]+ passed, 0 failed$' || fail "regression: $f failed"
done
log "check 4 OK: full regression incl. REPLAN suites + v3 contract tests"

log "ALL PRE-FLIGHT CHECKS PASSED. Starting vLLM server."
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
# Per-job port. Two of this project's jobs were co-scheduled onto the same
# node and both hard-coded :8000 -- the second silently attached to the
# FIRST job's vLLM, then died with APIConnectionError when that job exited
# and killed its server. Derive a unique port from SLURM_JOB_ID and refuse
# to start if anything is already listening on it.
PORT=$(( 8000 + (SLURM_JOB_ID % 900) ))
if curl -s -o /dev/null -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null; then
    fail "port ${PORT} is already serving before we started -- refusing to attach to another job's vLLM"
fi
log "using vLLM port ${PORT}"

vllm serve /vol/gpudata/jy625-ae-data/models/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B --port "${PORT}" --host 0.0.0.0 \
    > "$LOGDIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
BASE_URL="http://localhost:${PORT}/v1"
READY=0
for i in $(seq 1 60); do
    curl -s -o /dev/null -w "" "$BASE_URL/models" 2>/dev/null && { READY=1; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM died during startup -- see $LOGDIR/vllm_server.log"
    sleep 10
done
[ "$READY" -eq 1 ] || fail "vLLM not ready within 10 minutes"
kill -0 "$VLLM_PID" 2>/dev/null || fail "port ${PORT} answered but OUR vLLM process is dead -- would have been another job's server"
log "vLLM server ready on port ${PORT} (pid=$VLLM_PID)"

run_split() {
    local split=$1 run_name=$2
    "$PY" -m ae.runners.run_alfworld \
        --baseline ae_full --split "$split" \
        --run-name "$run_name" \
        --ae-config configs/controllers/ae_maxint6.yaml \
        --demo-config configs/demos/one_shot_v1.yaml \
        --termination-policy legacy_early_stop \
        --seed 42 \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$WD/ae/runners/runs" \
        > "$LOGDIR/${run_name}.log" 2>&1
}

for REP in 1 2; do
    log "=== repeat $REP: practice + exam in parallel (REFLECT-v3 + maxint6) ==="
    run_split practice "ae_combo_v3_maxint6_rep${REP}_practice" & p1=$!
    run_split exam     "ae_combo_v3_maxint6_rep${REP}_exam"     & p2=$!
    wait $p1; r1=$?
    wait $p2; r2=$?
    log "repeat $REP done: practice rc=$r1 exam rc=$r2"
    [ "$r1" -eq 0 ] || fail "rep$REP practice failed (rc=$r1)"
    [ "$r2" -eq 0 ] || fail "rep$REP exam failed (rc=$r2)"
done

kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; VLLM_PID=""
log "JOB COMPLETE."
