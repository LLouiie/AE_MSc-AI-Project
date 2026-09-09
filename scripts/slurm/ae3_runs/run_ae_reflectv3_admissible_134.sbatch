#!/bin/bash
#SBATCH --job-name=ae_reflectv3
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_reflectv3-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_reflectv3-%j.err

# ============================================================================
# REFLECT prompt-v3: admissible-action grounding, full 134 (practice + exam).
#
# WHAT CHANGED (isolated workdir admissible_reflect_workdir/, exactly two
# files differ from the official one-shot copy, diff-verified):
#   ae/controllers/stateful_controller.py  sha256=
#     3127c243b5abcbd9a5115c3d01bd19098dc62527ece63ad099d207a25207164c
#     -- records `admissible_after` into self.current_admissible_commands
#        and passes it to render_directive(). Pure bookkeeping: nothing
#        reads it for routing/signals/decisions (unit-tested: routing
#        decisions are identical with vs without it supplied).
#   ae/controllers/intervention_renderer.py  sha256=
#     d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f
#     -- REFLECT's directive appends the environment's own admissible
#        command list. REFLECT's v1 body text is preserved verbatim as the
#        prefix; VERIFY and REPLAN render byte-identically to before.
#
# WHY (this session's practice-log case study, see the renderer's
# _render_reflect docstring for the full write-up): all 42/42 practice
# failures contain at least one inadmissible action, 591 in total, while the
# median SUCCESSFUL episode is only 10.5 steps and no success ever needed
# more than 38 of the 50-step budget -- failing episodes overwhelmingly burn
# their budget on actions the environment silently ignores ("Nothing
# happens."). Worked example, look_at_obj_in_light-Bowl-None-DeskLamp-308:
# at step 17 the model held the bowl AND stood at the desk with the desklamp
# -- one `use desklamp 1` from success, and `use desklamp 1` was in the
# admissible list at that step -- but it had just tried
# `use desklamp 1 on bowl 2` and `turn on desklamp 1` (both inadmissible),
# walked away, then burned the remaining 33 steps alternating between two
# desks. Note the one-shot examine demo DOES teach the correct
# `use desklamp 1` form, so this is not the put->move class of demo/env
# mismatch -- the model simply loses track of what the environment accepts.
#
# This deliberately prescribes NO recovery action, unlike the rejected
# REFLECT-v2 (audit_reports/ae_reflect_prompt_v2_rejected.md), whose single
# static hint helped nothing and measurably hurt heat because the correct
# recovery is task-type-dependent and sometimes semantically opposite.
# Listing what the environment accepts makes no claim that can be wrong for
# any task type, and adds no task-type branching. It is grounding, not
# auto-correction: the list goes into the ephemeral prompt only (never into
# history/trajectory, same as every other directive) and the model still
# picks and emits its own action -- nothing rewrites what gets executed.
#
# SCOPE: practice (100) AND exam (34) in this one job. The user explicitly
# directed, twice, that exam be tuned on as well ("带着exam一起调"), so the
# resulting 134 number must be reported as a figure fitted on all 134 tasks,
# NOT as a generalization result -- this overrides the project's usual
# practice-only tuning constraint by explicit standing instruction, and the
# final report must say so plainly.
# ============================================================================

set -uo pipefail

WD=/vol/gpudata/jy625-ae-data/scratch_ae3/admissible_reflect_workdir
BASE=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_reflectv3_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

RUN_P=ae_reflectv3_admissible_oneshot0_seed42_practice
RUN_E=ae_reflectv3_admissible_oneshot0_seed42_exam

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() {
    echo "[$(date -Is)] PREFLIGHT FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/preflight_failure.log"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}
# Notification fires from the compute node itself, inside this script's own
# EXIT trap -- never from an external polling loop (those die whenever the
# driving session restarts; happened on jobs 271479, 273214/273236, and
# 273526/273531/273532).
notify() {
    curl -s -m 20 -d "$1" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
}
trap 'rc=$?; log "job ending, exit code $rc"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; notify "REFLECT-v3 admissible 134 (job ${SLURM_JOB_ID}) ended, exit=$rc, logdir=$LOGDIR"' EXIT

log "job started on $(hostname), SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

# ---------------------------------------------------------------------------
# Pre-flight 1: result dirs free.
# ---------------------------------------------------------------------------
for d in "$RUN_P" "$RUN_E"; do
    [ -e "$WD/ae/runners/runs/$d" ] && fail "result directory already exists, refusing to overwrite: $WD/ae/runners/runs/$d"
done
log "check 1 OK: 2 result dirs free"

# ---------------------------------------------------------------------------
# Pre-flight 2: sha256 + confirm ONLY the two intended controller files (and
# the new test) differ from the official one-shot copy.
# ---------------------------------------------------------------------------
{
    echo "=== admissible_reflect_workdir ==="
    sha256sum "$WD"/alfworld_runs_ae/agents.py \
              "$WD"/alfworld_runs_ae/demo_config.py \
              "$WD"/ae/controllers/intervention_renderer.py \
              "$WD"/ae/controllers/stateful_controller.py \
              "$WD"/ae/baselines/ae_full.py \
              "$WD"/ae/runners/run_alfworld.py \
              "$WD"/configs/controllers/ae_full_final_nopir_v2_5.yaml \
              "$WD"/configs/demos/one_shot_v1.yaml
    echo ""
    echo "=== diff vs one-shot baseline ==="
    diff -rq --exclude=runs --exclude='__pycache__' --exclude='*.pyc' "$BASE" "$WD" || true
} > "$LOGDIR/sha256.log" 2>&1

grep -q "3127c243b5abcbd9a5115c3d01bd19098dc62527ece63ad099d207a25207164c  $WD/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch vs expected v3 -- see $LOGDIR/sha256.log"
grep -q "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f  $WD/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch vs expected Appendix A renderer -- see $LOGDIR/sha256.log"
UNEXPECTED=$(grep "^Files .* differ$\|^Only in" "$LOGDIR/sha256.log" \
    | grep -v "intervention_renderer.py" \
    | grep -v "stateful_controller.py" \
    | grep -v "test_reflect_admissible_v3.py" || true)
if [ -n "$UNEXPECTED" ]; then
    fail "unexpected diff outside the two controller files + new test: $UNEXPECTED"
fi
log "check 2 OK: only intervention_renderer.py + stateful_controller.py (+ new test) differ (see $LOGDIR/sha256.log)"

# ---------------------------------------------------------------------------
# Pre-flight 3: regression suite, including the REPLAN prompt suites (which
# must stay green -- REPLAN's rendering is supposed to be untouched) and the
# new v3 contract tests.
# ---------------------------------------------------------------------------
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
    line=$(echo "$out" | tail -1)
    echo "$line" | grep -qE '^[0-9]+ passed, 0 failed$' \
        || fail "regression: $f did not fully pass ($line) -- see $LOGDIR/regression.log"
done
log "check 3 OK: regression suite passed incl. REPLAN suites + v3 contract tests"

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

run_split() {
    local split=$1 run_name=$2
    "$PY" -m ae.runners.run_alfworld \
        --baseline ae_full --split "$split" \
        --run-name "$run_name" \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
        --demo-config configs/demos/one_shot_v1.yaml \
        --termination-policy legacy_early_stop \
        --seed 42 \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$WD/ae/runners/runs" \
        > "$LOGDIR/${run_name}.log" 2>&1
}

log "=== practice (100) + exam (34) in parallel, REFLECT-v3 ==="
run_split practice "$RUN_P" & p1=$!
run_split exam "$RUN_E" & p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
log "practice rc=$r1  exam rc=$r2"
[ "$r1" -eq 0 ] || fail "practice run failed (rc=$r1) -- see $LOGDIR/${RUN_P}.log"
[ "$r2" -eq 0 ] || fail "exam run failed (rc=$r2) -- see $LOGDIR/${RUN_E}.log"

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
VLLM_PID=""
log "vLLM server stopped."
log "JOB COMPLETE."
