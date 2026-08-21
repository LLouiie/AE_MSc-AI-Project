#!/bin/bash
#SBATCH --job-name=ae_mixed_demo
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_mixed_demo-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_mixed_demo-%j.err

# ============================================================================
# Demo-selection ablation, task #77: frozen mixed-demo full 134-task
# validation run.
#
# Frozen mapping (configs/demos/frozen_mixed_demo_ablation_seed42.yaml,
# sha256=5ab63ed3d55dd4a58f71a47b270bbe2fe285ce91ab62a318f8e8f0e14dce2cec),
# decided from PRACTICE-ONLY data (task #75-76), exam never inspected
# before freezing:
#   put=0 clean=0 heat=0 (Group A, index 0, one_shot_v1.yaml)
#   cool=1 puttwo=1 examine=1 (Group B, index 1, one_shot_v1_index1.yaml)
#
# Realized as 4 runs (Group A x {practice,exam}, Group B x {practice,exam})
# via the --task-types filter added for task #75 -- pure task-selection,
# no demo_config.py/agents.py/controller changes. Same controller
# (ae_full_final_nopir_v2_5.yaml, old REPLAN-unconditional), same REPLAN
# prompt-v2, same seed=42, same termination_policy=legacy_early_stop, same
# task-list order as the uniform-_0 80/134 baseline
# (ae_full_nopir_v2_5_oldctrl_replanpromptv2_oneshot0_seed42_a40_*) --
# ONLY the per-type demo index differs, per the frozen mapping.
#
# Does NOT touch/overwrite the uniform-_0 baseline result dirs or the
# task #75 practice-only ablation result dir.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
SEED=42
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_mixed_demo_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

RUN_A_PRACTICE=ae_mixed_demo_groupA_index0_seed42_a40_practice
RUN_A_EXAM=ae_mixed_demo_groupA_index0_seed42_a40_exam
RUN_B_PRACTICE=ae_mixed_demo_groupB_index1_seed42_a40_practice
RUN_B_EXAM=ae_mixed_demo_groupB_index1_seed42_a40_exam

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
# Pre-flight 1: none of the 4 result dirs already exist; also confirm the
# uniform-_0 baseline dirs and task #75 ablation dir are untouched targets
# (existence-only check, never written to by this job).
# ---------------------------------------------------------------------------
for d in "$RUN_A_PRACTICE" "$RUN_A_EXAM" "$RUN_B_PRACTICE" "$RUN_B_EXAM"; do
    [ -e "$COPY/ae/runners/runs/$d" ] && fail "result directory already exists, refusing to overwrite: $COPY/ae/runners/runs/$d"
done
for d in ae_full_nopir_v2_5_oldctrl_replanpromptv2_oneshot0_seed42_a40_practice \
         ae_full_nopir_v2_5_oldctrl_replanpromptv2_oneshot0_seed42_a40_exam \
         ae_practice_ablation_hcpe_index1_seed42_a40; do
    [ -d "$COPY/ae/runners/runs/$d" ] || fail "expected pre-existing dir missing: $d"
done
log "check 1 OK: 4 new result dirs free; uniform-_0 baseline + task#75 ablation dirs present and untouched"

# ---------------------------------------------------------------------------
# Pre-flight 2: sha256 + confirm controller/renderer byte-identical to the
# uniform-_0 baseline and task #75 ablation runs (only variable is demo
# index selection via --task-types + --demo-config).
# ---------------------------------------------------------------------------
{
    echo "=== COPY ==="
    echo "path: $COPY"
    sha256sum "$COPY"/alfworld_runs_ae/agents.py \
              "$COPY"/alfworld_runs_ae/demo_config.py \
              "$COPY"/ae/controllers/intervention_renderer.py \
              "$COPY"/ae/controllers/stateful_controller.py \
              "$COPY"/ae/baselines/ae_full.py \
              "$COPY"/ae/runners/run_alfworld.py \
              "$COPY"/configs/controllers/ae_full_final_nopir_v2_5.yaml \
              "$COPY"/configs/demos/one_shot_v1.yaml \
              "$COPY"/configs/demos/one_shot_v1_index1.yaml \
              "$COPY"/configs/demos/frozen_mixed_demo_ablation_seed42.yaml
} > "$LOGDIR/sha256.log" 2>&1

grep -q "c4824c0606a87eedc32aa96564c9ddcedafc1fd11a7b212a3131382fe43367ba  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch -- see $LOGDIR/sha256.log"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch -- see $LOGDIR/sha256.log"
grep -q "5ab63ed3d55dd4a58f71a47b270bbe2fe285ce91ab62a318f8e8f0e14dce2cec  $COPY/configs/demos/frozen_mixed_demo_ablation_seed42.yaml" "$LOGDIR/sha256.log" \
    || fail "frozen_mixed_demo_ablation_seed42.yaml sha mismatch (mapping file changed after freeze!) -- see $LOGDIR/sha256.log"
log "check 2 OK: sha256 recorded, controller/renderer/frozen-mapping confirmed byte-identical to what was frozen (see $LOGDIR/sha256.log)"

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
# Pre-flight 4: confirm task-type filter group counts match expectation and
# groupA+groupB partition every task with no overlap/gap: practice
# A=61(put18+clean24+heat19) B=39(cool16+puttwo9+examine14); exam
# A=17(put6+clean7+heat4) B=17(cool5+puttwo8+examine4); A+B=100/34=134 total.
# ---------------------------------------------------------------------------
FILTER_CHECK=$("$PY" -c "
import sys
sys.path.insert(0, 'alfworld_runs_ae')
from environment import get_practice_tasks, get_exam_tasks, get_task_type
groupA = {'put','clean','heat'}
groupB = {'cool','puttwo','examine'}
def count(tasks, allowed):
    n = 0
    for t in tasks:
        env_name = t.get('env_name') or t['gamefile'].split('/')[-3]
        if get_task_type(env_name) in allowed:
            n += 1
    return n
p = get_practice_tasks(); e = get_exam_tasks()
pa, pb = count(p, groupA), count(p, groupB)
ea, eb = count(e, groupA), count(e, groupB)
print(f'practiceA={pa} practiceB={pb} examA={ea} examB={eb}')
assert pa == 61 and pb == 39 and ea == 17 and eb == 17, 'COUNT MISMATCH'
assert pa + pb == 100 and ea + eb == 34, 'PARTITION MISMATCH'
print('OK')
" 2>&1)
echo "$FILTER_CHECK" >> "$LOGDIR/batch.log"
echo "$FILTER_CHECK" | grep -q "^OK$" || fail "group task-type filter count mismatch, expected practiceA=61 practiceB=39 examA=17 examB=17 -- got: $FILTER_CHECK"
log "check 4 OK: group partition matches expectation ($FILTER_CHECK)"

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
# Smoke test: 2 tasks (group A, practice), not counted, deleted after.
# ---------------------------------------------------------------------------
log "=== starting smoke test (2 tasks, group A, practice split) ==="
"$PY" -m ae.runners.run_alfworld \
    --baseline ae_full --split practice \
    --run-name "smoke_mixed_demo" \
    --task-types put,clean,heat \
    --limit 2 --seed "$SEED" \
    --demo-config configs/demos/one_shot_v1.yaml \
    --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/smoke.log" 2>&1
rc_smoke=$?
[ "$rc_smoke" -eq 0 ] || fail "smoke test exited non-zero ($rc_smoke) -- see $LOGDIR/smoke.log"
grep -q "PILOT DONE" "$LOGDIR/smoke.log" || fail "smoke test did not report PILOT DONE -- see $LOGDIR/smoke.log"
rm -rf "$COPY/ae/runners/runs/smoke_mixed_demo"
log "smoke test OK (2/2 tasks completed, rc=0), smoke result dir removed"

# ---------------------------------------------------------------------------
# Full runs: 4 groups, run practice pair and exam pair each in parallel
# against the shared server (2 groups per split, sequential across splits
# to bound peak concurrent load to the same level the 80/134 baseline used).
# ---------------------------------------------------------------------------
log "=== starting practice: group A (61 tasks, index0) + group B (39 tasks, index1) in parallel ==="
"$PY" -m ae.runners.run_alfworld \
    --baseline ae_full --split practice \
    --run-name "$RUN_A_PRACTICE" \
    --task-types put,clean,heat \
    --seed "$SEED" \
    --demo-config configs/demos/one_shot_v1.yaml \
    --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/practice_A.log" 2>&1 &
pid_pa=$!
"$PY" -m ae.runners.run_alfworld \
    --baseline ae_full --split practice \
    --run-name "$RUN_B_PRACTICE" \
    --task-types cool,puttwo,examine \
    --seed "$SEED" \
    --demo-config configs/demos/one_shot_v1_index1.yaml \
    --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/practice_B.log" 2>&1 &
pid_pb=$!
wait "$pid_pa"; rc_pa=$?
wait "$pid_pb"; rc_pb=$?
log "=== finished practice: rc_A=$rc_pa rc_B=$rc_pb ==="
[ "$rc_pa" -eq 0 ] || fail "group A practice run exited non-zero ($rc_pa) -- see $LOGDIR/practice_A.log"
[ "$rc_pb" -eq 0 ] || fail "group B practice run exited non-zero ($rc_pb) -- see $LOGDIR/practice_B.log"

log "=== starting exam: group A (17 tasks, index0) + group B (17 tasks, index1) in parallel ==="
"$PY" -m ae.runners.run_alfworld \
    --baseline ae_full --split exam \
    --run-name "$RUN_A_EXAM" \
    --task-types put,clean,heat \
    --seed "$SEED" \
    --demo-config configs/demos/one_shot_v1.yaml \
    --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/exam_A.log" 2>&1 &
pid_ea=$!
"$PY" -m ae.runners.run_alfworld \
    --baseline ae_full --split exam \
    --run-name "$RUN_B_EXAM" \
    --task-types cool,puttwo,examine \
    --seed "$SEED" \
    --demo-config configs/demos/one_shot_v1_index1.yaml \
    --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
    --termination-policy legacy_early_stop \
    --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
    --output-dir "$COPY/ae/runners/runs" \
    > "$LOGDIR/exam_B.log" 2>&1 &
pid_eb=$!
wait "$pid_ea"; rc_ea=$?
wait "$pid_eb"; rc_eb=$?
log "=== finished exam: rc_A=$rc_ea rc_B=$rc_eb ==="
[ "$rc_ea" -eq 0 ] || fail "group A exam run exited non-zero ($rc_ea) -- see $LOGDIR/exam_A.log"
[ "$rc_eb" -eq 0 ] || fail "group B exam run exited non-zero ($rc_eb) -- see $LOGDIR/exam_B.log"

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
VLLM_PID=""
log "vLLM server stopped."
log "JOB COMPLETE."
