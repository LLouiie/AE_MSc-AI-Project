#!/bin/bash
#SBATCH --job-name=ae_demoidx2
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_demoidx2-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_demoidx2-%j.err

# ============================================================================
# Demo index 2 for heat/cool/puttwo/examine, full 134 (practice + exam).
#
# NO CODE CHANGE. Runs against the official one-shot copy using only
# pre-existing CLI flags (--demo-config, --task-types) plus one new config
# file (configs/demos/one_shot_v1_index2.yaml, demo_indices:[2]).
#
# Why: those four types' index-0 choice was a PRE-REGISTERED EXTRAPOLATION,
# never confirmed against ReflAct's paper (which prints one-shot examples
# only for put and clean -- those two are confirmed and are NOT touched
# here; --task-types keeps them out of every run below). Index 1 was tested
# in the demo-selection ablation (#74-#78). Index 2 (react_{type}_2 in
# alfworld_3prompts.json) has never been run for any type in this project,
# so this is genuinely new information rather than a re-draw.
#
# Existing per-type reference numbers on the full 134 (from
# audit_reports/ae_oneshot_demo_selection_ablation_per_task.json):
#     type      n    idx0    idx1
#     heat     23      14   (not run: mixed arm used idx0)
#     cool     21       8      11
#     examine  18       3       4
#     puttwo   17       9      11
# NOTE the noise floor is large and must bound any reading of this run:
# the ablation's heat arm scored 14 then 8 on two runs of the IDENTICAL
# index-0 config -- a 6-point swing on 23 tasks from run-to-run
# non-determinism alone. A single index-2 draw per type therefore cannot
# establish a real effect by itself; it can only flag candidates worth a
# repeat run.
#
# Splits: practice AND exam, per the user's explicit instruction to tune
# against all 134 including exam. Anything selected using this data must be
# reported as fitted-on-134, not as a generalization result.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_demoidx2_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

TYPES=heat,cool,puttwo,examine
RUN_P=ae_demoidx2_hcpe_oneshot_seed42_practice
RUN_E=ae_demoidx2_hcpe_oneshot_seed42_exam

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() {
    echo "[$(date -Is)] PREFLIGHT FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/preflight_failure.log"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}
notify() {
    curl -s -m 20 -d "$1" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
}
trap 'rc=$?; log "job ending, exit code $rc"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; notify "demo index2 hcpe 134 (job ${SLURM_JOB_ID}) ended, exit=$rc, logdir=$LOGDIR"' EXIT

log "job started on $(hostname), SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

for d in "$RUN_P" "$RUN_E"; do
    [ -e "$COPY/ae/runners/runs/$d" ] && fail "result directory already exists: $COPY/ae/runners/runs/$d"
done
log "check 1 OK: 2 result dirs free"

# sha256: controller/renderer/demo_config/agents must be byte-identical to
# the frozen 80/134 official run -- ONLY the demo index may differ.
{
    sha256sum "$COPY"/alfworld_runs_ae/agents.py \
              "$COPY"/alfworld_runs_ae/demo_config.py \
              "$COPY"/ae/controllers/intervention_renderer.py \
              "$COPY"/ae/controllers/stateful_controller.py \
              "$COPY"/ae/baselines/ae_full.py \
              "$COPY"/ae/runners/run_alfworld.py \
              "$COPY"/configs/controllers/ae_full_final_nopir_v2_5.yaml \
              "$COPY"/configs/demos/one_shot_v1_index2.yaml
} > "$LOGDIR/sha256.log" 2>&1
grep -q "c4824c0606a87eedc32aa96564c9ddcedafc1fd11a7b212a3131382fe43367ba  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch (must be the frozen v1 renderer) -- see $LOGDIR/sha256.log"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch -- see $LOGDIR/sha256.log"
log "check 2 OK: sha256 recorded, controller+renderer byte-identical to the frozen run"

# Confirm the index-2 config really says index 2, and that put/clean can
# never be reached by these invocations.
IDX=$("$PY" -c "
import yaml,sys
d=yaml.safe_load(open('$COPY/configs/demos/one_shot_v1_index2.yaml'))
print(d['num_demos'], d['demo_indices'])
" 2>&1)
[ "$IDX" = "1 [2]" ] || fail "index2 config is not num_demos=1 demo_indices=[2] (got: $IDX)"
log "check 3 OK: demo config is one-shot, index 2 ($IDX)"

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
        || fail "regression: $f did not fully pass ($line)"
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
    kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM died during startup -- see $LOGDIR/vllm_server.log"
    sleep 10
done
[ "$READY" -eq 1 ] || fail "vLLM not ready within 10 minutes"
log "vLLM server ready"

run_split() {
    local split=$1 run_name=$2
    "$PY" -m ae.runners.run_alfworld \
        --baseline ae_full --split "$split" \
        --run-name "$run_name" \
        --task-types "$TYPES" \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
        --demo-config configs/demos/one_shot_v1_index2.yaml \
        --termination-policy legacy_early_stop \
        --seed 42 \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$COPY/ae/runners/runs" \
        > "$LOGDIR/${run_name}.log" 2>&1
}

log "=== practice + exam (heat,cool,puttwo,examine only), demo index 2 ==="
run_split practice "$RUN_P" & p1=$!
run_split exam "$RUN_E" & p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
log "practice rc=$r1  exam rc=$r2"
[ "$r1" -eq 0 ] || fail "practice run failed (rc=$r1)"
[ "$r2" -eq 0 ] || fail "exam run failed (rc=$r2)"

# Post-run guard: put/clean must be entirely absent from both logs.
LEAK=$("$PY" -c "
import json
bad=[]
for p in ['$COPY/ae/runners/runs/$RUN_P/episode_log.jsonl','$COPY/ae/runners/runs/$RUN_E/episode_log.jsonl']:
    for l in open(p):
        r=json.loads(l)
        n=r['env_name']
        if n.startswith('pick_and_place_simple') or n.startswith('pick_clean_then_place'):
            bad.append(n)
print('LEAK' if bad else 'CLEAN', len(bad))
" 2>&1)
echo "$LEAK" | grep -q "^CLEAN" || fail "put/clean tasks leaked into an index-2 run: $LEAK"
log "post-run check OK: no put/clean task ran under index 2 ($LEAK)"

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
VLLM_PID=""
log "JOB COMPLETE."
