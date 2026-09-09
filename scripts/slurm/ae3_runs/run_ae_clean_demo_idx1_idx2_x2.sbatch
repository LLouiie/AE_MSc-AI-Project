#!/bin/bash
#SBATCH --job-name=ae_cleandemo
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_cleandemo-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_cleandemo-%j.err

# ============================================================================
# clean-only demo-index comparison: idx1 x2 and idx2 x2, full 134 (clean is
# 31 of the 134 tasks -- the largest single type).
#
# WHY: clean has been frozen at idx0 for this entire project on the grounds
# that put and clean are the only two types whose index-0 demo is CONFIRMED
# verbatim against ReflAct (arXiv:2505.15182v2) Appendix K. That confirmation
# is about protocol fidelity to ReflAct, not about which demo scores best --
# and the user has explicitly authorised changing demo selection to maximise
# the 134 number. clean has never been run at idx1 or idx2 for any task.
#
# The headroom is real: clean sits at 23.33/31 (75%) across four draws
# (23,23,23,23 -- notably the most stable type on record), i.e. 8 tasks
# failing, on the largest type. put is skipped: it is at 23/24 (96%), one
# task from ceiling, so there is nothing to win there.
#
# Both clean demos being tested are well-formed -- the structural scan of all
# 18 demos found only react_cool_0 defective (see
# audit_reports/ae_oneshot_cool_demo_defect.md), so this is a straight
# content comparison, not another defect hunt.
#
# IF idx1 or idx2 wins, note the cost: adopting it means the reported
# configuration no longer uses the ReflAct-confirmed demo for clean, which
# must be stated plainly wherever the number is reported, since protocol
# fidelity to ReflAct was the original reason index 0 was chosen.
#
# NO CODE CHANGE. Pre-existing --demo-config and --task-types flags only.
# put is never run here (--task-types clean).
#
# Splits: practice AND exam. Fitted-on-134, not generalization.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_cleandemo_${SLURM_JOB_ID}
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
trap 'rc=$?; log "job ending, exit code $rc"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; notify "heat demo idx1/idx2 x2 (job ${SLURM_JOB_ID}) ended, exit=$rc, logdir=$LOGDIR"' EXIT

log "job started on $(hostname), SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

for IDX in 1 2; do
  for REP in 1 2; do
    for s in practice exam; do
      d="ae_cleandemo_idx${IDX}_rep${REP}_${s}"
      [ -e "$COPY/ae/runners/runs/$d" ] && fail "result dir already exists: $d"
    done
  done
done
log "check 1 OK: 8 result dirs free"

{
    sha256sum "$COPY"/alfworld_runs_ae/agents.py \
              "$COPY"/alfworld_runs_ae/demo_config.py \
              "$COPY"/ae/controllers/intervention_renderer.py \
              "$COPY"/ae/controllers/stateful_controller.py \
              "$COPY"/ae/runners/run_alfworld.py \
              "$COPY"/configs/controllers/ae_full_final_nopir_v2_5.yaml \
              "$COPY"/configs/demos/one_shot_v1_index1.yaml \
              "$COPY"/configs/demos/one_shot_v1_index2.yaml
} > "$LOGDIR/sha256.log" 2>&1
grep -q "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch (must be the Appendix A renderer)"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch"
log "check 2 OK: code byte-identical to the frozen 80/134 run"

for spec in one_shot_v1_index1:1 one_shot_v1_index2:2; do
    f=${spec%%:*}; want=${spec##*:}
    got=$("$PY" -c "
import yaml; d=yaml.safe_load(open('$COPY/configs/demos/${f}.yaml'))
print(f\"{d['num_demos']}:{d['demo_indices']}\")")
    [ "$got" = "1:[$want]" ] || fail "${f}.yaml is not one-shot index $want (got $got)"
done
log "check 3 OK: demo configs are one-shot with the expected indices"

cd "$COPY" || fail "cannot cd into $COPY"
for f in alfworld_runs_ae/tests/test_demo_config.py \
         alfworld_runs_ae/tests/test_alfworld_action_normalize.py \
         alfworld_runs_ae/tests/test_output_parser.py \
         alfworld_runs_ae/tests/test_prompt_budget.py \
         alfworld_runs_ae/tests/test_put_move_compat_integration.py \
         alfworld_runs_ae/tests/test_environment_anchor.py; do
    out=$("$PY" "$f" 2>&1)
    { echo "=== $f ==="; echo "$out"; } >> "$LOGDIR/regression.log"
    echo "$out" | tail -1 | grep -qE '^[0-9]+ passed, 0 failed$' || fail "regression: $f failed"
done
log "check 4 OK: regression suite passed"

log "ALL PRE-FLIGHT CHECKS PASSED. Starting vLLM server."
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

# Per-job port (see the round-2 port-collision incident: two co-scheduled jobs
# both hard-coded :8000 and the second silently attached to the first's vLLM).
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
kill -0 "$VLLM_PID" 2>/dev/null || fail "port ${PORT} answered but OUR vLLM process is dead"
log "vLLM server ready on port ${PORT} (pid=$VLLM_PID)"

run_one() {
    local split=$1 demo=$2 run_name=$3
    "$PY" -m ae.runners.run_alfworld \
        --baseline ae_full --split "$split" \
        --run-name "$run_name" \
        --task-types clean \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
        --demo-config "configs/demos/${demo}.yaml" \
        --termination-policy legacy_early_stop \
        --seed 42 \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$COPY/ae/runners/runs" \
        > "$LOGDIR/${run_name}.log" 2>&1
}

for IDX in 1 2; do
  for REP in 1 2; do
    log "=== heat idx${IDX} repeat ${REP}: practice+exam parallel ==="
    run_one practice "one_shot_v1_index${IDX}" "ae_cleandemo_idx${IDX}_rep${REP}_practice" & p1=$!
    run_one exam     "one_shot_v1_index${IDX}" "ae_cleandemo_idx${IDX}_rep${REP}_exam"     & p2=$!
    wait $p1; r1=$?
    wait $p2; r2=$?
    log "heat idx${IDX} rep${REP} done: practice rc=$r1 exam rc=$r2"
    [ "$r1" -eq 0 ] || fail "idx${IDX} rep${REP} practice failed (rc=$r1)"
    [ "$r2" -eq 0 ] || fail "idx${IDX} rep${REP} exam failed (rc=$r2)"
  done
done

# Post-run guard: only heat tasks may appear, and each run must cover all 23.
CHECK=$("$PY" -c "
import json
bad=[]
for idx in (1,2):
  for rep in (1,2):
    n=0
    for s in ('practice','exam'):
        p=f'$COPY/ae/runners/runs/ae_cleandemo_idx{idx}_rep{rep}_{s}/episode_log.jsonl'
        for l in open(p):
            r=json.loads(l); n+=1
            if not r['env_name'].startswith('pick_clean_then_place_in_recep'):
                bad.append(f'idx{idx} rep{rep}: non-heat task {r[\"env_name\"]}')
    if n!=31: bad.append(f'idx{idx} rep{rep}: {n} tasks, expected 31')
print('BAD:'+'; '.join(bad[:5]) if bad else 'OK all four runs are 31 clean tasks')
" 2>&1)
echo "$CHECK" | grep -q "^OK" || fail "post-run check failed: $CHECK"
log "post-run check OK: $CHECK"

kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; VLLM_PID=""
log "JOB COMPLETE."
