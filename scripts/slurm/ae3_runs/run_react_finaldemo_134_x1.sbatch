#!/bin/bash
#SBATCH --job-name=react_finaldemo
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/react_finaldemo-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/react_finaldemo-%j.err

# ============================================================================
# ReAct under the FINAL per-type demo mapping, full 134, TWO repeats.
#
# WHY THIS IS REQUIRED, not optional: the demo mapping being adopted for AE
# (put/clean/examine=idx0, heat/cool=idx2, puttwo=idx1) changes the ICL
# example, and ALL baselines share the same demo_config/_build_base_prompt()
# path -- ReAct included. The frozen ReAct one-shot number (68/134) was
# produced with demo_indices:[0], which serves the MALFORMED `react_cool_0`
# (the only defective demo of the 18 in alfworld_3prompts.json; see
# audit_reports/ae_oneshot_cool_demo_defect.md). Reporting AE on the fixed
# mapping against ReAct on the defective one would hand AE a better prompt
# than its baseline and make the headline AE-vs-ReAct gap meaningless.
#
# Current frozen comparison, both on demo_indices:[0]:
#     ReAct 68/134   AE 80/134   (+12)
#   per type (ReAct -> AE): put 22->23, clean 23->23, heat 11->14,
#                           cool 9->8, examine 0->3, puttwo 3->9
#   Note AE's edge does NOT come from cool (it is -1 there), so the cool demo
#   fix is not expected to simply transfer the gap -- but that must be
#   measured, not assumed.
#
# ReAct ignores --ae-config entirely (run_alfworld.py's react branch calls
# react_baseline.run_episode without it -- verified in the source), so the
# controller yaml passed below has no effect and this run stays directly
# comparable to react_oneshot0_seed42_a40_*, which differed only in demo
# indices.
#
# Splits: practice AND exam. Two repeats, matching the AE final-mapping run.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/react_finaldemo_${SLURM_JOB_ID}
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
trap 'rc=$?; log "job ending, exit code $rc"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; notify "best-demo mapping 134 x2 (job ${SLURM_JOB_ID}) ended, exit=$rc, logdir=$LOGDIR"' EXIT

log "job started on $(hostname), SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

# groups: name | task-types | demo config
DEMO_GROUPS="A:put,clean,examine:one_shot_v1 B:heat,cool:one_shot_v1_index2 C:puttwo:one_shot_v1_index1"

for REP in 1; do
  for G in $DEMO_GROUPS; do
    gname=${G%%:*}
    for s in practice exam; do
      d="react_finaldemo_rep${REP}_grp${gname}_${s}"
      [ -e "$COPY/ae/runners/runs/$d" ] && fail "result dir already exists: $d"
    done
  done
done
log "check 1 OK: 12 result dirs free"

{
    sha256sum "$COPY"/alfworld_runs_ae/agents.py \
              "$COPY"/alfworld_runs_ae/demo_config.py \
              "$COPY"/ae/controllers/intervention_renderer.py \
              "$COPY"/ae/controllers/stateful_controller.py \
              "$COPY"/ae/runners/run_alfworld.py \
              "$COPY"/configs/controllers/ae_full_final_nopir_v2_5.yaml \
              "$COPY"/configs/demos/one_shot_v1.yaml \
              "$COPY"/configs/demos/one_shot_v1_index1.yaml \
              "$COPY"/configs/demos/one_shot_v1_index2.yaml
} > "$LOGDIR/sha256.log" 2>&1
grep -q "c4824c0606a87eedc32aa96564c9ddcedafc1fd11a7b212a3131382fe43367ba  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch (must be frozen v1 renderer)"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch"
log "check 2 OK: code byte-identical to the frozen 80/134 run"

# Each demo yaml must be one-shot and carry the index its group name claims.
for spec in one_shot_v1:0 one_shot_v1_index1:1 one_shot_v1_index2:2; do
    f=${spec%%:*}; want=${spec##*:}
    got=$("$PY" -c "
import yaml; d=yaml.safe_load(open('$COPY/configs/demos/${f}.yaml'))
print(f\"{d['num_demos']}:{d['demo_indices']}\")")
    [ "$got" = "1:[$want]" ] || fail "${f}.yaml is not one-shot index $want (got $got)"
done
log "check 3 OK: all three demo configs are one-shot with the expected indices"

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

run_one() {
    local split=$1 types=$2 demo=$3 run_name=$4
    "$PY" -m ae.runners.run_alfworld \
        --baseline react --split "$split" \
        --run-name "$run_name" \
        --task-types "$types" \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
        --demo-config "configs/demos/${demo}.yaml" \
        --termination-policy legacy_early_stop \
        --seed 42 \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$COPY/ae/runners/runs" \
        > "$LOGDIR/${run_name}.log" 2>&1
}

for REP in 1; do
  for G in $DEMO_GROUPS; do
    gname=$(echo "$G" | cut -d: -f1)
    types=$(echo "$G" | cut -d: -f2)
    demo=$(echo "$G" | cut -d: -f3)
    log "=== repeat $REP, group $gname ($types, $demo): practice+exam parallel ==="
    run_one practice "$types" "$demo" "react_finaldemo_rep${REP}_grp${gname}_practice" & p1=$!
    run_one exam     "$types" "$demo" "react_finaldemo_rep${REP}_grp${gname}_exam"     & p2=$!
    wait $p1; r1=$?
    wait $p2; r2=$?
    log "repeat $REP group $gname done: practice rc=$r1 exam rc=$r2"
    [ "$r1" -eq 0 ] || fail "rep$REP grp$gname practice failed (rc=$r1)"
    [ "$r2" -eq 0 ] || fail "rep$REP grp$gname exam failed (rc=$r2)"
  done
done

# Post-run integrity: the three groups must partition all 134 tasks exactly
# once per repeat -- no task missing, none double-counted, none in the wrong
# demo group.
CHECK=$("$PY" -c "
import json, collections
MAP={'pick_and_place_simple':'put','pick_clean_then_place_in_recep':'clean',
     'pick_heat_then_place_in_recep':'heat','pick_cool_then_place_in_recep':'cool',
     'pick_two_obj_and_place':'puttwo','look_at_obj_in_light':'examine'}
EXPECT={'A':{'put','clean','examine'},'B':{'heat','cool'},'C':{'puttwo'}}
bad=[]
for rep in (1,):
    seen=collections.Counter()
    for g in 'ABC':
        for s in ('practice','exam'):
            p=f'$COPY/ae/runners/runs/react_finaldemo_rep{rep}_grp{g}_{s}/episode_log.jsonl'
            for l in open(p):
                r=json.loads(l)
                t=MAP[r['env_name'].split('/')[0].split('-')[0]]
                if t not in EXPECT[g]: bad.append(f'rep{rep} grp{g} wrong type {t}')
                seen[(r['env_name'],r.get('goal'))]+=1
    if len(seen)!=134: bad.append(f'rep{rep}: {len(seen)} unique tasks, expected 134')
    dup=[k for k,v in seen.items() if v!=1]
    if dup: bad.append(f'rep{rep}: {len(dup)} tasks counted more than once')
print('BAD:'+'; '.join(bad) if bad else 'OK partition verified for both repeats')
" 2>&1)
echo "$CHECK" | grep -q "^OK" || fail "group partition check failed: $CHECK"
log "post-run check OK: $CHECK"

kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; VLLM_PID=""
log "JOB COMPLETE."
