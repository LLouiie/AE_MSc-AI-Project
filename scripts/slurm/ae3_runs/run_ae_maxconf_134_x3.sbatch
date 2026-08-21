#!/bin/bash
#SBATCH --job-name=ae_maxconf
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:30:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_maxconf-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_maxconf-%j.err

# ============================================================================
# MAX-SCORE configuration for one-shot AE, full 134, THREE repeats.
#
# Demo mapping: put/examine=idx0, heat/cool/clean=idx2, puttwo=idx1
# Controller:   REFLECT-v3 admissible-action grounding (admissible_reflect_workdir)
#
# Per-type basis (means of repeated draws):
#   put     idx0  23.0/24   -- ReflAct-confirmed, at ceiling, nothing to win
#   examine idx0   3.7/18   -- no index beats it; idx1 4(n=1), idx2 3.7
#   clean   idx2  25.5/31   -- NEW: idx0 23,23,23,23 vs idx2 26,25, ranges
#                              DISJOINT. clean had been frozen at idx0 all
#                              project on ReflAct-fidelity grounds, never tested.
#   heat    idx2  14.3/23   -- idx0 10.25, idx1 7.0
#   cool    idx2  17.3/21   -- idx0 9.8; the malformed-demo fix, strongest result
#   puttwo  idx1  10.3/17   -- idx0 7.4, idx2 5
#
# Measured stepping stones: demo mapping alone (clean at idx0) 87, 86, 88
# (mean 87.0, n=3); demo mapping + REFLECT-v3 89, 94 (mean 91.5, n=2). This
# run adds clean=idx2 on top of the latter.
#
# TWO HONEST CAVEATS, recorded before the run:
#   1. Adopting clean=idx2 means the reported configuration no longer uses the
#      ReflAct-confirmed demo for clean. put and clean were the ONLY two types
#      whose index-0 demo was verified verbatim against ReflAct Appendix K;
#      after this, only put retains that. Protocol fidelity to ReflAct is
#      knowingly traded for score, and must be stated wherever this number
#      appears.
#   2. REFLECT-v3's per-type effects have NOT been stable. It was first
#      credited with a reliable examine gain (7/18 in three runs) -- that did
#      not replicate when examine ran in a different task grouping (1/18,
#      4/18), and those three "7"s came from runs sharing a task ordering, so
#      they were correlated rather than independent evidence. What v3 did do
#      reproducibly in the combined runs is rescue heat (9.3 -> 16.0). Treat
#      the attribution as unsettled; only the total is being optimised here.
#
# Everything here is fitted on all 134 tasks per the user's explicit
# instruction, and is NOT a generalization result. Note also that the
# headline AE-vs-ReAct gap under a shared mapping is currently +1.67 with
# overlapping ranges (AE 87,86,88 vs ReAct 83,86,87) -- i.e. not established.
# Raising AE's absolute score does not address that.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/admissible_reflect_workdir
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_maxconf_${SLURM_JOB_ID}
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
DEMO_GROUPS="A:put,examine:one_shot_v1 B:heat,cool,clean:one_shot_v1_index2 C:puttwo:one_shot_v1_index1"

for REP in 1 2 3; do
  for G in $DEMO_GROUPS; do
    gname=${G%%:*}
    for s in practice exam; do
      d="ae_maxconf_rep${REP}_grp${gname}_${s}"
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
grep -q "52f20e7dfa794b44873e630ccf420fb78707edb12262787a3e539d8821186424  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch (must be the REFLECT-v3 renderer)"
grep -q "3127c243b5abcbd9a5115c3d01bd19098dc62527ece63ad099d207a25207164c  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch (must be the REFLECT-v3 controller)"
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
        --baseline ae_full --split "$split" \
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

for REP in 1 2 3; do
  for G in $DEMO_GROUPS; do
    gname=$(echo "$G" | cut -d: -f1)
    types=$(echo "$G" | cut -d: -f2)
    demo=$(echo "$G" | cut -d: -f3)
    log "=== repeat $REP, group $gname ($types, $demo): practice+exam parallel ==="
    run_one practice "$types" "$demo" "ae_maxconf_rep${REP}_grp${gname}_practice" & p1=$!
    run_one exam     "$types" "$demo" "ae_maxconf_rep${REP}_grp${gname}_exam"     & p2=$!
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
EXPECT={'A':{'put','examine'},'B':{'heat','cool','clean'},'C':{'puttwo'}}
bad=[]
for rep in (1,2,3):
    seen=collections.Counter()
    for g in 'ABC':
        for s in ('practice','exam'):
            p=f'$COPY/ae/runners/runs/ae_maxconf_rep{rep}_grp{g}_{s}/episode_log.jsonl'
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
