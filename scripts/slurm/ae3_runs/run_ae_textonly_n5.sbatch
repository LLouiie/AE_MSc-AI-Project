#!/bin/bash
#SBATCH --job-name=ae_textonly
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-5%5
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_textonly-%A_%a.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_textonly-%A_%a.err

# ============================================================================
# ABLATION: environment-agnostic signals only. 5 independent repeats, full 134.
# One array task = one repeat = its own SLURM job = its own fresh vLLM server,
# the same discipline as the AE n=20 run (274469).
#
# The ONLY difference from that n=20 run is --ae-config: this uses
# ae_ablation_text_only_signals.yaml, which zeroes the five weights belonging
# to the three signals that cannot be computed without environment-specific
# knowledge (invalid_action, repeated_action, unexpected_outcome). Verified by
# loading both configs and diffing the dataclasses: exactly 5 fields differ and
# nothing else. Demo mapping, controller thresholds, warmup/cooldown/budget,
# seed, termination policy and server flags are all identical, so any
# difference in outcome is attributable to the zeroed signals alone.
#
# Purpose: this is the evidence for the generality claim. If the score holds up
# without the environment adapter, AE needs no environment-specific code; if it
# drops, the adapter is demonstrably doing work. Both outcomes are reportable.
# Compare to AE-full PAIRED PER TASK (McNemar), never by aggregate deltas --
# all the statistical power is in the ~16% of tasks that flip between repeats.
#
# Fitted on all 134 tasks (practice + exam) per the user's explicit
# instruction. NOT a generalization result.
# ============================================================================

set -uo pipefail

REP=${SLURM_ARRAY_TASK_ID}
COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/admissible_reflect_workdir
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
ROOTLOG=/vol/gpudata/jy625-ae-data/logs/ae_textonly_${SLURM_ARRAY_JOB_ID}
LOGDIR=$ROOTLOG/rep${REP}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

log() { echo "[$(date -Is)] [rep${REP}] $*" | tee -a "$LOGDIR/batch.log"; }
fail() {
    echo "[$(date -Is)] [rep${REP}] FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/failure.log"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}

# --- push notification -------------------------------------------------
# The job reports its own outcome to ntfy so results arrive without anyone
# watching the terminal. -m 20 and `|| true` so a slow or unreachable ntfy
# can never wedge or fail the job it is reporting on.
JOBLABEL="AE text-only-signals ablation"
notify() { curl -s -m 20 -H "Title: $1" -d "$2" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true; }
on_exit() {
    rc=$?
    log "job ending, exit $rc"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    if [ "$rc" -eq 0 ] && [ -s "$LOGDIR/RESULT.txt" ]; then
        notify "OK - $JOBLABEL" "$(head -c 900 "$LOGDIR/RESULT.txt")"
    elif [ "$rc" -eq 0 ]; then
        notify "OK - $JOBLABEL" "job ${SLURM_JOB_ID:-$SLURM_ARRAY_JOB_ID} finished, exit 0. $(tail -2 "$LOGDIR/batch.log" 2>/dev/null | head -c 700)"
    else
        notify "FAILED rc=$rc - $JOBLABEL" "job ${SLURM_JOB_ID:-$SLURM_ARRAY_JOB_ID}: $(tail -4 "$LOGDIR/batch.log" 2>/dev/null | head -c 800)"
    fi
}
trap on_exit EXIT

DEMO_GROUPS="A:put,examine:one_shot_v1 B:heat,cool,clean:one_shot_v1_index2 C:puttwo:one_shot_v1_index1"
ABL=configs/controllers/ae_ablation_text_only_signals.yaml

log "started on $(hostname)"

for G in $DEMO_GROUPS; do
    gname=${G%%:*}
    for s in practice exam; do
        [ -e "$COPY/ae/runners/runs/ae_textonly_rep${REP}_grp${gname}_${s}" ] \
            && fail "result dir already exists: ae_textonly_rep${REP}_grp${gname}_${s}"
    done
done
log "check 1 OK: 6 result dirs free"

# Controller code must be byte-identical to the AE n=20 run this is compared to.
sha256sum "$COPY"/ae/controllers/intervention_renderer.py \
          "$COPY"/ae/controllers/stateful_controller.py > "$LOGDIR/sha256.log" 2>&1
grep -q "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch"
grep -q "3127c243b5abcbd9a5115c3d01bd19098dc62527ece63ad099d207a25207164c  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch"
log "check 2 OK: controller code matches the AE n=20 baseline"

cd "$COPY" || fail "cannot cd into $COPY"

# The ablation must differ from the baseline config in exactly the 5 intended
# weights. A typo that silently changed a threshold would invalidate the whole
# comparison, so assert it here rather than trusting the file.
"$PY" -c "
import sys, dataclasses; sys.path.insert(0,'.')
from ae.controllers.config import load_config
a=load_config('configs/controllers/ae_full_final_nopir_v2_5.yaml')
b=load_config('$ABL')
def flat(c,p=''):
    d={}
    for f in dataclasses.fields(c):
        v=getattr(c,f.name)
        d.update(flat(v,p+f.name+'.')) if dataclasses.is_dataclass(v) else d.update({p+f.name:v})
    return d
x,y=flat(a),flat(b)
diff=sorted(k for k in x if x[k]!=y[k])
expect=['confidence_weights.invalid_action','confidence_weights.repeated_action',
        'frustration_weights.invalid_action','frustration_weights.repeated_action',
        'surprise_weights.unexpected_outcome']
assert diff==expect, 'config diff is %r, expected %r' % (diff, expect)
assert all(y[k]==0.0 for k in expect), 'ablated weights are not all zero'
print('config diff verified')" > "$LOGDIR/config_check.log" 2>&1
grep -q "config diff verified" "$LOGDIR/config_check.log" || fail "config check failed: $(cat $LOGDIR/config_check.log)"
log "check 3 OK: ablation differs from baseline in exactly the 5 intended weights"

source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate

PORT=$(( 9500 + (SLURM_ARRAY_JOB_ID % 400) + REP ))
if curl -s -o /dev/null -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null; then
    fail "port ${PORT} already serving -- refusing to attach to another job's vLLM"
fi
log "starting vLLM on port ${PORT}"
vllm serve /vol/gpudata/jy625-ae-data/models/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B --port "${PORT}" --host 0.0.0.0 \
    > "$LOGDIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
BASE_URL="http://localhost:${PORT}/v1"
READY=0
# 20 minutes, not 10: repeat 5 of the ReAct run died at exactly the 10-minute
# mark loading the 16GB model off a busy /gpudata. The extra headroom is free.
for t in $(seq 1 270); do
    curl -s -o /dev/null -w "" "$BASE_URL/models" 2>/dev/null && { READY=1; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM died during startup -- see $LOGDIR/vllm_server.log"
    sleep 10
done
[ "$READY" -eq 1 ] || fail "vLLM not ready within 45 minutes"
log "vLLM ready (pid=$VLLM_PID)"

run_one() {
    local split=$1 types=$2 demo=$3 run_name=$4
    "$PY" -m ae.runners.run_alfworld \
        --baseline ae_full --split "$split" \
        --run-name "$run_name" \
        --task-types "$types" \
        --ae-config "$ABL" \
        --demo-config "configs/demos/${demo}.yaml" \
        --termination-policy legacy_early_stop \
        --seed 42 \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$COPY/ae/runners/runs" \
        > "$LOGDIR/${run_name}.log" 2>&1
}

for G in $DEMO_GROUPS; do
    gname=$(echo "$G" | cut -d: -f1)
    types=$(echo "$G" | cut -d: -f2)
    demo=$(echo "$G" | cut -d: -f3)
    log "=== group $gname ($types, $demo): practice+exam in parallel ==="
    run_one practice "$types" "$demo" "ae_textonly_rep${REP}_grp${gname}_practice" & p1=$!
    run_one exam     "$types" "$demo" "ae_textonly_rep${REP}_grp${gname}_exam"     & p2=$!
    wait $p1; r1=$?
    wait $p2; r2=$?
    [ "$r1" -eq 0 ] || fail "grp${gname} practice failed (rc=$r1)"
    [ "$r2" -eq 0 ] || fail "grp${gname} exam failed (rc=$r2)"
    log "group $gname done"
done

kill "$VLLM_PID" 2>/dev/null; VLLM_PID=""

OUT=$("$PY" -c "
import json, collections
MAP={'pick_and_place_simple':'put','pick_clean_then_place_in_recep':'clean',
     'pick_heat_then_place_in_recep':'heat','pick_cool_then_place_in_recep':'cool',
     'pick_two_obj_and_place':'puttwo','look_at_obj_in_light':'examine'}
EXPECT={'A':{'put','examine'},'B':{'heat','cool','clean'},'C':{'puttwo'}}
bad=[]; seen=collections.Counter(); score=0
for g in 'ABC':
    for sp in ('practice','exam'):
        p=f'$COPY/ae/runners/runs/ae_textonly_rep${REP}_grp{g}_{sp}/episode_log.jsonl'
        for l in open(p):
            r=json.loads(l)
            t=MAP[r['env_name'].split('/')[0].split('-')[0]]
            if t not in EXPECT[g]: bad.append(f'grp{g} wrong type {t}')
            seen[(r['env_name'],r.get('goal'))]+=1; score+=r['success']
if len(seen)!=134: bad.append(f'{len(seen)} unique tasks, expected 134')
dup=[k for k,v in seen.items() if v!=1]
if dup: bad.append(f'{len(dup)} tasks counted more than once')
print(('BAD:'+'; '.join(bad)) if bad else f'OK {score}')" 2>&1)
echo "$OUT" | grep -q "^OK" || fail "integrity check failed: $OUT"
SCORE=$(echo "$OUT" | cut -d' ' -f2)
echo "$SCORE" > "$ROOTLOG/score_rep${REP}.txt"
log "REP ${REP} COMPLETE: ${SCORE}/134"
