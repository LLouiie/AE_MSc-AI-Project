#!/bin/bash
#SBATCH --job-name=react_n5
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-4%4
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/react_n5-%A_%a.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/react_n5-%A_%a.err

# ============================================================================
# ReAct baseline, 5 INDEPENDENT repeats on the full 134 tasks, so the baseline
# row of the results table carries a real +- instead of a single number.
# One array task = one repeat = its own SLURM job = its own fresh vLLM server,
# which is the fresh-server-per-repeat discipline established by job 274469.
# (Cluster QOS caps this user at 6 submitted jobs; an array of 5 is fine.)
#
# DEMO MAPPING: deliberately IDENTICAL to the AE n=20 run (274469):
#     put,examine   -> one_shot_v1        (index 0)
#     heat,cool,clean -> one_shot_v1_index2 (index 2)
#     puttwo        -> one_shot_v1_index1 (index 1)
# The earlier ReAct n=3 (job 274054 "react_finaldemo") put `clean` on index 0.
# Comparing that against an AE run with `clean` on index 2 would mean the two
# methods saw different prompts, so any gap would be partly a prompt gap. With
# the mapping matched, the ONLY difference between this row and the AE n=20 row
# is the method itself, which is what a baseline comparison has to isolate.
# Consequence: these numbers supersede the old ReAct 85.3 +- 2.1 row rather
# than extending it -- do not pool the two.
#
# Everything else is held fixed: Qwen3-8B, temperature 0, seed 42 (inert here --
# see the n20 script header), legacy_early_stop, one-shot, default vLLM server
# flags so run-to-run variation matches every earlier run in this project.
#
# Fitted on all 134 tasks (practice + exam) per the user's explicit
# instruction. NOT a generalization result.
# ============================================================================

# ---------------------------------------------------------------------------
# RESUBMIT of repeats 1-4 only. In job 274830 those four died in preflight:
# the /gpudata quota was full, so `sha256sum ... > $LOGDIR/sha256.log` wrote an
# empty file and the sha grep found nothing, tripping the mismatch guard. No
# run directories were created, so this is a clean re-run, not a resume.
# Repeat 5 survived (it started after space was freed) and is kept -- hence
# ROOTLOG is pinned to the original 274830 log dir so all five score files
# land together.
# ---------------------------------------------------------------------------

set -uo pipefail

REP=${SLURM_ARRAY_TASK_ID}
COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
ROOTLOG=/vol/gpudata/jy625-ae-data/logs/react_n5_274830
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
notify() { curl -s -m 20 -d "$1" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true; }
trap 'rc=$?; log "job ending, exit $rc"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; notify "react_n5 rep${REP} (job ${SLURM_ARRAY_JOB_ID}_${REP}) ended, exit=$rc"' EXIT

# MUST match the AE n=20 grouping exactly.
DEMO_GROUPS="A:put,examine:one_shot_v1 B:heat,cool,clean:one_shot_v1_index2 C:puttwo:one_shot_v1_index1"

log "started on $(hostname)"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

# ---- preflight ----
for G in $DEMO_GROUPS; do
    gname=${G%%:*}
    for s in practice exam; do
        [ -e "$COPY/ae/runners/runs/react_n5_rep${REP}_grp${gname}_${s}" ] \
            && fail "result dir already exists: react_n5_rep${REP}_grp${gname}_${s}"
    done
done
log "check 1 OK: 6 result dirs free"

sha256sum "$COPY"/ae/controllers/intervention_renderer.py \
          "$COPY"/ae/controllers/stateful_controller.py \
          "$COPY"/ae/runners/run_alfworld.py > "$LOGDIR/sha256.log" 2>&1
grep -q "c4824c0606a87eedc32aa96564c9ddcedafc1fd11a7b212a3131382fe43367ba  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch"
log "check 2 OK: code byte-identical to the earlier ReAct baseline runs"

# Each demo yaml must still be one-shot and carry the index its group claims.
for spec in one_shot_v1:0 one_shot_v1_index1:1 one_shot_v1_index2:2; do
    f=${spec%%:*}; want=${spec##*:}
    got=$("$PY" -c "
import yaml; d=yaml.safe_load(open('$COPY/configs/demos/${f}.yaml'))
print(f\"{d['num_demos']}:{d['demo_indices']}\")")
    [ "$got" = "1:[$want]" ] || fail "${f}.yaml is not one-shot index $want (got $got)"
done
log "check 3 OK: all three demo configs are one-shot with the expected indices"

cd "$COPY" || fail "cannot cd into $COPY"
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate

# ---- fresh vLLM for this repeat ----
PORT=$(( 8000 + (SLURM_ARRAY_JOB_ID % 800) + REP ))
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
for t in $(seq 1 60); do
    curl -s -o /dev/null -w "" "$BASE_URL/models" 2>/dev/null && { READY=1; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM died during startup -- see $LOGDIR/vllm_server.log"
    sleep 10
done
[ "$READY" -eq 1 ] || fail "vLLM not ready within 10 minutes"
kill -0 "$VLLM_PID" 2>/dev/null || fail "port answered but OUR vLLM is dead"
log "vLLM ready (pid=$VLLM_PID)"

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

for G in $DEMO_GROUPS; do
    gname=$(echo "$G" | cut -d: -f1)
    types=$(echo "$G" | cut -d: -f2)
    demo=$(echo "$G" | cut -d: -f3)
    log "=== group $gname ($types, $demo): practice+exam in parallel ==="
    run_one practice "$types" "$demo" "react_n5_rep${REP}_grp${gname}_practice" & p1=$!
    run_one exam     "$types" "$demo" "react_n5_rep${REP}_grp${gname}_exam"     & p2=$!
    wait $p1; r1=$?
    wait $p2; r2=$?
    [ "$r1" -eq 0 ] || fail "grp${gname} practice failed (rc=$r1)"
    [ "$r2" -eq 0 ] || fail "grp${gname} exam failed (rc=$r2)"
    log "group $gname done"
done

kill "$VLLM_PID" 2>/dev/null; VLLM_PID=""

# ---- post-run integrity: the three groups must partition all 134 tasks
#      exactly once, with every task in the group its type belongs to ----
OUT=$("$PY" -c "
import json, collections
MAP={'pick_and_place_simple':'put','pick_clean_then_place_in_recep':'clean',
     'pick_heat_then_place_in_recep':'heat','pick_cool_then_place_in_recep':'cool',
     'pick_two_obj_and_place':'puttwo','look_at_obj_in_light':'examine'}
EXPECT={'A':{'put','examine'},'B':{'heat','cool','clean'},'C':{'puttwo'}}
bad=[]; seen=collections.Counter(); score=0; calls=0
for g in 'ABC':
    for sp in ('practice','exam'):
        p=f'$COPY/ae/runners/runs/react_n5_rep${REP}_grp{g}_{sp}/episode_log.jsonl'
        for l in open(p):
            r=json.loads(l)
            t=MAP[r['env_name'].split('/')[0].split('-')[0]]
            if t not in EXPECT[g]: bad.append(f'grp{g} wrong type {t}')
            seen[(r['env_name'],r.get('goal'))]+=1
            score+=r['success']; calls+=r.get('llm_calls') or 0
if len(seen)!=134: bad.append(f'{len(seen)} unique tasks, expected 134')
dup=[k for k,v in seen.items() if v!=1]
if dup: bad.append(f'{len(dup)} tasks counted more than once')
print(('BAD:'+'; '.join(bad)) if bad else f'OK {score} {calls}')
" 2>&1)
echo "$OUT" | grep -q "^OK" || fail "integrity check failed: $OUT"
SCORE=$(echo "$OUT" | cut -d' ' -f2)
CALLS=$(echo "$OUT" | cut -d' ' -f3)
echo "$SCORE" > "$ROOTLOG/score_rep${REP}.txt"
echo "$CALLS" > "$ROOTLOG/llmcalls_rep${REP}.txt"
log "REP ${REP} COMPLETE: ${SCORE}/134, ${CALLS} llm calls"
