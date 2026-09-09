#!/bin/bash
#SBATCH --job-name=ae_n20
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=09:00:00
#SBATCH --array=1-5%5
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_n20-%A_%a.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_n20-%A_%a.err

# ============================================================================
# 20 INDEPENDENT REPEATS of the max-score one-shot AE configuration, full 134.
# Submitted as 5 array tasks x 4 repeats each: the cluster QOS caps this user
# at 6 submitted jobs, so a 20-task array is rejected outright.
#   task 1 -> repeats 1-4, task 2 -> 5-8, ... task 5 -> 17-20
#
# Config (identical to job 274158 "ae_maxconf", which gave 91/100/103):
#   demos      put,examine=idx0 | heat,cool,clean=idx2 | puttwo=idx1
#   controller ae_full_final_nopir_v2_5 + REFLECT-v3 admissible grounding
#   seed 42, temperature 0, legacy_early_stop
#
# ONE deliberate difference from 274158: EVERY repeat gets its own freshly
# started vLLM server. In 274158 all three repeats shared one long-lived
# server. The determinism probe (job 274442) showed why that matters:
#
#   ARM A, default server flags, same 40 tasks twice back-to-back on one
#          server: 30/40 trajectories differed, 5 tasks flipped, 23 -> 26
#   ARM B, --enforce-eager --no-enable-prefix-caching --max-num-seqs 1:
#          37/37 trajectories BYTE-IDENTICAL, 0 flips
#
# So run-to-run variation comes from prefix caching / CUDA graphs, NOT from
# the seed -- the seed is inert (run_alfworld.py:95 calls random.seed() and
# `random.` appears nowhere else in ae/; both LLMs are temperature=0).
# Sharing a server additionally lets the prefix cache warm across repeats,
# biasing later repeats upward: exactly the monotone 91 -> 100 -> 103 of
# 274158, where a Cochran Q permutation test rejected exchangeability of the
# three repeats at p=0.0034.
#
# Default (nondeterministic) server flags are KEPT on purpose so these 20
# repeats vary the way all earlier runs did and stay comparable with them.
# The fresh-server-per-repeat removes the warm-cache drift, so the 20 draws
# are exchangeable and a mean +- sd over them is meaningful.
#
# Every number this produces is fitted on all 134 tasks (practice+exam) per
# the user's explicit instruction, and is NOT a generalization result.
# ============================================================================

set -uo pipefail

TASK=${SLURM_ARRAY_TASK_ID}
REPS_PER_TASK=4
COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/admissible_reflect_workdir
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
ROOTLOG=/vol/gpudata/jy625-ae-data/logs/ae_n20_${SLURM_ARRAY_JOB_ID}
mkdir -p "$ROOTLOG"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

log() { echo "[$(date -Is)] [task${TASK}] $*" | tee -a "$ROOTLOG/task${TASK}.log"; }
fail() {
    echo "[$(date -Is)] [task${TASK}] FAILURE: $*" | tee -a "$ROOTLOG/task${TASK}.log" "$ROOTLOG/task${TASK}.failure"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}
trap 'rc=$?; log "task ending, exit $rc"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null' EXIT

DEMO_GROUPS="A:put,examine:one_shot_v1 B:heat,cool,clean:one_shot_v1_index2 C:puttwo:one_shot_v1_index1"

log "starting on $(hostname), repeats $(( (TASK-1)*REPS_PER_TASK + 1 ))-$(( TASK*REPS_PER_TASK ))"

# ---- preflight (once per task) ----
sha256sum "$COPY"/ae/controllers/intervention_renderer.py \
          "$COPY"/ae/controllers/stateful_controller.py > "$ROOTLOG/sha256_task${TASK}.log" 2>&1
grep -q "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f  $COPY/ae/controllers/intervention_renderer.py" "$ROOTLOG/sha256_task${TASK}.log" \
    || fail "intervention_renderer.py sha mismatch"
grep -q "3127c243b5abcbd9a5115c3d01bd19098dc62527ece63ad099d207a25207164c  $COPY/ae/controllers/stateful_controller.py" "$ROOTLOG/sha256_task${TASK}.log" \
    || fail "stateful_controller.py sha mismatch"
log "preflight OK: controller code matches 274158"

cd "$COPY" || fail "cannot cd into $COPY"
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate

run_one() {
    local split=$1 types=$2 demo=$3 run_name=$4 logd=$5
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
        > "${logd}/${run_name}.log" 2>&1
}

for i in $(seq 1 "$REPS_PER_TASK"); do
    REP=$(( (TASK-1)*REPS_PER_TASK + i ))
    LOGDIR="$ROOTLOG/rep${REP}"
    mkdir -p "$LOGDIR"

    for G in $DEMO_GROUPS; do
        gname=${G%%:*}
        for s in practice exam; do
            [ -e "$COPY/ae/runners/runs/ae_n20_rep${REP}_grp${gname}_${s}" ] \
                && fail "result dir already exists: ae_n20_rep${REP}_grp${gname}_${s}"
        done
    done

    # ---- fresh server for THIS repeat ----
    PORT=$(( 9000 + (SLURM_ARRAY_JOB_ID % 400) + REP ))
    if curl -s -o /dev/null -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null; then
        fail "port ${PORT} already serving -- refusing to attach to another job's vLLM"
    fi
    log "rep${REP}: starting fresh vLLM on port ${PORT}"
    vllm serve /vol/gpudata/jy625-ae-data/models/Qwen3-8B \
        --served-model-name Qwen/Qwen3-8B --port "${PORT}" --host 0.0.0.0 \
        > "$LOGDIR/vllm_server.log" 2>&1 &
    VLLM_PID=$!
    BASE_URL="http://localhost:${PORT}/v1"
    READY=0
    for t in $(seq 1 60); do
        curl -s -o /dev/null -w "" "$BASE_URL/models" 2>/dev/null && { READY=1; break; }
        kill -0 "$VLLM_PID" 2>/dev/null || fail "rep${REP}: vLLM died during startup"
        sleep 10
    done
    [ "$READY" -eq 1 ] || fail "rep${REP}: vLLM not ready within 10 minutes"
    kill -0 "$VLLM_PID" 2>/dev/null || fail "rep${REP}: port answered but OUR vLLM is dead"
    log "rep${REP}: vLLM ready (pid=$VLLM_PID)"

    for G in $DEMO_GROUPS; do
        gname=$(echo "$G" | cut -d: -f1)
        types=$(echo "$G" | cut -d: -f2)
        demo=$(echo "$G" | cut -d: -f3)
        run_one practice "$types" "$demo" "ae_n20_rep${REP}_grp${gname}_practice" "$LOGDIR" & p1=$!
        run_one exam     "$types" "$demo" "ae_n20_rep${REP}_grp${gname}_exam"     "$LOGDIR" & p2=$!
        wait $p1; r1=$?
        wait $p2; r2=$?
        [ "$r1" -eq 0 ] || fail "rep${REP} grp${gname} practice failed (rc=$r1)"
        [ "$r2" -eq 0 ] || fail "rep${REP} grp${gname} exam failed (rc=$r2)"
    done

    # tear the server down so the next repeat starts cold
    kill "$VLLM_PID" 2>/dev/null
    for t in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 2; done
    VLLM_PID=""
    sleep 5

    TOT=$("$PY" -c "
import json
seen=set(); s=0
for g in 'ABC':
    for sp in ('practice','exam'):
        for l in open(f'$COPY/ae/runners/runs/ae_n20_rep${REP}_grp{g}_{sp}/episode_log.jsonl'):
            x=json.loads(l); seen.add((x['env_name'],x['goal'])); s+=x['success']
print(f'{len(seen)} {s}')")
    NTASK=$(echo "$TOT" | cut -d' ' -f1)
    SCORE=$(echo "$TOT" | cut -d' ' -f2)
    [ "$NTASK" = "134" ] || fail "rep${REP} covered $NTASK unique tasks, expected 134"
    echo "$SCORE" > "$ROOTLOG/score_rep${REP}.txt"
    log "rep${REP} COMPLETE: ${SCORE}/134"
done

log "TASK ${TASK} COMPLETE (all ${REPS_PER_TASK} repeats)"
