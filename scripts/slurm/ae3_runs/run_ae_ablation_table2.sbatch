#!/bin/bash
#SBATCH --job-name=ae_abl_t2
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --array=0-5%5
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_abl_t2-%A_%a.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_abl_t2-%A_%a.err

set -uo pipefail

MODES=(full no_trigger random_trigger reflect_only replan_only verify_only)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
COPY=/homes/jy625/projects/AE3
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
VLLM=/vol/gpudata/jy625-ae-data/ae3-venv/bin/vllm
SOURCE_MODEL=/vol/gpudata/jy625-ae-data/models/Qwen3-8B
MODEL=/data/jy625-ae/Qwen3-8B
ROOT=/vol/gpudata/jy625-ae-data/logs/ae_ablation_table2_${SLURM_ARRAY_JOB_ID}
MODE_ROOT=$ROOT/$MODE
RUNS=$MODE_ROOT/runs
mkdir -p "$RUNS"
export TMPDIR=/data/jy625-ae/tmp
export XDG_CACHE_HOME=/data/jy625-ae/cache
export TORCHINDUCTOR_CACHE_DIR=/data/jy625-ae/torchinductor_cache
export VLLM_CACHE_ROOT=/data/jy625-ae/vllm_cache
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ "${RUN_FULL:-0}" = 1 ]; then
    EXPECTED=134
    LIMIT_ARGS=()
    PHASE=full134
else
    EXPECTED=12
    LIMIT_ARGS=(--limit 2)
    PHASE=smoke12
fi

log() { echo "[$(date -Is)] [$MODE/$PHASE] $*" | tee -a "$MODE_ROOT/batch.log"; }
notify() {
    curl -fsS -m 10 -d "$1" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
}
fail() { log "FAILURE: $*"; notify "AE ablation $MODE $PHASE FAILED: $*"; exit 1; }
trap 'rc=$?; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true; exit $rc' EXIT

cd "$COPY" || fail "cannot cd to repo"

mkdir -p /data/jy625-ae "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" || fail "cannot create node-local cache dirs"
if [ ! -f "$MODEL/.ae3_complete" ]; then
    log "staging model from Ceph to node-local $MODEL"
    STAGE=$(mktemp -d /data/jy625-ae/Qwen3-8B.stage.XXXXXX) || fail "mktemp failed"
    cp -a "$SOURCE_MODEL/." "$STAGE/" || fail "node-local model copy failed"
    touch "$STAGE/.ae3_complete"
    if [ -e "$MODEL" ]; then
        fail "incomplete node-local destination already exists: $MODEL"
    fi
    mv "$STAGE" "$MODEL" || fail "cannot publish node-local model cache"
    log "node-local model staging complete"
else
    log "reusing node-local model cache $MODEL"
fi

PORT=$((9300 + (SLURM_ARRAY_JOB_ID % 300) + SLURM_ARRAY_TASK_ID))
curl -s -o /dev/null -m 2 "http://localhost:${PORT}/v1/models" && fail "port $PORT occupied"
log "starting fresh vLLM on $(hostname) port $PORT"
"$VLLM" serve "$MODEL" --served-model-name Qwen/Qwen3-8B --port "$PORT" --host 0.0.0.0 \
    > "$MODE_ROOT/vllm.log" 2>&1 &
VLLM_PID=$!
BASE_URL=http://localhost:${PORT}/v1
for _ in $(seq 1 1440); do
    curl -fsS -m 2 "$BASE_URL/models" >/dev/null 2>&1 && break
    kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM exited"
    sleep 5
done
curl -fsS -m 2 "$BASE_URL/models" >/dev/null 2>&1 || fail "vLLM not ready"

run_group() {
    split=$1; types=$2; demo=$3; group=$4
    name=table2_${PHASE}_${MODE}_${group}_${split}
    "$PY" -m ae.runners.run_alfworld \
        --baseline ae_full --ae-ablation "$MODE" --ae-random-seed 42 \
        --split "$split" --task-types "$types" "${LIMIT_ARGS[@]}" \
        --run-name "$name" --output-dir "$RUNS" \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
        --demo-config "configs/demos/${demo}.yaml" \
        --termination-policy legacy_early_stop --seed 42 \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        > "$MODE_ROOT/${name}.log" 2>&1 || fail "$name runner failed"
}

run_group practice put,examine one_shot_v1 A
run_group exam put,examine one_shot_v1 A
run_group practice heat,cool,clean one_shot_v1_index2 B
run_group exam heat,cool,clean one_shot_v1_index2 B
run_group practice puttwo one_shot_v1_index1 C
run_group exam puttwo one_shot_v1_index1 C

mapfile -t LOGS < <(find "$RUNS" -name episode_log.jsonl -type f | sort)
test "${#LOGS[@]}" = 6 || fail "expected 6 logs, got ${#LOGS[@]}"
"$PY" scripts/analyze_ae_ablation.py --mode "$MODE" --repeat 1 \
    --expected-n "$EXPECTED" --output-dir "$MODE_ROOT" "${LOGS[@]}" \
    > "$MODE_ROOT/summary.stdout" || fail "summary validation failed"
notify "AE ablation $MODE $PHASE done: $(tr '\n' ' ' < "$MODE_ROOT/summary.json")"
log "DONE"
