#!/bin/bash
#SBATCH --job-name=adapt_n10
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/adapt_n10-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/adapt_n10-%j.err

# Ten full paper-protocol ADaPT repeats. They run sequentially in one Slurm
# job to fit the six-submitted-job QOS cap. vLLM is restarted per repeat.

set -uo pipefail

ADAPT=/homes/jy625/projects/AE3/external/ADaPT
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/adapt_n10_${SLURM_JOB_ID}
METRICDIR=$LOGDIR/metrics
mkdir -p "$LOGDIR" "$METRICDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export ALFWORLD_DATA=/vol/gpudata/jy625-ae-data/alfworld_data

MODEL_SOURCE=/vol/gpudata/jy625-ae-data/models/Qwen3-8B
NODE_TMP=${SLURM_TMPDIR:-/tmp/ae3_${USER}_${SLURM_JOB_ID}}
LOCAL_MODEL=$NODE_TMP/Qwen3-8B
MODEL=Qwen/Qwen3-8B
VLLM_PID=""

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() {
    echo "[$(date -Is)] FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/failure.log"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}
notify() { curl -s -m 20 -H "Title: $1" -d "$2" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true; }
on_exit() {
    rc=$?
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    if [ "${STAGED_LOCAL:-0}" = 1 ] && [ -z "${SLURM_TMPDIR:-}" ]; then
        rm -rf "$NODE_TMP"
    fi
    log "job ending, exit $rc"
    if [ "$rc" -eq 0 ]; then
        notify "OK - ADaPT n=10" "$(tail -c 900 "$LOGDIR/RESULT.txt" 2>/dev/null)"
    else
        notify "FAILED - ADaPT n=10" "job $SLURM_JOB_ID: $(tail -4 "$LOGDIR/batch.log" 2>/dev/null | head -c 750)"
    fi
}
trap on_exit EXIT

[ -d "$ALFWORLD_DATA/json_2.1.1/valid_unseen" ] || fail "ALFWORLD_DATA is incomplete"
cd "$ADAPT" || fail "cannot cd into $ADAPT"
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate

mkdir -p "$NODE_TMP" || fail "cannot create $NODE_TMP"
avail_kb=$(df -Pk "$NODE_TMP" | awk 'NR==2 {print $4}')
STAGED_LOCAL=0
if [ "${avail_kb:-0}" -ge 20971520 ]; then
    log "staging model to node-local storage"
    cp -a "$MODEL_SOURCE" "$LOCAL_MODEL" || fail "model staging failed"
    STAGED_LOCAL=1
else
    LOCAL_MODEL=$MODEL_SOURCE
    log "only ${avail_kb:-0} KiB free in $NODE_TMP; using shared model path"
fi

for repeat_id in $(seq 0 9); do
    FNAME=adapt_n10_${SLURM_JOB_ID}_r${repeat_id}
    DEST_DIR=$ADAPT/results/comparison/Qwen/Qwen3-8B/${FNAME}_Qwen
    RESULT_JSON=$DEST_DIR/Qwen3-8B.json
    [ ! -e "$DEST_DIR" ] || fail "collision: $DEST_DIR exists"
    mkdir -p "$DEST_DIR"

    PORT=$(( 8200 + ((SLURM_JOB_ID + repeat_id) % 600) ))
    curl -s -o /dev/null -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null && fail "port $PORT busy"
    log "repeat $repeat_id: starting fresh vLLM on $PORT"
    vllm serve "$LOCAL_MODEL" \
        --served-model-name "$MODEL" --port "$PORT" --host 0.0.0.0 \
        > "$LOGDIR/vllm_repeat_${repeat_id}.log" 2>&1 &
    VLLM_PID=$!
    export OPENAI_BASE_URL=http://localhost:${PORT}/v1
    READY=0
    for t in $(seq 1 270); do
        curl -s -o /dev/null "$OPENAI_BASE_URL/models" 2>/dev/null && { READY=1; break; }
        kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM died starting repeat $repeat_id"
        sleep 10
    done
    [ "$READY" -eq 1 ] || fail "vLLM not ready for repeat $repeat_id within 45 minutes"
    log "repeat $repeat_id: vLLM ready"

    "$PY" run_alfworld.py \
        --LM "$MODEL" --exec-LM "$MODEL" \
        --eval-type ood --eval-all True \
        --executor hybrid \
        --max-depth 3 --max-runs 20 \
        --fname "$FNAME" --verbose True \
        > "$LOGDIR/repeat_${repeat_id}.log" 2>&1
    rc=$?
    kill "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    [ "$rc" -eq 0 ] || fail "repeat $repeat_id runner exited rc=$rc"
    [ -s "$RESULT_JSON" ] || fail "repeat $repeat_id wrote no result JSON"

    "$PY" - "$RESULT_JSON" "$repeat_id" "$METRICDIR/repeat_${repeat_id}.json" <<'PYEOF' || fail "metrics failed for repeat $repeat_id"
import json, os, sys
src, rep, out = sys.argv[1:]
d = json.load(open(src))
tasks = [v for group, entries in d.items() if group != "overall" for v in entries.values()]
assert len(tasks) == 134, f"expected 134 tasks, got {len(tasks)}"
# run_alfworld.py increments `runs` once for every executor LLM call and once
# for each planner call. Every executor call immediately executes env.step.
# A decomposition marker (or INVALID PLAN) is emitted once per planner call.
planner_calls = sum(
    1 for task in tasks for p in task.get("plans", [])
    if " and logic " in p or p.startswith("INVALID PLAN")
)
llm_calls = sum(int(task["runs"]) for task in tasks)
steps = llm_calls - planner_calls
successes = sum(int(task["reward"]) for task in tasks)
metric = {
    "repeat": int(rep), "episodes": len(tasks), "successes": successes,
    "success_rate": successes / len(tasks), "llm_calls": llm_calls,
    "planner_calls": planner_calls, "steps": steps,
    "mean_llm_calls": llm_calls / len(tasks), "mean_steps": steps / len(tasks),
}
tmp = out + ".tmp"
with open(tmp, "w") as f: json.dump(metric, f, sort_keys=True)
os.replace(tmp, out)
print(json.dumps(metric, sort_keys=True))
PYEOF
    log "repeat $repeat_id complete: $(cat "$METRICDIR/repeat_${repeat_id}.json")"
done

"$PY" - "$METRICDIR" > "$LOGDIR/RESULT.txt" <<'PYEOF'
import glob, json, sys
rows = [json.load(open(p)) for p in sorted(glob.glob(sys.argv[1] + "/repeat_*.json"))]
assert len(rows) == 10, f"expected 10 repeats, got {len(rows)}"
for r in sorted(rows, key=lambda x: x["repeat"]):
    print(f"repeat {r['repeat']:2d}: success={r['successes']:3d}/134 ({r['success_rate']:.1%}), "
          f"LLM calls={r['llm_calls']:5d} (mean {r['mean_llm_calls']:.2f}), "
          f"steps={r['steps']:5d} (mean {r['mean_steps']:.2f})")
successes = sum(r["successes"] for r in rows)
episodes = sum(r["episodes"] for r in rows)
calls = sum(r["llm_calls"] for r in rows)
steps = sum(r["steps"] for r in rows)
print(f"OVERALL n=10: success={successes}/{episodes} ({successes/episodes:.1%}), "
      f"LLM calls={calls} (mean {calls/episodes:.2f}), "
      f"steps={steps} (mean {steps/episodes:.2f})")
PYEOF
log "ADaPT n=10 RUN COMPLETED"
