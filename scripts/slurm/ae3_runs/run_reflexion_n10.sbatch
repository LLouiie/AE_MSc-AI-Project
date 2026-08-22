#!/bin/bash
#SBATCH --job-name=reflex_n10
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --array=0-3
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/reflex_n10-%A_%a.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/reflex_n10-%A_%a.err

# Ten independent matched-protocol Reflexion repeats.  Four array tasks keep
# the submission within the six-job QOS cap; repeats are distributed 3+3+2+2.
# vLLM is restarted for every repeat so no repeat shares server state.

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
ROOTLOG=/vol/gpudata/jy625-ae-data/logs/reflex_n10_${SLURM_ARRAY_JOB_ID}
LOGDIR=${ROOTLOG}/task_${SLURM_ARRAY_TASK_ID}
METRICDIR=${ROOTLOG}/metrics
mkdir -p "$LOGDIR" "$METRICDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL_SOURCE=/vol/gpudata/jy625-ae-data/models/Qwen3-8B
NODE_TMP=${SLURM_TMPDIR:-/tmp/ae3_${USER}_${SLURM_JOB_ID}}
LOCAL_MODEL=$NODE_TMP/Qwen3-8B
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
    log "array task ending, exit $rc"
    if [ "$rc" -eq 0 ]; then
        notify "OK - Reflexion n10 task ${SLURM_ARRAY_TASK_ID}" "$(tail -c 900 "$LOGDIR/RESULT.txt" 2>/dev/null)"
    else
        notify "FAILED - Reflexion n10 task ${SLURM_ARRAY_TASK_ID}" "job ${SLURM_JOB_ID}: $(tail -4 "$LOGDIR/batch.log" 2>/dev/null | head -c 750)"
    fi
}
trap on_exit EXIT

cd "$COPY" || fail "cannot cd into $COPY"
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
mkdir -p "$NODE_TMP" || fail "cannot create $NODE_TMP"
avail_kb=$(df -Pk "$NODE_TMP" | awk 'NR==2 {print $4}')
[ "${avail_kb:-0}" -ge 20971520 ] || fail "less than 20 GiB free in $NODE_TMP"
log "staging model to node-local storage"
cp -a "$MODEL_SOURCE" "$LOCAL_MODEL" || fail "model staging failed"

case "$SLURM_ARRAY_TASK_ID" in
    0) REPEATS=(0 1 2) ;;
    1) REPEATS=(3 4 5) ;;
    2) REPEATS=(6 7) ;;
    3) REPEATS=(8 9) ;;
    *) fail "unexpected array task $SLURM_ARRAY_TASK_ID" ;;
esac

groups=("put,examine" "heat,cool,clean" "puttwo")
group_names=("a" "b" "c")
demo_configs=("configs/demos/one_shot_v1.yaml"
              "configs/demos/one_shot_v1_index2.yaml"
              "configs/demos/one_shot_v1_index1.yaml")

for repeat_id in "${REPEATS[@]}"; do
    log "starting repeat ${repeat_id}"
    for i in 0 1 2; do
        for split in practice exam; do
            run_dir="$COPY/ae/runners/runs/reflexion_n10_${SLURM_ARRAY_JOB_ID}_r${repeat_id}_${group_names[$i]}_${split}"
            [ ! -e "$run_dir" ] || fail "collision: $run_dir already exists"
        done
    done

    PORT=$(( 8800 + ((SLURM_JOB_ID * 11 + repeat_id) % 600) ))
    curl -s -o /dev/null -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null && fail "port ${PORT} busy"
    vllm serve "$LOCAL_MODEL" \
        --served-model-name Qwen/Qwen3-8B --port "$PORT" --host 0.0.0.0 \
        > "$LOGDIR/vllm_repeat_${repeat_id}.log" 2>&1 &
    VLLM_PID=$!
    BASE_URL=http://localhost:${PORT}/v1
    READY=0
    for t in $(seq 1 270); do
        curl -s -o /dev/null "$BASE_URL/models" 2>/dev/null && { READY=1; break; }
        kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM died starting repeat $repeat_id"
        sleep 10
    done
    [ "$READY" -eq 1 ] || fail "vLLM not ready for repeat $repeat_id within 45 minutes"
    log "vLLM ready for repeat ${repeat_id}"

    for i in 0 1 2; do
        for split in practice exam; do
            run_name="reflexion_n10_${SLURM_ARRAY_JOB_ID}_r${repeat_id}_${group_names[$i]}_${split}"
            log "running $run_name"
            "$PY" -m ae.runners.run_alfworld \
                --baseline reflexion --split "$split" \
                --task-types "${groups[$i]}" --run-name "$run_name" \
                --max-trials 4 \
                --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
                --demo-config "${demo_configs[$i]}" \
                --termination-policy legacy_early_stop --seed 42 \
                --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
                --output-dir "$COPY/ae/runners/runs" \
                >> "$LOGDIR/repeat_${repeat_id}.log" 2>&1
            rc=$?
            [ "$rc" -eq 0 ] || fail "$run_name exited rc=$rc"
        done
    done
    kill "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""

    "$PY" - "$COPY" "$SLURM_ARRAY_JOB_ID" "$repeat_id" "$METRICDIR/repeat_${repeat_id}.json" <<'PYEOF' || fail "metric extraction failed for repeat $repeat_id"
import glob, json, os, sys
copy, job, rep, out = sys.argv[1:]
paths = glob.glob(f"{copy}/ae/runners/runs/reflexion_n10_{job}_r{rep}_*/episode_log.jsonl")
rows = [json.loads(line) for path in paths for line in open(path)]
assert len(paths) == 6, f"expected 6 subruns, got {len(paths)}"
assert len(rows) == 134, f"expected 134 episodes, got {len(rows)}"
steps = sum(len(tr.get("step_log", [])) for row in rows for tr in row.get("trial_records", []))
metric = {
    "repeat": int(rep), "episodes": len(rows),
    "successes": sum(int(r["success"]) for r in rows),
    "success_rate": sum(float(r["success"]) for r in rows) / len(rows),
    "llm_calls": sum(int(r.get("llm_calls", 0)) for r in rows),
    "steps": steps,
    "mean_llm_calls": sum(int(r.get("llm_calls", 0)) for r in rows) / len(rows),
    "mean_steps": steps / len(rows),
}
tmp = out + ".tmp"
with open(tmp, "w") as f: json.dump(metric, f, sort_keys=True)
os.replace(tmp, out)
print(json.dumps(metric, sort_keys=True))
PYEOF
    log "repeat ${repeat_id} complete: $(cat "$METRICDIR/repeat_${repeat_id}.json")"
done

"$PY" - "$METRICDIR" > "$LOGDIR/RESULT.txt" <<'PYEOF'
import glob, json, os, sys
metricdir = sys.argv[1]
rows = [json.load(open(p)) for p in sorted(glob.glob(metricdir + "/repeat_*.json"))]
mine = set(map(int, os.environ.get("REPEAT_IDS", "").split())) if os.environ.get("REPEAT_IDS") else None
print("completed metrics currently available:")
for r in sorted(rows, key=lambda x: x["repeat"]):
    print(f"repeat {r['repeat']:2d}: success={r['successes']:3d}/134 ({r['success_rate']:.1%}), "
          f"LLM calls={r['llm_calls']:5d} (mean {r['mean_llm_calls']:.2f}), "
          f"steps={r['steps']:5d} (mean {r['mean_steps']:.2f})")
if len(rows) == 10:
    successes = sum(r["successes"] for r in rows)
    episodes = sum(r["episodes"] for r in rows)
    calls = sum(r["llm_calls"] for r in rows)
    steps = sum(r["steps"] for r in rows)
    overall = (f"OVERALL n=10: success={successes}/{episodes} ({successes/episodes:.1%}), "
               f"LLM calls={calls} (mean {calls/episodes:.2f}), "
               f"steps={steps} (mean {steps/episodes:.2f})")
    print(overall)
    with open(os.path.join(os.path.dirname(metricdir), "OVERALL_RESULT.txt"), "w") as f:
        f.write(overall + "\n")
PYEOF
log "assigned repeats completed: ${REPEATS[*]}"
