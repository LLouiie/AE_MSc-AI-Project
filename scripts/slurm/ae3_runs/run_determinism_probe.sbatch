#!/bin/bash
#SBATCH --job-name=det_probe
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/det_probe-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/det_probe-%j.err

# ============================================================================
# DETERMINISM PROBE
#
# Question: why do identical configurations, identical seed, temperature=0
# give 91/100/103 on 134 tasks, with 21/134 (15.7%) of tasks flipping
# success<->failure across repeats?
#
# Established facts (checked, not assumed):
#   - --seed 42 on every run ever submitted; seed was never varied
#   - the seed is inert: run_alfworld.py:95 does random.seed(args.seed) and
#     `random.` appears NOWHERE else in ae/ -- it seeds an RNG nobody draws from
#   - both LLMs are temperature=0 (run_alfworld.py:49,52), i.e. greedy argmax,
#     so there is no sampling RNG for a seed to control either way
#   - therefore the run-to-run variance is NOT seed variance and cannot be
#     removed by changing the seed
#
# Remaining candidate causes, all numerical rather than stochastic:
#   (1) CONCURRENCY. The batch scripts run practice and exam simultaneously
#       against one vLLM server ("run_one practice & run_one exam & wait").
#       Two live request streams mean batch composition varies with timing,
#       which varies the float reduction order, which can flip an argmax when
#       two logits are near-tied.
#   (2) PREFIX CACHING (enable_prefix_caching=True in the server config).
#       A ReAct-style agent resends a growing prompt each step; whether a
#       prefix block is recomputed or read from cache is a different numerical
#       path for the same token, and cache/eviction state differs per run.
#   (3) CUDA GRAPHS (enforce_eager=False). Kernel/split selection is bucketed
#       by padded shape, so different lengths take different reduction orders.
#
# Design: same 40 tasks, run TWICE per arm, strictly serially (never two
# runners at once), and diff the results.
#   ARM A (control):       default server flags   -> isolates cause (1)
#   ARM B (deterministic): eager + no prefix cache + max-num-seqs 1
# If ARM A already reproduces, the fix is free: serialise practice/exam and
# pay nothing. If only ARM B reproduces, determinism costs ~50% throughput.
# If NEITHER reproduces, the nondeterminism is in the kernels themselves and
# the only remaining option is to stop chasing effects smaller than the noise.
#
# This probe changes NO project code. It only varies vLLM server flags and
# the runner's concurrency, and it writes to its own run dirs.
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/admissible_reflect_workdir
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/det_probe_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

NTASK=40

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() {
    echo "[$(date -Is)] FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/failure.log"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}
notify() { curl -s -m 20 -d "$1" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true; }
trap 'rc=$?; log "job ending, exit code $rc"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; notify "determinism probe (job ${SLURM_JOB_ID}) ended, exit=$rc, logdir=$LOGDIR"' EXIT

log "job started on $(hostname), SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

for arm in ctrl det; do
  for r in 1 2; do
    d="det_probe_${arm}_rep${r}"
    [ -e "$COPY/ae/runners/runs/$d" ] && fail "result dir already exists: $d"
  done
done
log "check 1 OK: 4 result dirs free"

# Code must be byte-identical to the runs whose variance we are explaining.
sha256sum "$COPY"/ae/controllers/intervention_renderer.py \
          "$COPY"/ae/controllers/stateful_controller.py \
          "$COPY"/ae/runners/run_alfworld.py > "$LOGDIR/sha256.log" 2>&1
grep -q "d171ca687e35deefa1e25729dd172bd797df2eab8ca4b4a1f71eaf551b34406f  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch"
grep -q "3127c243b5abcbd9a5115c3d01bd19098dc62527ece63ad099d207a25207164c  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch"
log "check 2 OK: controller code matches the runs under investigation"

cd "$COPY" || fail "cannot cd into $COPY"
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

start_server() {
    local arm=$1; shift
    # distinct port per arm: the two servers are sequential, but a slow
    # shutdown must never let arm B attach to arm A's socket
    local off=0
    [ "$arm" = "det" ] && off=1
    PORT=$(( 8000 + (SLURM_JOB_ID % 900) + off ))
    if curl -s -o /dev/null -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null; then
        fail "port ${PORT} already serving -- refusing to attach to another job's vLLM"
    fi
    log "starting vLLM for arm=$arm on port $PORT with extra flags: $*"
    vllm serve /vol/gpudata/jy625-ae-data/models/Qwen3-8B \
        --served-model-name Qwen/Qwen3-8B --port "${PORT}" --host 0.0.0.0 \
        "$@" > "$LOGDIR/vllm_${arm}.log" 2>&1 &
    VLLM_PID=$!
    BASE_URL="http://localhost:${PORT}/v1"
    READY=0
    for i in $(seq 1 60); do
        curl -s -o /dev/null -w "" "$BASE_URL/models" 2>/dev/null && { READY=1; break; }
        kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM (arm=$arm) died during startup -- check $LOGDIR/vllm_${arm}.log for a bad flag name"
        sleep 10
    done
    [ "$READY" -eq 1 ] || fail "vLLM (arm=$arm) not ready within 10 minutes"
    kill -0 "$VLLM_PID" 2>/dev/null || fail "port answered but OUR vLLM is dead"
    # record the flags the server actually adopted
    grep -aoE "enable_prefix_caching=[A-Za-z]+|enforce_eager=[A-Za-z]+|max_num_seqs=[0-9]+" \
        "$LOGDIR/vllm_${arm}.log" | sort -u | tee -a "$LOGDIR/server_config_${arm}.log"
    log "vLLM ready for arm=$arm (pid=$VLLM_PID)"
}

stop_server() {
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    for i in $(seq 1 30); do kill -0 "${VLLM_PID:-0}" 2>/dev/null || break; sleep 2; done
    VLLM_PID=""
    sleep 10
    log "vLLM stopped"
}

# NOTE: strictly serial -- never two runners at once. That is the point.
run_one() {
    local run_name=$1
    "$PY" -m ae.runners.run_alfworld \
        --baseline ae_full --split practice \
        --run-name "$run_name" \
        --limit "$NTASK" \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
        --demo-config configs/demos/one_shot_v1.yaml \
        --termination-policy legacy_early_stop \
        --seed 42 \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$COPY/ae/runners/runs" \
        > "$LOGDIR/${run_name}.log" 2>&1
    local rc=$?
    [ "$rc" -eq 0 ] || fail "$run_name failed (rc=$rc)"
    log "$run_name done"
}

# ---------------- ARM A: control, default server flags ----------------
start_server ctrl
run_one det_probe_ctrl_rep1
run_one det_probe_ctrl_rep2
stop_server

# ---------------- ARM B: deterministic server flags -------------------
start_server det --enforce-eager --no-enable-prefix-caching --max-num-seqs 1
run_one det_probe_det_rep1
run_one det_probe_det_rep2
stop_server

# ---------------- comparison ----------------
"$PY" - << 'PYEOF' 2>&1 | tee "$LOGDIR/RESULT.txt"
import json, hashlib, os
R="/vol/gpudata/jy625-ae-data/scratch_ae3/admissible_reflect_workdir/ae/runners/runs"
def load(name):
    d={}
    for l in open(f"{R}/{name}/episode_log.jsonl"):
        x=json.loads(l)
        d[(x['env_name'],x['goal'])]=(x['success'],
            hashlib.sha256(json.dumps(x['trajectory'],sort_keys=True).encode()).hexdigest()[:16])
    return d
print("="*70)
print("DETERMINISM PROBE RESULT")
print("="*70)
for arm,label in (("ctrl","ARM A  default server flags, serial"),
                  ("det","ARM B  eager + no prefix cache + max-num-seqs 1, serial")):
    try:
        a=load(f"det_probe_{arm}_rep1"); b=load(f"det_probe_{arm}_rep2")
    except FileNotFoundError as e:
        print(f"\n{label}\n  MISSING: {e}"); continue
    keys=set(a)&set(b)
    sflip=[k for k in keys if a[k][0]!=b[k][0]]
    tdiff=[k for k in keys if a[k][1]!=b[k][1]]
    print(f"\n{label}")
    print(f"  tasks compared         : {len(keys)}")
    print(f"  score rep1 / rep2      : {sum(v[0] for v in a.values())} / {sum(v[0] for v in b.values())}")
    print(f"  success flipped        : {len(sflip)}")
    print(f"  trajectory differed    : {len(tdiff)}  <-- 0 means BITWISE REPRODUCIBLE")
    print(f"  VERDICT                : {'REPRODUCIBLE' if not tdiff else 'NONDETERMINISTIC'}")
print("\nReference: the 3 full-134 repeats of the max-score config flipped")
print("21/134 tasks (15.7%) with scores 91/100/103.")
print("="*70)
PYEOF

log "JOB COMPLETE"
