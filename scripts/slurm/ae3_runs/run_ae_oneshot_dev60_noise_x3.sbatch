#!/bin/bash
#SBATCH --job-name=ae_1shot_noise
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ae_1shot_noise-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ae_1shot_noise-%j.err

# ============================================================================
# task #82: run-to-run noise quantification for the one-shot AE baseline
# config, on the frozen 60-task dev set (disjoint from the 134-task
# practice+exam split -- never touches practice/exam).
#
# Motivation: the demo-selection ablation's frozen-mapping validation
# (task #77) found heat's success count swing by -6 (14/23 -> 8/23) between
# two runs with IDENTICAL demo config (both _0) -- pure run-to-run
# inference noise (same known effect as this project's prior grace5
# 44/35/42 three-run spread, documented in
# audit_reports/ae_full_sustained_v1_134_report.md). Before any further
# one-shot AE tuning (controller hyperparameter resweep, VERIFY/REFLECT
# prompt-v2, etc.), we need to know the noise floor UNDER THIS EXACT
# one-shot setup specifically -- the existing dev60 repeat data (task #65,
# 8 repeat runs of baseline/grace5/patch2/cooldown5) is TWO-SHOT and used a
# different (though closely related) controller sha
# (35249c6949ab5e8fcd2d0292f34c4c68be78c78c57f15a92a088177dbd6f2f3f) than
# this one-shot line's old-REPLAN-unconditional controller
# (6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484,
# confirmed byte-identical to replan_v2_exp_oldcontroller_seed43) --
# NOT directly reusable as a noise reference for one-shot.
#
# Config: EXACT same as the official one-shot 80/134 baseline run
# (ae_full_nopir_v2_5_oldctrl_replanpromptv2_oneshot0_seed42_a40_*) --
# same controller (ae_full_final_nopir_v2_5.yaml, old REPLAN-unconditional),
# same REPLAN prompt-v2, same demo config (one_shot_v1.yaml, uniform _0),
# same seed=42, same termination_policy=legacy_early_stop -- run 3 times
# back-to-back against the same vLLM server on the 60-task dev60 manifest.
# temperature=0 throughout; any spread across the 3 runs is pure inference-
# session non-determinism, not a config difference.
#
# New code this run depends on (task #81): ae/runners/run_alfworld_devset.py
# ported from the live repo + --demo-config CLI arg added (pure additive
# threading, same pattern as run_alfworld.py's existing --demo-config --
# no agents.py/controller/demo_config.py changes). dev_set/ manifest copied
# byte-identical from the live repo (sha256 verified below).
# ============================================================================

set -uo pipefail

COPY=/vol/gpudata/jy625-ae-data/scratch_ae3/one_shot_oldcontroller_promptv2_seed42
PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
LOGDIR=/vol/gpudata/jy625-ae-data/logs/ae_1shot_noise_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
mkdir -p "$TMPDIR"

RUN1=ae_oneshot_dev60_noise_rep1
RUN2=ae_oneshot_dev60_noise_rep2
RUN3=ae_oneshot_dev60_noise_rep3

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/batch.log"; }
fail() {
    echo "[$(date -Is)] PREFLIGHT FAILURE: $*" | tee -a "$LOGDIR/batch.log" "$LOGDIR/preflight_failure.log"
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    exit 1
}
trap 'log "job ending, exit code $?"; [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null' EXIT

log "job started on $(hostname), SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L | tee "$LOGDIR/gpu_info.log"

# ---------------------------------------------------------------------------
# Pre-flight 1: none of the 3 result dirs already exist.
# ---------------------------------------------------------------------------
for d in "$RUN1" "$RUN2" "$RUN3"; do
    [ -e "$COPY/ae/runners/runs/$d" ] && fail "result directory already exists, refusing to overwrite: $COPY/ae/runners/runs/$d"
done
log "check 1 OK: 3 result dirs free"

# ---------------------------------------------------------------------------
# Pre-flight 2: sha256 + confirm controller/renderer/demo-config byte-
# identical to the official one-shot 80/134 baseline run, and dev60
# manifest byte-identical to the live repo's frozen copy.
# ---------------------------------------------------------------------------
{
    echo "=== COPY ==="
    sha256sum "$COPY"/alfworld_runs_ae/agents.py \
              "$COPY"/alfworld_runs_ae/demo_config.py \
              "$COPY"/ae/controllers/intervention_renderer.py \
              "$COPY"/ae/controllers/stateful_controller.py \
              "$COPY"/ae/baselines/ae_full.py \
              "$COPY"/ae/runners/run_alfworld_devset.py \
              "$COPY"/configs/controllers/ae_full_final_nopir_v2_5.yaml \
              "$COPY"/configs/demos/one_shot_v1.yaml \
              "$COPY"/dev_set/dev60_valid_seen_manifest.json
} > "$LOGDIR/sha256.log" 2>&1
grep -q "c4824c0606a87eedc32aa96564c9ddcedafc1fd11a7b212a3131382fe43367ba  $COPY/ae/controllers/intervention_renderer.py" "$LOGDIR/sha256.log" \
    || fail "intervention_renderer.py sha mismatch -- see $LOGDIR/sha256.log"
grep -q "6fefd3324e6d2578b4a835a52c598ed878b205427c7bee1c0f9db6decc249484  $COPY/ae/controllers/stateful_controller.py" "$LOGDIR/sha256.log" \
    || fail "stateful_controller.py sha mismatch -- see $LOGDIR/sha256.log"
grep -q "8e96ddfaaee64c65c16df42b542ef71d3681922418f7f9b6c5b28527fff635eb  $COPY/dev_set/dev60_valid_seen_manifest.json" "$LOGDIR/sha256.log" \
    || fail "dev60 manifest sha mismatch vs live-repo frozen copy -- see $LOGDIR/sha256.log"
log "check 2 OK: sha256 recorded, controller/renderer/manifest confirmed byte-identical (see $LOGDIR/sha256.log)"

# ---------------------------------------------------------------------------
# Pre-flight 3: regression suite.
# ---------------------------------------------------------------------------
cd "$COPY" || fail "cannot cd into $COPY"
for f in alfworld_runs_ae/tests/test_demo_config.py \
         alfworld_runs_ae/tests/test_alfworld_action_normalize.py \
         alfworld_runs_ae/tests/test_output_parser.py \
         alfworld_runs_ae/tests/test_prompt_budget.py \
         alfworld_runs_ae/tests/test_put_move_compat_integration.py \
         alfworld_runs_ae/tests/test_environment_anchor.py; do
    out=$("$PY" "$f" 2>&1)
    { echo "=== $f ==="; echo "$out"; } >> "$LOGDIR/regression.log"
    line=$(echo "$out" | tail -1)
    echo "$line" | grep -qE '^[0-9]+ passed, 0 failed$' \
        || fail "regression: $f did not fully pass ($line) -- see $LOGDIR/regression.log"
done
log "check 3 OK: regression suite passed (see $LOGDIR/regression.log)"

# ---------------------------------------------------------------------------
# Pre-flight 4: manifest has exactly 60 tasks with valid, hash-verified
# gamefiles.
# ---------------------------------------------------------------------------
MANIFEST_CHECK=$("$PY" -c "
import json, hashlib, os
m = json.load(open('dev_set/dev60_valid_seen_manifest.json'))
tasks = m['manifest']
assert len(tasks) == 60, f'expected 60, got {len(tasks)}'
bad = []
for t in tasks:
    gf = t['gamefile']
    if not os.path.exists(gf):
        bad.append((t['dev_idx'], 'missing'))
        continue
    h = hashlib.sha256(open(gf,'rb').read()).hexdigest()
    if h != t['gamefile_sha256']:
        bad.append((t['dev_idx'], 'sha_mismatch'))
if bad:
    print('BAD', bad)
else:
    print('OK 60/60 gamefiles present and sha-verified')
" 2>&1)
echo "$MANIFEST_CHECK" >> "$LOGDIR/batch.log"
echo "$MANIFEST_CHECK" | grep -q "^OK " || fail "dev60 manifest gamefile check failed: $MANIFEST_CHECK"
log "check 4 OK: $MANIFEST_CHECK"

log "ALL PRE-FLIGHT CHECKS PASSED. Starting vLLM server."

source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /vol/gpudata/jy625-ae-data/models/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --port 8000 --host 0.0.0.0 \
    > "$LOGDIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
log "vLLM server starting, pid=$VLLM_PID, log=$LOGDIR/vllm_server.log"

BASE_URL="http://localhost:8000/v1"
READY=0
for i in $(seq 1 60); do
    if curl -s -o /dev/null -w "" "$BASE_URL/models" 2>/dev/null; then
        READY=1
        break
    fi
    kill -0 "$VLLM_PID" 2>/dev/null || fail "vLLM server process died during startup (pid=$VLLM_PID) -- see $LOGDIR/vllm_server.log"
    sleep 10
done
[ "$READY" -eq 1 ] || fail "vLLM server did not become ready within 10 minutes -- see $LOGDIR/vllm_server.log"
log "vLLM server ready at $BASE_URL"

log "=== starting 3 sequential repeat runs (60 dev tasks each, one-shot _0) ==="
for RUN in "$RUN1" "$RUN2" "$RUN3"; do
    log "--- starting $RUN ---"
    "$PY" -m ae.runners.run_alfworld_devset \
        --run-name "$RUN" \
        --ae-config configs/controllers/ae_full_final_nopir_v2_5.yaml \
        --demo-config configs/demos/one_shot_v1.yaml \
        --termination-policy legacy_early_stop \
        --model Qwen/Qwen3-8B --base-url "$BASE_URL" \
        --output-dir "$COPY/ae/runners/runs" \
        > "$LOGDIR/${RUN}.log" 2>&1
    rc=$?
    log "--- finished $RUN rc=$rc ---"
    [ "$rc" -eq 0 ] || fail "$RUN exited non-zero ($rc) -- see $LOGDIR/${RUN}.log"
done

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
VLLM_PID=""
log "vLLM server stopped."
log "JOB COMPLETE."
