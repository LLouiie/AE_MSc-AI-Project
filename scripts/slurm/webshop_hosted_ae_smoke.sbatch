#!/bin/bash
#SBATCH --partition=t4
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=ws_hosted_smoke
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/ws_hosted_smoke-%j.out
#SBATCH --error=/vol/gpudata/jy625-ae-data/logs/ws_hosted_smoke-%j.err

set -euo pipefail
. /homes/jy625/.config/ae/openai.env

REPO=/homes/jy625/projects/AE3
WS_ROOT=/vol/gpudata/jy625-ae-data/webshop_official
WS_PY=/vol/gpudata/jy625-ae-data/miniconda3/envs/webshop/bin/python
AE_PY=/vol/gpudata/jy625-ae-data/ae3-venv/bin/python3
MODEL=${1:-${HOSTED_MODEL:?HOSTED_MODEL or argument 1 is required}}
LABEL=${2:-${MODEL_LABEL:?MODEL_LABEL or argument 2 is required}}
NUM_ENVS=${3:-${NUM_ENVS:-3}}
OUT=/vol/gpudata/jy625-ae-data/results/webshop_hosted_smoke/${LABEL}_${SLURM_JOB_ID}
mkdir -p "$OUT"

cleanup() {
  rc=$?
  set +e
  [ -n "${WS_PID:-}" ] && kill "$WS_PID" 2>/dev/null
  if [ "$rc" -eq 0 ]; then
    title="WebShop ${LABEL} AE smoke complete"
  else
    title="WebShop ${LABEL} AE smoke failed"
  fi
  curl -s -m 20 -H "Title: $title" -d "Job ${SLURM_JOB_ID}, exit ${rc}, results ${OUT}" https://ntfy.sh/AE_new_alert >/dev/null 2>&1 || true
  exit "$rc"
}
trap cleanup EXIT

export JAVA_HOME=/vol/gpudata/jy625-ae-data/miniconda3/envs/webshop/lib/jvm
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
cd "$WS_ROOT"
PYTHONPATH="$WS_ROOT" "$WS_PY" "$REPO/webshop_runs/run_frozen_webshop.py" \
  --log --attrs --goal-seed 233 >"$OUT/webshop.log" 2>&1 &
WS_PID=$!
for _ in $(seq 1 180); do
  curl -fsS http://127.0.0.1:3000/fixed_0 >/dev/null 2>&1 && READY=1 && break
  sleep 2
done
[ "${READY:-0}" = 1 ]

export WEBSHOP_URL=http://127.0.0.1:3000
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL="$MODEL"
cd "$REPO/webshop_runs"
for mode in full no_trigger random_trigger verify_only reflect_only replan_only; do
  "$AE_PY" run_ae.py --num-envs "$NUM_ENVS" \
    --ae-config "$REPO/configs/controllers/ae_full_final_nopir_v2_5.yaml" \
    --shopping-guidance on --ae-ablation "$mode" \
    --ae-random-seed 42 --ae-random-trigger-probability 0.5 \
    --output "$OUT/${mode}.json"
done

"$AE_PY" - "$OUT" "$NUM_ENVS" <<'PY'
import glob, json, os, sys
root, expected = sys.argv[1], int(sys.argv[2])
for path in sorted(glob.glob(os.path.join(root, "*.json"))):
    data = json.load(open(path))
    assert len(data["episodes"]) == expected
    print(os.path.basename(path), json.dumps(data["overall"], sort_keys=True))
PY
