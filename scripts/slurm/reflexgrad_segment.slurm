#!/bin/bash
#SBATCH --job-name=reflexgrad-seg
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/reflexgrad-seg-%j.out

# Usage: sbatch reflexgrad_segment.slurm <OFFSET> <NUM_ENVS> <RUN_NAME>
#
# Positional args, NOT --export=VAR=val: confirmed empirically (2026-07-25)
# that on this cluster, `sbatch --export=...,KEY=val ...` (with or without
# ALL) always fails immediately with JobState=PENDING
# Reason=user_env_retrieval_failed_requeued_held, regardless of script
# content -- reproduced on a trivial one-line echo script, while the exact
# same script submitted with plain positional args or no --export at all
# completes normally. This is a scheduler-side bug specific to custom
# --export key=value pairs on this SLURM deployment, not anything about
# this script or the reflexgrad job itself.
OFFSET="${1:-0}"
NUM_ENVS="${2:-45}"
RUN_NAME="${3:-reflexgrad_seg}"

source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp
mkdir -p "$TMPDIR"
export ALFWORLD_DATA=/vol/gpudata/jy625-ae-data/alfworld_data

VLLM_NODE=$(cat /vol/gpudata/jy625-ae-data/logs/vllm-qwen3-8b-node.txt | awk '{print $2}')
export VLLM_BASE_URL="http://${VLLM_NODE}:8000/v1"
export VLLM_MODEL=Qwen/Qwen3-8B
export VLLM_FAST_MODEL=Qwen/Qwen3-8B
export ALFWORLD_ENV_OFFSET="$OFFSET"
echo "Using vLLM server at $VLLM_BASE_URL, offset=$ALFWORLD_ENV_OFFSET, num_envs=$NUM_ENVS"

cd /homes/jy625/projects/AE3/external/reflexgrad
python main.py --model_provider vllm --num_trials 1 --num_envs "$NUM_ENVS" \
  --run_name "$RUN_NAME" --env_type alfworld --max_steps 15

echo "=== ReflexGrad segment DONE ==="
