#!/bin/bash
#SBATCH --job-name=reflexgrad-seg
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/reflexgrad-seg-%j.out

# Usage: sbatch --export=OFFSET=<n>,NUM_ENVS=<n>,RUN_NAME=<name> reflexgrad_segment.slurm
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export ALFWORLD_DATA=/vol/gpudata/jy625-ae-data/alfworld_data

VLLM_NODE=$(cat /vol/gpudata/jy625-ae-data/logs/vllm-qwen3-8b-node.txt | awk '{print $2}')
export VLLM_BASE_URL="http://${VLLM_NODE}:8000/v1"
export VLLM_MODEL=Qwen/Qwen3-8B
export VLLM_FAST_MODEL=Qwen/Qwen3-8B
export ALFWORLD_ENV_OFFSET=${OFFSET:-0}
echo "Using vLLM server at $VLLM_BASE_URL, offset=$ALFWORLD_ENV_OFFSET, num_envs=${NUM_ENVS:-45}"

cd /homes/jy625/projects/AE3/external/reflexgrad
python main.py --model_provider vllm --num_trials 1 --num_envs ${NUM_ENVS:-45} \
  --run_name ${RUN_NAME:-reflexgrad_seg} --env_type alfworld --max_steps 15

echo "=== ReflexGrad segment DONE ==="
