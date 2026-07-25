#!/bin/bash
#SBATCH --job-name=full-adapt
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/full-adapt-%j.out

source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export ALFWORLD_DATA=/vol/gpudata/jy625-ae-data/alfworld_data

VLLM_NODE=$(cat /vol/gpudata/jy625-ae-data/logs/vllm-qwen3-8b-node.txt | awk '{print $2}')
export OPENAI_BASE_URL="http://${VLLM_NODE}:8000/v1"
echo "Using vLLM server at $OPENAI_BASE_URL"

cd /homes/jy625/projects/AE3/external/ADaPT
mkdir -p results/comparison/Qwen/Qwen3-8B

echo "=== ADaPT: ALFWorld full valid_unseen (134, eval-all) ==="
python run_alfworld.py \
  --LM Qwen/Qwen3-8B --exec-LM Qwen/Qwen3-8B --force-same-LM True \
  --eval-all True --max-depth 3 --fname full_adapt_alfworld

echo "=== ADaPT DONE ==="
