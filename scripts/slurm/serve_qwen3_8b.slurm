#!/bin/bash
#SBATCH --job-name=vllm-qwen3-8b
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/vllm-qwen3-8b-%j.out

mkdir -p /vol/gpudata/jy625-ae-data/logs
echo "Node: $(hostname)" | tee /vol/gpudata/jy625-ae-data/logs/vllm-qwen3-8b-node.txt
nvidia-smi -L

source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export HF_HOME=/vol/gpudata/jy625-ae-data/hf_cache
# No nvcc/CUDA toolkit on this node, so flashinfer can't JIT its sampling
# kernel; fall back to vLLM's non-JIT sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /vol/gpudata/jy625-ae-data/models/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --port 8000 --host 0.0.0.0
