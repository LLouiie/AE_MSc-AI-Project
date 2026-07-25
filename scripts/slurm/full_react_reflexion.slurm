#!/bin/bash
#SBATCH --job-name=full-react-reflexion
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/vol/gpudata/jy625-ae-data/logs/full-react-reflexion-%j.out

# CPU-only driver job: just makes HTTP calls to the vLLM server, no GPU needed here.
source /vol/gpudata/jy625-ae-data/ae3-venv/bin/activate
export ALFWORLD_DATA=/vol/gpudata/jy625-ae-data/alfworld_data
export WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (jy625@imperial.ac.uk)'

VLLM_NODE=$(cat /vol/gpudata/jy625-ae-data/logs/vllm-qwen3-8b-node.txt | awk '{print $2}')
BASE_URL="http://${VLLM_NODE}:8000/v1"
echo "Using vLLM server at $BASE_URL"

cd /homes/jy625/projects/AE3

echo "=== ReAct: ALFWorld practice (100) ==="
python -m ae.runners.run_alfworld --baseline react --split practice \
  --run-name full_react_alfworld_practice --model Qwen/Qwen3-8B --base-url "$BASE_URL"

echo "=== ReAct: ALFWorld exam (34) ==="
python -m ae.runners.run_alfworld --baseline react --split exam \
  --run-name full_react_alfworld_exam --model Qwen/Qwen3-8B --base-url "$BASE_URL"

echo "=== ReAct: HotpotQA (100) ==="
cd hotpotqa_runs
python3 run_episode.py --data data/hotpot-qa-distractor-sample.joblib \
  --run-name full_react_hotpotqa --retrieval distractor --strategy react \
  --model Qwen/Qwen3-8B --base-url "$BASE_URL"
cd ..

echo "=== ReAct: FEVER practice (70) ==="
cd fever_runs
python3 run_episode.py --split practice --run-name full_react_fever_practice \
  --strategy react --model Qwen/Qwen3-8B --base-url "$BASE_URL"

echo "=== ReAct: FEVER exam (30) ==="
python3 run_episode.py --split exam --run-name full_react_fever_exam \
  --strategy react --model Qwen/Qwen3-8B --base-url "$BASE_URL"
cd ..

echo "=== Reflexion: ALFWorld practice (100) ==="
python -m ae.runners.run_alfworld --baseline reflexion --split practice \
  --run-name full_reflexion_alfworld_practice --max-trials 4 \
  --model Qwen/Qwen3-8B --base-url "$BASE_URL"

echo "=== Reflexion: ALFWorld exam (34) ==="
python -m ae.runners.run_alfworld --baseline reflexion --split exam \
  --run-name full_reflexion_alfworld_exam --max-trials 4 \
  --model Qwen/Qwen3-8B --base-url "$BASE_URL"

echo "=== Reflexion: HotpotQA (100) ==="
cd hotpotqa_runs
python3 run_episode.py --data data/hotpot-qa-distractor-sample.joblib \
  --run-name full_reflexion_hotpotqa --retrieval distractor --strategy reflexion --max-trials 4 \
  --model Qwen/Qwen3-8B --base-url "$BASE_URL"
cd ..

echo "=== Reflexion: FEVER practice (70) ==="
cd fever_runs
python3 run_episode.py --split practice --run-name full_reflexion_fever_practice \
  --strategy reflexion --max-trials 4 --model Qwen/Qwen3-8B --base-url "$BASE_URL"

echo "=== Reflexion: FEVER exam (30) ==="
python3 run_episode.py --split exam --run-name full_reflexion_fever_exam \
  --strategy reflexion --max-trials 4 --model Qwen/Qwen3-8B --base-url "$BASE_URL"
cd ..

echo "=== ALL DONE ==="
