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
# The shared login node's /tmp filled up (other users' files) and ALFWorld's
# fast_downward dependency extracts a .so there at import time -> crashed
# both ALFWorld ReAct runs with ENOSPC on the first attempt (2026-07-25).
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp
mkdir -p "$TMPDIR"
export ALFWORLD_DATA=/vol/gpudata/jy625-ae-data/alfworld_data
export WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (jy625@imperial.ac.uk)'

VLLM_NODE=$(cat /vol/gpudata/jy625-ae-data/logs/vllm-qwen3-8b-node.txt | awk '{print $2}')
BASE_URL="http://${VLLM_NODE}:8000/v1"
echo "Using vLLM server at $BASE_URL"

cd /homes/jy625/projects/AE3

# Fixed 2026-07-27: this script used to end with an unconditional
# "=== ALL DONE ===", even though a mid-run crash (e.g. the Reflexion x
# ALFWorld exam context-length overflow) left a combo's episode_log.jsonl
# with fewer rows than expected -- the plain sequential command list below
# has no `&&`/`set -e`, so one combo's non-zero exit never stopped the
# rest of the script, and nothing ever checked that. run_step() now
# checks BOTH the sub-process exit code AND the actual row count in that
# combo's episode_log.jsonl against the expected task count, and (for the
# ae/runners/run_alfworld.py combos specifically) greps for the
# `"incomplete": true` marker a task can now be tagged with when its
# prompt still exceeds the shared context budget after truncation + one
# retry (see alfworld_runs_ae/agents.py::PromptBudgetExceededError). A row
# count matching the expected N is NOT sufficient on its own: an
# incomplete task still writes exactly one row.
FAILED_COMBOS=()

run_step() {
  local desc="$1"; local log_path="$2"; local expected_n="$3"; shift 3
  echo "=== $desc ==="
  "$@"
  local rc=$?
  local actual_n=0
  local incomplete_n=0
  if [ -f "$log_path" ]; then
    actual_n=$(wc -l < "$log_path")
    incomplete_n=$(grep -c '"incomplete": true' "$log_path" || true)
  fi
  if [ "$rc" -ne 0 ] || [ "$actual_n" -ne "$expected_n" ] || [ "$incomplete_n" -gt 0 ]; then
    echo "  FAIL: exit_code=$rc rows=$actual_n/$expected_n incomplete=$incomplete_n log=$log_path"
    FAILED_COMBOS+=("$desc (exit=$rc, rows=$actual_n/$expected_n, incomplete=$incomplete_n)")
  else
    echo "  OK: rows=$actual_n/$expected_n"
  fi
}

run_step "ReAct: ALFWorld practice (100)" \
  "ae/runners/runs/full_react_alfworld_practice/episode_log.jsonl" 100 \
  python -m ae.runners.run_alfworld --baseline react --split practice \
  --run-name full_react_alfworld_practice --model Qwen/Qwen3-8B --base-url "$BASE_URL"

run_step "ReAct: ALFWorld exam (34)" \
  "ae/runners/runs/full_react_alfworld_exam/episode_log.jsonl" 34 \
  python -m ae.runners.run_alfworld --baseline react --split exam \
  --run-name full_react_alfworld_exam --model Qwen/Qwen3-8B --base-url "$BASE_URL"

run_step "ReAct: HotpotQA (100)" \
  "hotpotqa_runs/runs/full_react_hotpotqa/episode_log.jsonl" 100 \
  bash -c "cd hotpotqa_runs && python3 run_episode.py --data data/hotpot-qa-distractor-sample.joblib \
  --run-name full_react_hotpotqa --retrieval distractor --strategy react \
  --model Qwen/Qwen3-8B --base-url '$BASE_URL'"

run_step "ReAct: FEVER practice (70)" \
  "fever_runs/runs/full_react_fever_practice/episode_log.jsonl" 70 \
  bash -c "cd fever_runs && python3 run_episode.py --split practice --run-name full_react_fever_practice \
  --strategy react --model Qwen/Qwen3-8B --base-url '$BASE_URL'"

run_step "ReAct: FEVER exam (30)" \
  "fever_runs/runs/full_react_fever_exam/episode_log.jsonl" 30 \
  bash -c "cd fever_runs && python3 run_episode.py --split exam --run-name full_react_fever_exam \
  --strategy react --model Qwen/Qwen3-8B --base-url '$BASE_URL'"

run_step "Reflexion: ALFWorld practice (100)" \
  "ae/runners/runs/full_reflexion_alfworld_practice/episode_log.jsonl" 100 \
  python -m ae.runners.run_alfworld --baseline reflexion --split practice \
  --run-name full_reflexion_alfworld_practice --max-trials 4 \
  --model Qwen/Qwen3-8B --base-url "$BASE_URL"

run_step "Reflexion: ALFWorld exam (34)" \
  "ae/runners/runs/full_reflexion_alfworld_exam/episode_log.jsonl" 34 \
  python -m ae.runners.run_alfworld --baseline reflexion --split exam \
  --run-name full_reflexion_alfworld_exam --max-trials 4 \
  --model Qwen/Qwen3-8B --base-url "$BASE_URL"

run_step "Reflexion: HotpotQA (100)" \
  "hotpotqa_runs/runs/full_reflexion_hotpotqa/episode_log.jsonl" 100 \
  bash -c "cd hotpotqa_runs && python3 run_episode.py --data data/hotpot-qa-distractor-sample.joblib \
  --run-name full_reflexion_hotpotqa --retrieval distractor --strategy reflexion --max-trials 4 \
  --model Qwen/Qwen3-8B --base-url '$BASE_URL'"

run_step "Reflexion: FEVER practice (70)" \
  "fever_runs/runs/full_reflexion_fever_practice/episode_log.jsonl" 70 \
  bash -c "cd fever_runs && python3 run_episode.py --split practice --run-name full_reflexion_fever_practice \
  --strategy reflexion --max-trials 4 --model Qwen/Qwen3-8B --base-url '$BASE_URL'"

run_step "Reflexion: FEVER exam (30)" \
  "fever_runs/runs/full_reflexion_fever_exam/episode_log.jsonl" 30 \
  bash -c "cd fever_runs && python3 run_episode.py --split exam --run-name full_reflexion_fever_exam \
  --strategy reflexion --max-trials 4 --model Qwen/Qwen3-8B --base-url '$BASE_URL'"

if [ ${#FAILED_COMBOS[@]} -eq 0 ]; then
  echo "=== ALL DONE ==="
else
  echo "=== INCOMPLETE: ${#FAILED_COMBOS[@]} combo(s) failed ==="
  for c in "${FAILED_COMBOS[@]}"; do
    echo "  - $c"
  done
  exit 1
fi
