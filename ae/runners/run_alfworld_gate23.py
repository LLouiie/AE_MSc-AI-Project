"""reflect-first-encounter-entry Gate: runs a FIXED set of abs_idx (from the
combined practice[0:100]+exam[100:134] 134-task ordering, same convention
used throughout this project) through ae_full, for the paired Control vs
Candidate comparison. Mirrors run_alfworld.py's dispatch/logging exactly,
just filtered to a specific abs_idx list instead of a full split.

Usage:
    python -m ae.runners.run_alfworld_gate23 \
        --run-name ae_gate_reflect_entry_control \
        --ae-config configs/controllers/ae_gate_reflect_entry_control.yaml \
        --model Qwen/Qwen3-8B --base-url http://gpuvm35:8000/v1
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "alfworld_runs_ae"))

from environment import (  # noqa: E402
    get_practice_tasks, get_exam_tasks, get_task_type,
    make_single_task_env, resolve_gamefile,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ae.llm_client import AnyOpenAILLM, call_counter  # noqa: E402
from ae.logging_utils import snapshot_config, load_done_ids, JsonlLogger  # noqa: E402
from ae.baselines import ae_full as ae_full_baseline  # noqa: E402
from ae.controllers.config import load_config as load_ae_config  # noqa: E402
from agents import ACT_MAX_TOKENS, ACT_STOP, PromptBudgetExceededError  # noqa: E402

# The 23 abs_idx from the reflect-first-encounter-entry audit
# (ae_reflect_replan_upgrade_chain_audit.md, corrected via abs_idx not env_name).
GATE_ABS_IDX = [0, 8, 14, 17, 32, 37, 38, 47, 49, 55, 60, 62, 63, 67, 68, 81, 82, 98, 100, 106, 120, 125, 133]


def build_llms(model: str, base_url: str):
    act_llm = AnyOpenAILLM(temperature=0, max_tokens=ACT_MAX_TOKENS, model_name=model,
                            model_kwargs={"stop": ACT_STOP},
                            openai_api_key="EMPTY", openai_api_base=base_url)
    return act_llm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--ae-config", required=True)
    p.add_argument("--termination-policy", default="legacy_early_stop",
                    choices=["fixed_horizon", "legacy_early_stop"])
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "runs"))
    args = p.parse_args()

    run_dir = os.path.join(args.output_dir, args.run_name)
    log_path = os.path.join(run_dir, "episode_log.jsonl")
    snapshot_config(run_dir, vars(args), REPO_ROOT)

    practice = get_practice_tasks()
    exam = get_exam_tasks()
    assert len(practice) == 100 and len(exam) == 34
    combined = {}
    for i, t in enumerate(practice):
        combined[i] = t
    for i, t in enumerate(exam):
        combined[100 + i] = t

    tasks = [(idx, combined[idx]) for idx in GATE_ABS_IDX]

    done = load_done_ids(log_path, id_field="abs_idx")
    if done:
        print(f"checkpoint: {len(done)} done, resuming")

    act_llm = build_llms(args.model, args.base_url)
    logger = JsonlLogger(log_path)
    ae_config = load_ae_config(args.ae_config)
    print(f"ae_config loaded: warmup={ae_config.warmup_steps} grace={ae_config.recovery_grace_steps} "
          f"patch={ae_config.patch_duration_steps} max_interventions={ae_config.max_interventions} "
          f"cooldown={ae_config.cooldown_steps} pir_enabled={ae_config.post_intervention_repeat_enabled} "
          f"reflect_first_encounter_repeat_enabled={ae_config.reflect_first_encounter_repeat_enabled}")

    for qi, (abs_idx, task) in enumerate(tasks, 1):
        if abs_idx in done:
            continue
        env_name = task.get("env_name") or task["gamefile"].split("/")[-3]
        goal = task["goal"]
        task_type = get_task_type(env_name)

        env = make_single_task_env(resolve_gamefile(task["gamefile"]))
        c0 = call_counter()
        t0 = time.time()

        try:
            result = ae_full_baseline.run_episode(
                env, goal, task_type, act_llm, ae_config,
                termination_policy=args.termination_policy, task_id=env_name,
            )
        except PromptBudgetExceededError as e:
            print(f"[{qi}/{len(tasks)}] abs_idx={abs_idx} env={env_name} INCOMPLETE: {e}")
            result = {
                "baseline": "ae_full", "success": 0, "incomplete": True,
                "incomplete_reason": str(e),
                "tokens_before_truncation": e.tokens_before,
                "tokens_after_truncation": e.tokens_after,
            }
        finally:
            env.close()

        record = {
            "abs_idx": abs_idx, "env_name": env_name, "goal": goal,
            "llm_calls": call_counter() - c0,
            "wall_s": round(time.time() - t0, 1),
            **result,
        }
        logger.write(record)
        if not result.get("incomplete"):
            print(f"[{qi}/{len(tasks)}] abs_idx={abs_idx} success={record['success']} "
                  f"calls={record['llm_calls']} wall_s={record['wall_s']}")

    logger.close()

    rows = [json.loads(l) for l in open(log_path)]
    n = len(rows)
    print("\n===== GATE23 DONE =====")
    print(f"N={n}  success_rate={sum(r['success'] for r in rows)/n:.3f}  "
          f"avg_calls={sum(r['llm_calls'] for r in rows)/n:.1f}")


if __name__ == "__main__":
    main()
