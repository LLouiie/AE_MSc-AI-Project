"""dev60 tuning-round entry point. Mirrors run_alfworld.py's ae_full path
exactly (same baseline dispatch, same JSONL logging contract, same
checkpoint/resume behavior) but sources tasks from the frozen dev60
manifest (dev_set/dev60_valid_seen_manifest.json) instead of
get_practice_tasks()/get_exam_tasks(), so the parameter-sensitivity sweep
never touches the official 134-task practice/exam split.

Usage (from repo root):
    python -m ae.runners.run_alfworld_devset \
        --run-name ae_tune_dev_baseline \
        --ae-config configs/controllers/ae_tune_dev_baseline.yaml \
        --model Qwen/Qwen3-8B --base-url http://gpuvm35:8000/v1
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "alfworld_runs_ae"))

from environment import make_single_task_env  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ae.llm_client import AnyOpenAILLM, call_counter  # noqa: E402
from ae.logging_utils import snapshot_config, load_done_ids, JsonlLogger  # noqa: E402
from ae.baselines import ae_full as ae_full_baseline  # noqa: E402
from ae.controllers.config import load_config as load_ae_config  # noqa: E402
from agents import ACT_MAX_TOKENS, ACT_STOP, PromptBudgetExceededError  # noqa: E402

DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "dev_set", "dev60_valid_seen_manifest.json")


def build_act_llm(model: str, base_url: str):
    return AnyOpenAILLM(temperature=0, max_tokens=ACT_MAX_TOKENS, model_name=model,
                         model_kwargs={"stop": ACT_STOP},
                         openai_api_key="EMPTY", openai_api_base=base_url)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
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

    manifest_doc = json.load(open(args.manifest))
    tasks = manifest_doc["manifest"]
    assert len(tasks) == 60, f"expected 60 dev tasks, got {len(tasks)}"

    done = load_done_ids(log_path, id_field="dev_idx")
    if done:
        print(f"checkpoint: {len(done)} done, resuming")

    act_llm = build_act_llm(args.model, args.base_url)
    logger = JsonlLogger(log_path)
    ae_config = load_ae_config(args.ae_config)
    print(f"ae_config loaded: warmup={ae_config.warmup_steps} grace={ae_config.recovery_grace_steps} "
          f"patch={ae_config.patch_duration_steps} max_interventions={ae_config.max_interventions} "
          f"cooldown={ae_config.cooldown_steps} pir_enabled={ae_config.post_intervention_repeat_enabled} "
          f"pir_reflect_double={ae_config.post_intervention_repeat_reflect_double}")

    for task in tasks:
        dev_idx = task["dev_idx"]
        if dev_idx in done:
            continue
        goal = task["goal_text"]
        task_type = task["task_type"]
        env_name = f"{task['task_dir']}/{task['trial_dir']}"

        env = make_single_task_env(task["gamefile"])
        c0 = call_counter()
        t0 = time.time()

        try:
            result = ae_full_baseline.run_episode(
                env, goal, task_type, act_llm, ae_config,
                termination_policy=args.termination_policy, task_id=env_name,
            )
        except PromptBudgetExceededError as e:
            print(f"[{dev_idx}/60] env={env_name} INCOMPLETE: {e}")
            result = {
                "baseline": "ae_full", "success": 0, "incomplete": True,
                "incomplete_reason": str(e),
                "tokens_before_truncation": e.tokens_before,
                "tokens_after_truncation": e.tokens_after,
            }
        finally:
            env.close()

        record = {
            "dev_idx": dev_idx, "task_type": task_type, "env_name": env_name,
            "task_dir": task["task_dir"], "trial_dir": task["trial_dir"],
            "goal": goal,
            "llm_calls": call_counter() - c0,
            "wall_s": round(time.time() - t0, 1),
            **result,
        }
        logger.write(record)
        if not result.get("incomplete"):
            print(f"[{dev_idx}/60] type={task_type} success={record['success']} "
                  f"calls={record['llm_calls']} wall_s={record['wall_s']}")

    logger.close()

    rows = [json.loads(l) for l in open(log_path)]
    n = len(rows)
    incomplete_rows = [r for r in rows if r.get("incomplete")]
    print("\n===== DEV60 RUN DONE =====")
    print(f"N={n}  success_rate={sum(r['success'] for r in rows)/n:.3f}  "
          f"avg_calls={sum(r['llm_calls'] for r in rows)/n:.1f}")
    if incomplete_rows:
        print(f"INCOMPLETE: {len(incomplete_rows)}/{n} task(s) never got a real result: "
              f"{[r['env_name'] for r in incomplete_rows]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
