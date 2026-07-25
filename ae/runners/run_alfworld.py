"""Unified ALFWorld pilot entry point (Stage 7 of AE_MIGRATION_AUDIT.md).

Bare single-episode loop per task, no RulePool/scheduler — mirrors
hotpotqa_runs/run_episode.py's "no inter-episode state" design. Each
baseline is dispatched behind the same CLI/logging/checkpoint contract so
runs are directly comparable.

Usage (from repo root, reflexion_hotpot / alfworld035 conda env):
    python -m ae.runners.run_alfworld \
        --baseline react --split practice --limit 10 \
        --run-name pilot_react_smoke10 --seed 42
"""

import argparse
import json
import os
import random
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
from ae.baselines import react as react_baseline  # noqa: E402
from ae.baselines import reflexion as reflexion_baseline  # noqa: E402
from ae.baselines import ae_full as ae_full_baseline  # noqa: E402
from ae.controllers.config import load_config as load_ae_config  # noqa: E402

IMPLEMENTED_BASELINES = {"react", "reflexion", "ae_full"}
# fixed_interval / stateless_trigger / ae_no_hysteresis: interface reserved
# (see ae/controllers/config.py::AEConfig.mode docstring), not implemented
# this pass per the task spec ("第一轮优先保证 react 和 ae_full 可运行").
PLANNED_BASELINES = {"adapt", "reflact", "reflexgrad", "fixed_interval", "stateless_trigger", "ae_no_hysteresis"}


def build_llms(model: str, base_url: str):
    act_llm = AnyOpenAILLM(temperature=0, max_tokens=50, model_name=model,
                            model_kwargs={"stop": ["\n"]},
                            openai_api_key="EMPTY", openai_api_base=base_url)
    reflect_llm = AnyOpenAILLM(temperature=0, max_tokens=300, model_name=model,
                                openai_api_key="EMPTY", openai_api_base=base_url)
    return act_llm, reflect_llm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                    choices=sorted(IMPLEMENTED_BASELINES | PLANNED_BASELINES))
    p.add_argument("--split", choices=["practice", "exam"], default="practice")
    p.add_argument("--run-name", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-trials", type=int, default=4,
                    help="Reflexion only: max retry trials per task")
    p.add_argument("--ae-config", default=os.path.join(REPO_ROOT, "configs", "controllers", "ae_full.yaml"),
                    help="ae_full only: path to AEConfig yaml")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct"))
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "runs"))
    args = p.parse_args()

    if args.baseline in PLANNED_BASELINES:
        raise SystemExit(
            f"--baseline {args.baseline} is scaffolded but not implemented yet "
            f"(see BASELINE_RESEARCH.md). Implemented: {sorted(IMPLEMENTED_BASELINES)}"
        )

    random.seed(args.seed)

    run_dir = os.path.join(args.output_dir, args.run_name)
    log_path = os.path.join(run_dir, "episode_log.jsonl")
    snapshot_config(run_dir, vars(args), REPO_ROOT)

    tasks = get_practice_tasks() if args.split == "practice" else get_exam_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    done = load_done_ids(log_path, id_field="env_name")
    if done:
        print(f"checkpoint: {len(done)} done, resuming")

    act_llm, reflect_llm = build_llms(args.model, args.base_url)
    logger = JsonlLogger(log_path)
    ae_config = load_ae_config(args.ae_config) if args.baseline == "ae_full" else None

    for qi, task in enumerate(tasks, 1):
        # alfworld_tasks_suffix.json rows have {"goal", "gamefile"}, no
        # "env_name" — task_id is the gamefile's task-directory segment
        # (.../valid_unseen/<task_id>/trial_.../game.tw-pddl). The legacy
        # run_practice.py silently fell back to task.get("env_name", ...)
        # here, which never matched, so every task there was mis-typed as
        # the default "put"; fixed for this runner.
        env_name = task.get("env_name") or task["gamefile"].split("/")[-3]
        if env_name in done:
            continue
        goal = task["goal"]
        task_type = get_task_type(env_name)

        # One env per task, registered with exactly this task's gamefile
        # (ExpeL's pattern) — make_alfworld_env's shared batch env cycles
        # through all 134 games in AlfredTWEnv's own directory-scan order,
        # which does not match alfworld_tasks_suffix.json's order, so the
        # goal shown to the agent would not match the room/objects it's
        # placed in (confirmed empirically 2026-07-25, 3/3 sampled resets
        # mismatched). This guarantees goal and env always correspond.
        env = make_single_task_env(resolve_gamefile(task["gamefile"]))

        c0 = call_counter()
        t0 = time.time()

        if args.baseline == "react":
            result = react_baseline.run_episode(env, goal, task_type, act_llm)
        elif args.baseline == "reflexion":
            result = reflexion_baseline.run_episode(
                env, goal, task_type, act_llm, reflect_llm, max_trials=args.max_trials
            )
        elif args.baseline == "ae_full":
            result = ae_full_baseline.run_episode(env, goal, task_type, act_llm, ae_config)
        else:
            raise AssertionError("unreachable")

        env.close()

        record = {
            "env_name": env_name, "q_index": qi, "goal": goal,
            "llm_calls": call_counter() - c0,
            "wall_s": round(time.time() - t0, 1),
            **result,
        }
        logger.write(record)
        print(f"[{qi}/{len(tasks)}] baseline={args.baseline} success={record['success']} "
              f"calls={record['llm_calls']} wall_s={record['wall_s']}")

    logger.close()

    rows = [json.loads(l) for l in open(log_path)]
    n = len(rows)
    print("\n===== PILOT DONE =====")
    print(f"N={n}  success_rate={sum(r['success'] for r in rows)/n:.3f}  "
          f"avg_calls={sum(r['llm_calls'] for r in rows)/n:.1f}")


if __name__ == "__main__":
    main()
