"""Published-anchor ALFWorld entry point (reproduction spec section one).

Separate from ae/runners/run_alfworld.py on purpose: that runner's
practice(100)/exam(34) split is this project's own no-test-set-tuning
protocol, layered onto react/reflexion/ae_full by this project, not part of
any of the published methods reproduced here. run_alfworld.py's default
semantics are untouched by this file's existence -- nothing here imports
into it or changes its behavior.

This runner always evaluates the full, unsplit 134-task
eval_out_of_distribution set (alfworld_runs_ae/environment.py::
get_all_tasks) and always stamps the manifest with:
    protocol_kind: published_anchor
    evaluation_scope: full_134
so a published-anchor result can never be silently compared column-for-
column against the AE project's own 34-task exam number.

Reuses this repo's own environment creation (make_single_task_env,
resolve_gamefile), success determination (info["won"], inside each
baseline's run_episode), and logging/resume infrastructure
(ae/logging_utils.py) -- no parallel environment implementation.

Usage (from repo root):
    python -m ae.runners.run_alfworld_anchor \
        --baseline react_reflact_anchor --run-name anchor_react_smoke2 --limit 2
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "alfworld_runs_ae"))

from environment import get_all_tasks, get_task_type, make_single_task_env, resolve_gamefile  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ae.logging_utils import snapshot_config, load_done_ids, JsonlLogger  # noqa: E402
from ae.baselines import react_reflact_anchor  # noqa: E402
from ae.baselines import reflexgrad_v4, reflexion_only_reflexgrad_v4  # noqa: E402
from ae.baselines import reflexgrad_v4_episode  # noqa: E402
from ae.baselines.reflexgrad_v4_llm import ReflexGradChatLLM  # noqa: E402

IMPLEMENTED_ANCHOR_BASELINES = {
    "react_reflact_anchor", "reflexgrad_v4", "reflexion_only_reflexgrad_v4",
}
PLANNED_ANCHOR_BASELINES = set()  # original_reflexion_qwen3_ours -- explicitly deferred, not this phase

_REFLEXGRAD_BUILD_ENGINE_FN = {
    "reflexgrad_v4": reflexgrad_v4.build_engine,
    "reflexion_only_reflexgrad_v4": reflexion_only_reflexgrad_v4.build_engine,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                    choices=sorted(IMPLEMENTED_ANCHOR_BASELINES | PLANNED_ANCHOR_BASELINES))
    p.add_argument("--run-name", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--max-agent-steps", type=int, default=None,
                    help="Defaults to 30 for react_reflact_anchor, 15 for reflexgrad_v4/"
                         "reflexion_only_reflexgrad_v4 (reproduction spec sections 2/3/4).")
    p.add_argument("--max-completion-tokens", type=int,
                    default=react_reflact_anchor.MAX_COMPLETION_TOKENS,
                    help="react_reflact_anchor only.")
    p.add_argument("--reflexgrad-temperature", type=float, default=0.2,
                    help="reflexgrad_v4/reflexion_only_reflexgrad_v4 only -- matches the "
                         "current official wrapper's behavior, NOT the paper's stated "
                         "'default deterministic' text (disclosed reproduction gap).")
    p.add_argument("--max-api-retries", type=int, default=0,
                    help="reflexgrad_v4/reflexion_only_reflexgrad_v4 only -- physical HTTP "
                         "retry count per logical LLM call, 0 = no retries.")
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "runs"))
    args = p.parse_args()

    if args.baseline in PLANNED_ANCHOR_BASELINES:
        raise SystemExit(
            f"--baseline {args.baseline} is not implemented yet. "
            f"Implemented: {sorted(IMPLEMENTED_ANCHOR_BASELINES)}"
        )

    if args.max_agent_steps is None:
        args.max_agent_steps = (
            react_reflact_anchor.MAX_AGENT_STEPS if args.baseline == "react_reflact_anchor"
            else reflexgrad_v4_episode.MAX_AGENT_STEPS
        )

    run_dir = os.path.join(args.output_dir, args.run_name)
    log_path = os.path.join(run_dir, "episode_log.jsonl")
    snapshot_config(run_dir, {
        **vars(args),
        "protocol_kind": "published_anchor",
        "evaluation_scope": "full_134",
    }, REPO_ROOT)

    tasks = get_all_tasks()
    assert len(tasks) == 134, f"published-anchor evaluation_scope=full_134 expected 134 tasks, got {len(tasks)}"
    if args.limit:
        tasks = tasks[: args.limit]

    done = load_done_ids(log_path, id_field="env_name")
    if done:
        print(f"checkpoint: {len(done)} done, resuming")

    if args.baseline == "react_reflact_anchor":
        chat_llm = react_reflact_anchor.QwenChatAnchorLLM(
            model=args.model, base_url=args.base_url, max_tokens=args.max_completion_tokens,
        )
    else:
        chat_llm = ReflexGradChatLLM(
            model=args.model, base_url=args.base_url, temperature=args.reflexgrad_temperature,
            max_api_retries=args.max_api_retries,
        )
    logger = JsonlLogger(log_path)

    for qi, task in enumerate(tasks, 1):
        env_name = task.get("env_name") or task["gamefile"].split("/")[-3]
        if env_name in done:
            continue
        task_type = get_task_type(env_name)

        env = make_single_task_env(resolve_gamefile(task["gamefile"]))
        t0 = time.time()
        try:
            if args.baseline == "react_reflact_anchor":
                result = react_reflact_anchor.run_episode(
                    env, env_name, chat_llm, task_id=env_name,
                    max_agent_steps=args.max_agent_steps,
                )
            elif args.baseline in _REFLEXGRAD_BUILD_ENGINE_FN:
                result = reflexgrad_v4_episode.run_episode(
                    env, env_name, task["goal"], chat_llm,
                    build_engine_fn=_REFLEXGRAD_BUILD_ENGINE_FN[args.baseline],
                    task_id=env_name, max_agent_steps=args.max_agent_steps,
                )
            else:
                raise AssertionError("unreachable")
        finally:
            env.close()

        record = {
            "env_name": env_name, "q_index": qi, "goal": task["goal"], "task_type": task_type,
            "wall_s": round(time.time() - t0, 1),
            **result,
        }
        logger.write(record)
        print(f"[{qi}/{len(tasks)}] baseline={args.baseline} env={env_name} "
              f"success={record['success']} agent_steps={record['agent_steps']} "
              f"env_actions={record['env_actions']} wall_s={record['wall_s']}")

    logger.close()

    rows = [json.loads(l) for l in open(log_path)]
    n = len(rows)
    print("\n===== PUBLISHED-ANCHOR RUN DONE =====")
    print(f"N={n}  success_rate={sum(r['success'] for r in rows)/n:.3f}  "
          f"protocol_kind=published_anchor  evaluation_scope=full_134")


if __name__ == "__main__":
    main()
