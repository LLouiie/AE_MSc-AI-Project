"""Run the production AE controller on the original WebShop ReAct loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEBSHOP_DIR = Path(__file__).resolve().parent
if str(WEBSHOP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBSHOP_DIR))

from ae.controllers.config import load_config
from ae.controllers.stateful_controller import StatefulController
from ae.core import InterventionType
from ae_signal_adapter import WebShopAdmissible, WebShopSignalExtractor
from webshop_trial import BASE_PROMPT, llm, webshopEnv


MAX_STEPS = 50
_CLICK_TARGET = re.compile(r"\[([^\[\]\n]+)\]")


def _state_signature(state: dict) -> tuple:
    return (
        state.get("page_type"), state.get("query_string"), state.get("page_num"),
        state.get("asin"), state.get("subpage"),
        tuple(sorted(state.get("options", {}).items())),
    )


def _admissible(observation: str, page_type: str,
                state: dict | None = None) -> WebShopAdmissible:
    commands = [f"click[{target}]" for target in _CLICK_TARGET.findall(observation)]
    if page_type == "init":
        commands.append("search[query]")
    return WebShopAdmissible(
        commands, state_signature=_state_signature(state or {})
    )

def _webshop_directive(
    controller: StatefulController, goal: str, *, shopping_guidance: bool = True
) -> str:
    """Render the active directive using WebShop wording and output format."""
    directive = controller.active_directive(goal=goal)
    if not directive:
        return ""
    is_replan = controller.active_patch_type == InterventionType.REPLAN
    directive = directive.split("\nRespond with exactly two lines:", 1)[0].rstrip()
    if is_replan:
        directive = directive.replace(
            "Execute only the next unmet subgoal.",
            "Consider the route internally and execute only the next unmet subgoal.",
        )
    else:
        if controller.active_patch_type == InterventionType.REFLECT:
            directive = directive.replace(
                "Diagnose why the recent actions did not change the environment.",
                "Diagnose why the recent actions did not change the page or provide useful information.",
            ).replace(
                "Choose a corrected action from the current environment state.",
                "Choose a corrected action from the current page state.",
            )
        if shopping_guidance:
            task_match = re.search(r"Instruction:\s*(.*?)\s*\[Search\]", goal, re.DOTALL)
            shopping_task = task_match.group(1).strip() if task_match else goal.strip()
            directive += (
                f"\nShopping task: {shopping_task}\n"
                "Prioritize hard constraints: product type, required size or quantity, "
                "required selectable options, and maximum price. Treat wording such as "
                "prefer or would like as a preference when an exact match is unavailable. "
                "Use the recent trajectory: do not repeat the same search query or the "
                "same navigation loop. Make at most one materially different search or "
                "one additional detail check for missing evidence. Then choose the "
                "best-supported visible candidate; before click[Buy Now], select any "
                "visible required options."
            )
    directive += (
        "\nReturn exactly one WebShop command:\n"
        "search[query], click[visible target], or think[short note].\n"
        "Do not output a Thought: or Action: prefix or any extra text."
    )
    return directive


def _build_model_prompt(prompt: str, directive: str, goal: str) -> str:
    """Keep the task and output cue while trimming the oldest trajectory."""
    if directive:
        action_cue = "\n\nAction:"
        if not prompt.endswith(action_cue):
            raise ValueError("WebShop trajectory must end with an Action cue")
        prompt = prompt[:-len(action_cue)] + f"\n\n{directive}{action_cue}"
    trajectory_budget = 6400 - len(BASE_PROMPT)
    if len(prompt) <= trajectory_budget:
        return BASE_PROMPT + prompt
    task_prefix = f"{goal}\n\n"
    history_budget = trajectory_budget - len(task_prefix)
    if history_budget <= 0:
        raise ValueError("WebShop task exceeds the prompt budget")
    return BASE_PROMPT + task_prefix + prompt[-history_budget:]



def run_episode(
    index: int,
    controller_config,
    *,
    shopping_guidance: bool,
    ablation_mode: str,
    random_seed: int,
    random_trigger_probability: float,
) -> dict:
    session = f"fixed_{index}"
    env = webshopEnv()
    observation, _, _ = env.step(session, "reset")
    goal = observation
    controller = StatefulController(
        controller_config,
        ablation_mode=ablation_mode,
        random_seed=random_seed,
        random_trigger_probability=random_trigger_probability,
    )
    controller.signal_extractor = WebShopSignalExtractor(controller_config.signals)
    prompt = f"{observation}\n\nAction:"
    page_observation = observation
    actions: list[str] = []
    observations: list[str] = []
    reward = 0.0
    done = False

    # Use the WebShop evaluation horizon, including its initial reset,
    # hence at most MAX_STEPS - 1 model-generated environment actions.
    for _ in range(MAX_STEPS - 1):
        directive = _webshop_directive(
            controller, goal, shopping_guidance=shopping_guidance
        )
        model_prompt = _build_model_prompt(prompt, directive, goal)
        action = llm(model_prompt, stop=["\n"]).lstrip(" ")
        page_before = env.sessions[session]["page_type"]
        admissible_before = _admissible(
            page_observation, page_before, env.sessions[session]
        )
        # WebShop accepts arbitrary query text on its initial page.
        if page_before == "init" and action.startswith("search["):
            admissible_before.append(action)
        try:
            raw_observation, reward, done = env.step(session, action)
            action_valid = True
        except AssertionError:
            raw_observation, reward, done = (
                "Invalid action! From a product or search-results page, use "
                "click[Back to Search] before search[...]. Choose a currently "
                "visible click target.",
                0.0,
                False,
            )
            action_valid = False
        # Option clicks return only an acknowledgement such as "You have
        # clicked blue."  Keep the last full page in that case so buttons
        # and remaining options stay admissible on the following step.
        if action_valid and (done or _CLICK_TARGET.search(raw_observation)):
            page_observation = raw_observation
        next_observation = "OK." if action.startswith("think[") else raw_observation

        page_after = env.sessions[session]["page_type"]
        controller.step(
            action=action,
            observation=next_observation,
            admissible_before=admissible_before,
            recent_actions=actions,
            recent_observations=observations,
            max_steps=MAX_STEPS,
            is_think_action=action.startswith("think["),
            admissible_after=_admissible(
                page_observation, page_after, env.sessions[session]
            ),
            reward=reward,
            done=done,
            won=done and reward == 1.0,
            display_action_text=action,
            display_observation_text=next_observation,
        )
        actions.append(action)
        observations.append(next_observation)
        prompt += f" {action}\nObservation: {next_observation}\n\nAction:"
        observation = next_observation
        if done:
            break

    success = done and reward == 1.0
    termination = "success" if success else ("env_done_partial" if done else "max_steps")
    ae_summary = controller.summary(
        success=success, env_steps=len(actions), termination_reason=termination
    )
    terminal_intervention = None
    if done and controller.step_log:
        final_step = controller.step_log[-1]
        final_type = final_step.get("intervention_type")
        if final_type and final_step.get("intervention_id") is not None:
            terminal_intervention = final_type
            ae_summary["intervention_count"] -= 1
            ae_summary["intervention_counts"][final_type] -= 1
            ae_summary["trigger_steps"] = [
                step for step in ae_summary["trigger_steps"]
                if step != final_step["step"]
            ]
    ae_summary["terminal_intervention_excluded"] = terminal_intervention
    return {
        "session": session,
        "goal": goal,
        "reward": float(reward),
        "success": bool(success),
        "done": bool(done),
        "actions": actions,
        "observations": observations,
        "ae_step_log": controller.step_log,
        "ae_summary": ae_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--ae-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shopping-guidance", choices=("on", "off"), default="on")
    parser.add_argument(
        "--ae-ablation",
        choices=(
            "full", "no_trigger", "random_trigger", "reflect_only",
            "replan_only", "verify_only",
        ),
        default="full",
    )
    parser.add_argument("--ae-random-seed", type=int, default=42)
    parser.add_argument("--ae-random-trigger-probability", type=float, default=0.5)
    args = parser.parse_args()
    config = load_config(args.ae_config)
    shopping_guidance = args.shopping_guidance == "on"
    episodes = [
        run_episode(
            i,
            config,
            shopping_guidance=shopping_guidance,
            ablation_mode=args.ae_ablation,
            random_seed=args.ae_random_seed + i,
            random_trigger_probability=args.ae_random_trigger_probability,
        )
        for i in range(args.num_envs)
    ]
    goal_manifest = [
        {"session": episode["session"], "goal": episode["goal"]}
        for episode in episodes
    ]
    goal_manifest_sha256 = hashlib.sha256(
        json.dumps(goal_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "method": "AE",
        "count": len(episodes),
        "average_reward": sum(x["reward"] for x in episodes) / len(episodes),
        "success_rate": sum(x["success"] for x in episodes) / len(episodes),
        "triggered_episodes": sum(x["ae_summary"]["intervention_count"] > 0 for x in episodes),
        "interventions": sum(x["ae_summary"]["intervention_count"] for x in episodes),
        "goal_seed": 233,
        "goal_manifest_sha256": goal_manifest_sha256,
        "shopping_guidance": args.shopping_guidance,
        "ae_ablation": args.ae_ablation,
        "ae_random_seed": args.ae_random_seed,
        "ae_random_seed_policy": "base_seed_plus_episode_index",
        "ae_random_trigger_probability": args.ae_random_trigger_probability,
        "average_environment_steps": sum(len(x["actions"]) for x in episodes) / len(episodes),
        "intervention_counts": {
            kind: sum(
                x["ae_summary"]["intervention_counts"].get(kind, 0)
                for x in episodes
            )
            for kind in ("verify", "reflect", "replan")
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"episodes": episodes, "overall": summary}, indent=2))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
