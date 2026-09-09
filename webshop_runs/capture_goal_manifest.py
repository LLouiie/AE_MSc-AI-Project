"""Capture the exact fixed WebShop instructions used by an evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from webshop_trial import webshopEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    env = webshopEnv()
    goals = []
    for index in range(args.num_envs):
        session = f"fixed_{index}"
        goal, _, _ = env.step(session, "reset")
        goals.append({"session": session, "goal": goal})
    encoded = json.dumps(goals, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "goal_seed": 233,
        "count": len(goals),
        "goal_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "goals": goals,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
