"""Launch WebShop with deterministic generation of the fixed goal set."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import web_agent_site.app as backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--attrs", action="store_true")
    parser.add_argument("--goal-seed", type=int, default=233)
    args = parser.parse_args()

    # backend.index() lazily calls get_goals() on the first HTTP request.
    # Seed here, after all imports and immediately before app.run(), so goal
    # price limits and the subsequent fixed ordering are reproducible.
    random.seed(args.goal_seed)
    if args.log:
        backend.user_log_dir = Path("user_session_logs/mturk")
        backend.user_log_dir.mkdir(parents=True, exist_ok=True)
    backend.SHOW_ATTRS_TAB = args.attrs
    backend.app.run(host="0.0.0.0", port=3000)


if __name__ == "__main__":
    main()
