"""JSONL append and config snapshot helpers,
reused so every ae/ baseline produces directly comparable run directories.
"""

from __future__ import annotations

import json
import os
import subprocess
import time


def snapshot_config(run_dir: str, args_dict: dict, repo_root: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=repo_root,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, cwd=repo_root,
    ).stdout.strip()
    payload = {**args_dict, "git_commit": commit, "git_branch": branch,
               "started": time.strftime("%F %T")}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(payload, f, indent=2)


def load_done_ids(log_path: str, id_field: str = "task_id") -> set:
    if not os.path.exists(log_path):
        return set()
    done = set()
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)[id_field])
    return done


class JsonlLogger:
    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "a")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
