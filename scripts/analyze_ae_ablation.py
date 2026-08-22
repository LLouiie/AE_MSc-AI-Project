#!/usr/bin/env python3
"""Aggregate auditable Table-2 AE ablation episode logs."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--expected-n", type=int, default=134)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()

    rows = []
    for name in args.logs:
        path = Path(name)
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())

    incomplete = [row for row in rows if row.get("incomplete")]
    unique_tasks = {row["env_name"] for row in rows}
    counts = {"verify": 0, "reflect": 0, "replan": 0}
    calls = steps = successes = triggered = 0
    for row in rows:
        summary = row.get("ae_summary") or {}
        successes += int(row.get("success", 0))
        calls += int(row.get("llm_calls", 0))
        steps += int(summary.get("environment_steps", 0))
        intervention_count = int(summary.get("intervention_count", 0))
        triggered += int(intervention_count > 0)
        for key in counts:
            counts[key] += int((summary.get("intervention_counts") or {}).get(key, 0))

    n = len(rows)
    result = {
        "method": args.mode,
        "repeat": args.repeat,
        "n_episodes": n,
        "n_unique_tasks": len(unique_tasks),
        "incomplete_episodes": len(incomplete),
        "successful_tasks": successes,
        "success_rate": successes / n if n else None,
        "average_llm_calls": calls / n if n else None,
        "average_environment_steps": steps / n if n else None,
        "verify_interventions": counts["verify"],
        "reflect_interventions": counts["reflect"],
        "replan_interventions": counts["replan"],
        "total_interventions": sum(counts.values()),
        "triggered_episodes": triggered,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2, sort_keys=True))

    if n != args.expected_n or len(unique_tasks) != args.expected_n or incomplete:
        raise SystemExit(
            f"invalid full run: episodes={n}, unique={len(unique_tasks)}, incomplete={len(incomplete)}"
        )


if __name__ == "__main__":
    main()
