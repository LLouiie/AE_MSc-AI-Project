"""Tests for environment.py::get_all_tasks (published-anchor, full 134
tasks) and the regression guarantee that adding it left
get_practice_tasks/get_exam_tasks completely unchanged.

Plain assert-based, same convention as test_output_parser.py.
Run directly:
    python3 alfworld_runs_ae/tests/test_environment_anchor.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # alfworld_runs_ae

from environment import N_EXAM, N_PRACTICE, get_all_tasks, get_exam_tasks, get_practice_tasks  # noqa: E402

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def _task_id(t):
    return t.get("env_name") or t["gamefile"].split("/")[-3]


def test_get_all_tasks_returns_134():
    tasks = get_all_tasks()
    check("1. get_all_tasks returns exactly 134 tasks", len(tasks) == 134, f"got {len(tasks)}")


def test_get_all_tasks_ids_are_unique_and_stable_across_calls():
    tasks_a = get_all_tasks()
    tasks_b = get_all_tasks()
    ids_a = [_task_id(t) for t in tasks_a]
    ids_b = [_task_id(t) for t in tasks_b]
    check("2. task IDs identical across repeated calls (stable)", ids_a == ids_b)
    check("3. all 134 gamefiles are unique (no accidental duplicate task rows)",
          len(set(t["gamefile"] for t in tasks_a)) == 134,
          f"got {len(set(t['gamefile'] for t in tasks_a))} unique gamefiles")


def test_practice_exam_split_unchanged_and_composes_to_all_tasks():
    practice = get_practice_tasks()
    exam = get_exam_tasks()
    all_tasks = get_all_tasks()
    check("4. practice split size unchanged (N_PRACTICE)", len(practice) == N_PRACTICE == 100)
    check("5. exam split size unchanged (N_EXAM)", len(exam) == N_EXAM == 34)
    check("6. practice ++ exam == get_all_tasks() (same content, same order)",
          practice + exam == all_tasks)


def test_get_practice_and_exam_do_not_import_or_call_get_all_tasks_differently():
    # Regression guard against a future edit accidentally changing
    # get_practice_tasks/get_exam_tasks to route through get_all_tasks in a
    # way that changes slicing semantics -- re-verifies the exact slice
    # boundaries directly against load_task_list, independent of get_all_tasks.
    from environment import load_task_list
    raw = load_task_list()
    check("7. get_practice_tasks() == load_task_list()[:100] exactly",
          get_practice_tasks() == raw[:100])
    check("8. get_exam_tasks() == load_task_list()[100:] exactly",
          get_exam_tasks() == raw[100:])


if __name__ == "__main__":
    test_get_all_tasks_returns_134()
    test_get_all_tasks_ids_are_unique_and_stable_across_calls()
    test_practice_exam_split_unchanged_and_composes_to_all_tasks()
    test_get_practice_and_exam_do_not_import_or_call_get_all_tasks_differently()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
