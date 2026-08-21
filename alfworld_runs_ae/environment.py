"""
ALFWorld environment wrapper for the AE practice/exam protocol.

Data setup:
    pip install alfworld              # or: conda activate reflexion_hotpot
    export ALFWORLD_DATA=~/alfworld_data
    alfworld-download                 # downloads to $ALFWORLD_DATA

ExpeL task file (134 solvable valid_unseen tasks):
    Download alfworld_tasks_suffix.json from
    https://github.com/LeapLabTHU/ExpeL/blob/main/data/alfworld/alfworld_tasks_suffix.json
    and place at data/alfworld/alfworld_tasks_suffix.json (repo root), or
    point ALFWORLD_TASKS_FILE at wherever it actually lives — the previous
    hardcoded ../../AE/data/alfworld/... relative path assumed a sibling
    `AE` checkout next to this repo, which doesn't hold on a fresh clone.

Split: first 100 → practice, last 34 → exam.
"""

import os, json, yaml, importlib
import alfworld
import alfworld.agents.environment

TASKS_FILE = os.environ.get(
    "ALFWORLD_TASKS_FILE",
    os.path.join(os.path.dirname(__file__), '..', 'data', 'alfworld', 'alfworld_tasks_suffix.json'),
)
CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'alfworld_runs', 'base_config.yaml'
)

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo',
}

N_PRACTICE = 100
N_EXAM     = 34  # tasks 100-133


def load_task_list(tasks_file: str = TASKS_FILE):
    with open(tasks_file) as f:
        return json.load(f)


def get_practice_tasks(tasks_file: str = TASKS_FILE):
    return load_task_list(tasks_file)[:N_PRACTICE]


def get_exam_tasks(tasks_file: str = TASKS_FILE):
    return load_task_list(tasks_file)[N_PRACTICE:]


def get_all_tasks(tasks_file: str = TASKS_FILE):
    """Full, unsplit 134-task set -- for published-anchor baselines
    (react_reflact_anchor, reflexgrad_v4, reflexion_only_reflexgrad_v4),
    which reproduce published results on the whole eval_out_of_distribution
    split and must NOT go through the practice(100)/exam(34) split that is
    this project's own no-test-set-tuning protocol, not part of any of the
    published methods being reproduced. get_practice_tasks/get_exam_tasks
    above are untouched by this addition -- their behavior (and every
    caller's, e.g. ae/runners/run_alfworld.py) is unchanged."""
    return load_task_list(tasks_file)


def get_task_type(name: str) -> str:
    """Return the react_* prompt key prefix for a task name."""
    for prefix, key in PREFIXES.items():
        if name.startswith(prefix):
            return key
    return 'put'


def _resolve_env_cls(type_name: str):
    """alfworld<=0.3.5 re-exports Alfred*Env at the `environment` package
    level; alfworld>=0.4 moved them into per-class submodules
    (environment.alfred_tw_env.AlfredTWEnv etc.) without re-exporting.
    Try both so this works unpinned across the alfworld035 (0.3.5) and
    reflexion_hotpot (0.4.2) conda envs."""
    if hasattr(alfworld.agents.environment, type_name):
        return getattr(alfworld.agents.environment, type_name)
    submodule_name = {
        "AlfredTWEnv": "alfred_tw_env",
        "AlfredThorEnv": "alfred_thor_env",
        "AlfredHybrid": "alfred_hybrid",
    }[type_name]
    submodule = importlib.import_module(f"alfworld.agents.environment.{submodule_name}")
    return getattr(submodule, type_name)


def make_alfworld_env(config_file: str = CONFIG_FILE):
    """Create and return an ALFWorld AlfredTWEnv batch env (batch_size=1).

    WARNING: this registers all 134 valid_unseen games at once; each
    env.reset() advances through AlfredTWEnv's own directory-scan (os.walk)
    order, which does NOT match alfworld_tasks_suffix.json's order. Calling
    code that pulls a goal/task_type from the task list while resetting this
    env will get a goal/room mismatch (confirmed empirically 2026-07-25 —
    3/3 sampled resets loaded a different game than the task list's row
    implied). Use make_single_task_env for anything that needs a specific
    task's env to actually match its goal.
    """
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    split = "eval_out_of_distribution"
    env_cls = _resolve_env_cls(config["env"]["type"])
    env = env_cls(config, train_eval=split)
    return env.init_env(batch_size=1)


def resolve_gamefile(relative_gamefile: str) -> str:
    """alfworld_tasks_suffix.json's `gamefile` field is a path relative to
    ExpeL's own repo layout (e.g. 'data/alfworld/json_2.1.1/valid_unseen/...'),
    not a real path on this machine. Reconstruct the real path under
    $ALFWORLD_DATA by keeping everything from 'valid_unseen/' onward."""
    alfworld_data = os.environ["ALFWORLD_DATA"]
    suffix = relative_gamefile.split("valid_unseen/", 1)[1]
    return os.path.join(alfworld_data, "json_2.1.1", "valid_unseen", suffix)


def make_single_task_env(game_file_path: str, config_file: str = CONFIG_FILE):
    """Create an ALFWorld env that serves exactly one specific game file on
    every reset(), following ExpeL's per-task env pattern (each task gets
    its own env registered with only that one gamefile) instead of
    AlfredTWEnv's default of registering all 134 games and cycling through
    them in directory-scan order. This is what guarantees the goal text
    shown to the agent actually matches the room/objects it's placed in.
    Reuses AlfredTWEnv's own game-collection/filtering logic, just narrows
    `game_files` to one entry before registration."""
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    split = "eval_out_of_distribution"
    env_cls = _resolve_env_cls(config["env"]["type"])
    env = env_cls(config, train_eval=split)
    env.game_files = [game_file_path]
    env.num_games = 1
    return env.init_env(batch_size=1)


def process_ob(ob: str) -> str:
    """Strip the verbose navigation header from alfworld observations."""
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ') + 2:]
    return ob
