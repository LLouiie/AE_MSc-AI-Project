"""
ALFWorld environment wrapper for the AE practice/exam protocol.

Data setup:
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate reflexion_hotpot
    alfworld-download          # downloads to $ALFWORLD_DATA (default: ~/alfworld_data)

ExpeL task file (134 solvable valid_unseen tasks):
    Download alfworld_tasks_suffix.json from
    https://github.com/LeapLabTHU/ExpeL/blob/main/data/alfworld/alfworld_tasks_suffix.json
    and place at ~/projects/AE/data/alfworld/alfworld_tasks_suffix.json

Split: first 100 → practice, last 34 → exam.
"""

import os, json, yaml, importlib
import alfworld
import alfworld.agents.environment

TASKS_FILE = os.path.join(
    os.path.dirname(__file__), '..', '..', 'AE', 'data', 'alfworld', 'alfworld_tasks_suffix.json'
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


def get_task_type(name: str) -> str:
    """Return the react_* prompt key prefix for a task name."""
    for prefix, key in PREFIXES.items():
        if name.startswith(prefix):
            return key
    return 'put'


def make_alfworld_env(config_file: str = CONFIG_FILE):
    """Create and return an ALFWorld AlfredTWEnv batch env (batch_size=1)."""
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    split = "eval_out_of_distribution"
    env_cls = getattr(alfworld.agents.environment, config["env"]["type"])
    env = env_cls(config, train_eval=split)
    return env.init_env(batch_size=1)


def process_ob(ob: str) -> str:
    """Strip the verbose navigation header from alfworld observations."""
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ') + 2:]
    return ob
