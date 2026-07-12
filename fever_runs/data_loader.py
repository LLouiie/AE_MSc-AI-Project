"""
FEVER task selection following ExpeL/ReAct conventions.

ExpeL uses:
    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)
    tasks = [FeverEnv(idx) for idx in idxs[:100]]

We further split those 100 into practice (first 70) and exam (last 30).
Data file: ~/projects/AE/data/fever/paper_dev.jsonl (9999 rows)

Download: https://fever.ai/dataset/fever.html  (paper_dev.jsonl)
"""

import json
import random
import os

DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'AE', 'data', 'fever'
)
DEFAULT_DEV = os.path.join(DATA_DIR, 'paper_dev.jsonl')
SPLITS_DIR  = os.path.join(DATA_DIR, 'splits')

SEED = 233
POOL_SIZE = 7405
N_CANONICAL = 100
N_PRACTICE  = 70
N_EXAM      = 30


def _load_all(data_path: str):
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _canonical_idxs() -> list:
    idxs = list(range(POOL_SIZE))
    random.Random(SEED).shuffle(idxs)
    return idxs[:N_CANONICAL]


def get_canonical_tasks(data_path: str = DEFAULT_DEV) -> list:
    """Return the 100 canonical FEVER tasks used by ExpeL/ReAct."""
    records = _load_all(data_path)
    return [
        {"id": idx, "question": records[idx]["claim"], "answer": records[idx]["label"]}
        for idx in _canonical_idxs()
    ]


def get_practice_tasks(data_path: str = DEFAULT_DEV) -> list:
    """First 70 of the canonical 100."""
    return get_canonical_tasks(data_path)[:N_PRACTICE]


def get_exam_tasks(data_path: str = DEFAULT_DEV) -> list:
    """Last 30 of the canonical 100 (held-out)."""
    return get_canonical_tasks(data_path)[N_PRACTICE:]


def make_splits(data_path: str = DEFAULT_DEV, out_dir: str = SPLITS_DIR):
    """Persist practice/exam splits to JSON files."""
    os.makedirs(out_dir, exist_ok=True)
    practice = get_practice_tasks(data_path)
    exam = get_exam_tasks(data_path)

    with open(os.path.join(out_dir, 'practice_fever70.json'), 'w') as f:
        json.dump(practice, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, 'exam_fever30.json'), 'w') as f:
        json.dump(exam, f, indent=2, ensure_ascii=False)

    manifest = {
        "source": os.path.abspath(data_path),
        "pool_size": POOL_SIZE,
        "seed": SEED,
        "canonical_n": N_CANONICAL,
        "practice_n": N_PRACTICE,
        "exam_n": N_EXAM,
        "note": "ExpeL canonical 100 (idxs[:100]) split 70/30 for AE practice/exam protocol",
    }
    with open(os.path.join(out_dir, 'fever_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved practice ({N_PRACTICE}) and exam ({N_EXAM}) splits to {out_dir}")


if __name__ == "__main__":
    make_splits()
