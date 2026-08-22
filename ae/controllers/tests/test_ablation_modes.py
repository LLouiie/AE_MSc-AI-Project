"""Single-variable invariants for the Table 2 AE ablations."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from ae.controllers.config import AEConfig
from ae.controllers.stateful_controller import StatefulController
from ae.core import InterventionType


def test_type_restrictions():
    reflect = StatefulController(AEConfig(), ablation_mode="reflect_only")
    replan = StatefulController(AEConfig(), ablation_mode="replan_only")
    for intervention in (InterventionType.VERIFY, InterventionType.REFLECT, InterventionType.REPLAN):
        assert reflect._apply_type_ablation(intervention) == InterventionType.REFLECT
        assert replan._apply_type_ablation(intervention) == InterventionType.REPLAN


def test_random_schedule_is_seeded():
    a = StatefulController(AEConfig(), ablation_mode="random_trigger", random_seed=123)
    b = StatefulController(AEConfig(), ablation_mode="random_trigger", random_seed=123)
    assert [a._random_intervention() for _ in range(50)] == [b._random_intervention() for _ in range(50)]


if __name__ == "__main__":
    test_type_restrictions()
    test_random_schedule_is_seeded()
    print("2 passed, 0 failed")
