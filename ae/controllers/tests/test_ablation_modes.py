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
    verify = StatefulController(AEConfig(), ablation_mode="verify_only")
    for intervention in (InterventionType.VERIFY, InterventionType.REFLECT, InterventionType.REPLAN):
        assert reflect._apply_type_ablation(intervention) == (
            intervention if intervention == InterventionType.REFLECT else InterventionType.CONTINUE
        )
        assert replan._apply_type_ablation(intervention) == (
            intervention if intervention == InterventionType.REPLAN else InterventionType.CONTINUE
        )
        assert verify._apply_type_ablation(intervention) == (
            intervention if intervention == InterventionType.VERIFY else InterventionType.CONTINUE
        )


def test_random_schedule_is_seeded():
    a = StatefulController(AEConfig(), ablation_mode="random_trigger", random_seed=123)
    b = StatefulController(AEConfig(), ablation_mode="random_trigger", random_seed=123)
    assert a.random_trigger_probability == 0.5
    draws_a = [a._random_intervention() for _ in range(3000)]
    draws_b = [b._random_intervention() for _ in range(3000)]
    assert draws_a == draws_b
    for intervention in (InterventionType.VERIFY, InterventionType.REFLECT, InterventionType.REPLAN):
        assert abs(draws_a.count(intervention) / 3000 - 1 / 3) < 0.03


if __name__ == "__main__":
    test_type_restrictions()
    test_random_schedule_is_seeded()
    print("2 passed, 0 failed")
