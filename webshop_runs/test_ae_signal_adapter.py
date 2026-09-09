from ae.controllers.config import load_config
from ae_signal_adapter import WebShopAdmissible, WebShopSignalExtractor

CONFIG = load_config("../configs/controllers/ae_full_final_nopir_v2_5.yaml")

def _extract(action, observation, before, after,
             recent_actions=(), recent_observations=(), think=False):
    return WebShopSignalExtractor(CONFIG.signals).extract(
        action=action, observation=observation,
        admissible_before=before, admissible_after=after,
        recent_actions=list(recent_actions),
        recent_observations=list(recent_observations),
        step_index=2, max_steps=15, is_think_action=think,
    )

def test_think_ok_is_not_world_repetition():
    state = WebShopAdmissible(
        ["click[Next >]"], state_signature=("search", 1)
    )
    s = _extract("think[inspect]", "OK.", state, state,
                 recent_actions=["think[inspect]"],
                 recent_observations=["OK."], think=True)
    assert s.invalid_action == 0
    assert s.repeated_observation == 0
    assert s.progress_signal == 0

def test_page_transition_is_progress():
    before = WebShopAdmissible(
        ["click[Next >]"], state_signature=("search", 1)
    )
    after = WebShopAdmissible(
        ["click[Next >]"], state_signature=("search", 2)
    )
    s = _extract("click[Next >]", "Page 2", before, after)
    assert s.invalid_action == 0
    assert s.local_state_change_proxy == 1
    assert s.progress_signal == 1

def test_terminal_empty_commands_still_preserve_end_state():
    before = WebShopAdmissible(
        ["click[Buy Now]"], state_signature=("item", "B1")
    )
    after = WebShopAdmissible([], state_signature=("end", "B1"))
    s = _extract("click[Buy Now]", "Your score: 1.0", before, after)
    assert s.invalid_action == 0
    assert s.local_state_change_proxy == 1
    assert s.unexpected_outcome == 0
    assert s.progress_signal == 1

def test_webshop_actions_repeat_only_on_same_command():
    state = WebShopAdmissible(["click[B2]"], state_signature=("search", 1))
    different = _extract(
        "click[B2]", "page", state, state, recent_actions=["click[B1]"]
    )
    same = _extract(
        "click[B2]", "page", state, state, recent_actions=["click[B2]"]
    )
    assert different.repeated_action == 0
    assert same.consecutive_exact_action_repeat == 1
