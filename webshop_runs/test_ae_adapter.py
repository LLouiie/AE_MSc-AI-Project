from ae.controllers.config import load_config
from ae.controllers.stateful_controller import StatefulController
from ae.core import InterventionType

from run_ae import _admissible, _webshop_directive


def test_page_commands_are_kept_separate_from_think_display():
    page = "[Next >] [B012345678]"
    commands = _admissible(page, "search")
    assert "click[Next >]" in commands
    assert "click[B012345678]" in commands
    assert _admissible("OK.", "search") != commands


def test_replan_uses_webshop_single_command_protocol():
    config = load_config("../configs/controllers/ae_full_final_nopir_v2_5.yaml")
    controller = StatefulController(config)
    controller.active_patch_type = InterventionType.REPLAN
    controller.active_patch_remaining = 1
    directive = _webshop_directive(controller, "buy the requested product")
    assert "Original task: buy the requested product" in directive
    assert "In one concise Thought line:" not in directive
    assert "Do not output a Thought: or Action: prefix." in directive
