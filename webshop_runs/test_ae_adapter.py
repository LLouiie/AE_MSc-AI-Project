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
    assert "[ACTIVE CONTROL DIRECTIVE: REPLAN]" in directive
    assert "No observed state change has occurred for 0 steps." in directive
    assert "Consider the route internally and execute only the next unmet subgoal." in directive
    assert "Respond with exactly two lines:" not in directive
    assert "Do not output a Thought: or Action: prefix or any extra text." in directive


def _directive_for(kind):
    config = load_config("../configs/controllers/ae_full_final_nopir_v2_5.yaml")
    controller = StatefulController(config)
    controller.active_patch_type = kind
    controller.active_patch_remaining = 1
    return _webshop_directive(controller, "Instruction: buy blue socks [Search]")


def test_verify_uses_webshop_appendix_protocol():
    directive = _directive_for(InterventionType.VERIFY)
    assert directive.startswith("[ACTIVE CONTROL DIRECTIVE: VERIFY]\n")
    assert "Re-check the current goal and latest observation." in directive
    assert directive.endswith("Do not output a Thought: or Action: prefix or any extra text.")
    assert "Respond with exactly two lines:" not in directive


def test_reflect_uses_webshop_appendix_wording():
    directive = _directive_for(InterventionType.REFLECT)
    assert directive.startswith("[ACTIVE CONTROL DIRECTIVE: REFLECT]\n")
    assert "did not change the page or provide useful information" in directive
    assert "current page state" in directive
    assert "did not change the environment" not in directive
    assert "current environment state" not in directive


def test_replan_omits_shopping_block_and_uses_internal_route():
    directive = _directive_for(InterventionType.REPLAN)
    assert "Consider the route internally and execute only the next unmet subgoal." in directive
    assert "Shopping task:" not in directive
    assert "Execute only the next unmet subgoal.\n" not in directive
