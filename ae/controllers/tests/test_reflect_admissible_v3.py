"""REFLECT prompt-v3 (admissible-action grounding) unit tests.

Covers the contract of the change: the v1 REFLECT body is preserved
verbatim and only appended to; the admissible list is rendered when
supplied and omitted honestly when not; VERIFY and REPLAN render
byte-identically to v1; and the controller threads its recorded
admissible_after through active_directive() without it ever influencing
routing.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ae.core import InterventionType
from ae.controllers.intervention_renderer import (
    render_directive, _STATIC_DIRECTIVES, _ALFWORLD_OUTPUT, _ALFWORLD_REFLECT_OUTPUT,
)

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


V1_REFLECT_BODY = (
    "Diagnose why the recent actions did not change the environment.\n"
    "Do not repeat the same failed action.\n"
    "Choose a corrected action from the current environment state."
)

V1_VERIFY_BODY = (
    "Re-check the current goal and latest observation.\n"
    "Identify what information is still missing.\n"
    "Prefer one information-gathering action before changing the whole strategy."
)

ADM = ["go to desk 1", "go to desk 2", "use desklamp 1", "inventory", "look"]

# 1. v1 REFLECT body is preserved byte-for-byte in the dict.
check("1. _STATIC_DIRECTIVES REFLECT body unchanged from v1",
      _STATIC_DIRECTIVES[InterventionType.REFLECT] == V1_REFLECT_BODY)

# 2. VERIFY unchanged from v1, with and without admissible commands passed.
v_none = render_directive(InterventionType.VERIFY)
v_adm = render_directive(InterventionType.VERIFY, admissible_commands=ADM)
check("2a. VERIFY renders v1 body", v_none == f"[ACTIVE CONTROL DIRECTIVE: VERIFY]\n{V1_VERIFY_BODY}\n{_ALFWORLD_OUTPUT}")
check("2b. VERIFY ignores admissible_commands entirely", v_none == v_adm)

# 3. REFLECT with no admissible list renders exactly v1.
r_none = render_directive(InterventionType.REFLECT)
check("3a. REFLECT without list == v1 exactly",
      r_none == f"[ACTIVE CONTROL DIRECTIVE: REFLECT]\n{V1_REFLECT_BODY}\n{_ALFWORLD_REFLECT_OUTPUT}")
check("3b. REFLECT with empty list == v1 exactly",
      render_directive(InterventionType.REFLECT, admissible_commands=[]) == r_none)
check("3c. REFLECT with whitespace-only entries == v1 exactly",
      render_directive(InterventionType.REFLECT, admissible_commands=["  ", ""]) == r_none)

# 4. REFLECT with a list: v1 body preserved as a prefix, all commands present.
r_adm = render_directive(InterventionType.REFLECT, admissible_commands=ADM)
check("4a. v1 body still present verbatim", V1_REFLECT_BODY in r_adm)
check("4b. every admissible command is listed", all(c in r_adm for c in ADM))
check("4c. states that off-list commands do nothing", "will do nothing" in r_adm)
check("4d. tells the model to copy wording exactly", "copying its wording exactly" in r_adm)
check("4e. commands are listed alphabetically",
      r_adm.index("go to desk 1") < r_adm.index("inventory") < r_adm.index("use desklamp 1"))

# 5. Truncation is disclosed, never silent.
many = [f"go to shelf {i}" for i in range(60)]
r_many = render_directive(InterventionType.REFLECT, admissible_commands=many)
check("5a. truncation is stated explicitly", "first 40 of 60" in r_many)
check("5b. truncated render lists exactly 40 commands",
      sum(1 for line in r_many.splitlines() if line.startswith("  go to shelf ")) == 40)
r_exact = render_directive(InterventionType.REFLECT,
                           admissible_commands=[f"go to shelf {i}" for i in range(40)])
check("5c. exactly-40 case is not labelled truncated", "first 40 of" not in r_exact)

# 6. REPLAN is untouched by the new keyword.
p_none = render_directive(InterventionType.REPLAN, goal="g", steps_since_meaningful_change=2,
                          last_nonprogress_action="a", last_nonprogress_observation="o",
                          current_observation="c")
p_adm = render_directive(InterventionType.REPLAN, goal="g", steps_since_meaningful_change=2,
                         last_nonprogress_action="a", last_nonprogress_observation="o",
                         current_observation="c", admissible_commands=ADM)
check("6a. REPLAN ignores admissible_commands", p_none == p_adm)
check("6b. REPLAN still carries its v2 header", "[ACTIVE CONTROL DIRECTIVE: REPLAN]" in p_none)
check("6c. no admissible command leaks into REPLAN", "use desklamp 1" not in p_none)

# 7. Controller threads admissible_after through without affecting routing.
from ae.controllers.config import AEConfig  # noqa: E402
from ae.controllers.stateful_controller import StatefulController  # noqa: E402

cfg_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "configs", "controllers", "ae_full_final_nopir_v2_5.yaml"))
from ae.controllers.config import load_config  # noqa: E402
cfg = load_config(cfg_path)

c = StatefulController(cfg)
check("7a. starts with no recorded admissible commands",
      c.current_admissible_commands is None)
c.step(action="go to desk 1", observation="You arrive at desk 1.",
       admissible_before=["go to desk 1"], recent_actions=[], recent_observations=[],
       max_steps=50, is_think_action=False, admissible_after=ADM,
       legacy_termination_candidate=None, termination_suppressed=False,
       suppression_reason=None, reward=0, done=False, won=False)
check("7b. records admissible_after verbatim", c.current_admissible_commands == ADM)
check("7c. recorded list is a copy, not the caller's object",
      c.current_admissible_commands is not ADM)

# Routing must be unaffected: same controller state, same decision, whether
# or not admissible_after was supplied.
c_with = StatefulController(cfg)
c_without = StatefulController(cfg)
decisions_with = []
decisions_without = []
for i in range(12):
    for ctrl, sink, adm in ((c_with, decisions_with, ADM), (c_without, decisions_without, None)):
        d = ctrl.step(action="go to drawer 1", observation="Nothing happens.",
                      admissible_before=["go to drawer 1"], recent_actions=[], recent_observations=[],
                      max_steps=50, is_think_action=False, admissible_after=adm,
                      legacy_termination_candidate=None, termination_suppressed=False,
                      suppression_reason=None, reward=0, done=False, won=False)
        sink.append((d.get("intervention"), d.get("intervention_reason")))
check("7d. routing decisions identical with vs without admissible_after",
      decisions_with == decisions_without)

# 8. A think-step (admissible_after None) leaves the previous list intact
#    rather than rendering an empty/wrong list.
c2 = StatefulController(cfg)
c2.step(action="go to desk 1", observation="You arrive at desk 1.",
        admissible_before=["go to desk 1"], recent_actions=[], recent_observations=[],
        max_steps=50, is_think_action=False, admissible_after=ADM,
        legacy_termination_candidate=None, termination_suppressed=False,
        suppression_reason=None, reward=0, done=False, won=False)
c2.step(action="think: plan", observation="OK.",
        admissible_before=ADM, recent_actions=[], recent_observations=[],
        max_steps=50, is_think_action=True, admissible_after=None,
        legacy_termination_candidate=None, termination_suppressed=False,
        suppression_reason=None, reward=0, done=False, won=False)
check("8. think-step with no admissible_after clears to None (honest, not stale)",
      c2.current_admissible_commands is None)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
