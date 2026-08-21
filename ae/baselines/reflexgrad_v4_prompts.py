"""Prompt templates + provenance for the nine LLM roles ReflexGradV4Engine
calls (reproduction spec sections 3/4, Phase A3).

Per explicit instruction, provenance is recorded PER ROLE, not as one
blanket "all from Appendix E" claim -- exactly two of the nine roles
(evaluator, and the six loss/gradient/optimizer/trajectory_analyzer/
causal_diagnoser/plan_generator ones) are Appendix E verbatim; the other
three (decomposer, todo_verifier, initial_base_policy is not a "role" call
but is recorded here too since the reproduction instructions require it)
are NOT in Appendix E and are sourced/labeled individually below.

Appendix E source: https://arxiv.org/html/2511.14584v4 (section "Appendix E
Full LLM prompts (reproducibility)", anchors A5.SS0.SSS0.Px1-Px8), fetched
2026-07-29. Appendix E's own Px7 ("Reflexion Stage 4 (Cooldown Activator)")
is explicitly documented in the paper as "Deterministic; no LLM call" --
this confirms ReflexGradV4Engine's plain cooldown countdown (no LLM call)
already matches the paper, it is not a missing ninth role.

Official repo source (for decomposer/todo_verifier/initial_base_policy):
https://github.com/qpiai/reflexgrad, vendored at external/reflexgrad/,
commit 1fd292a (`git -C external/reflexgrad rev-parse --short HEAD`).

actor has NO identifiable single canonical prompt in the official repo:
external/reflexgrad/reflexgrad_trial.py is ~15,000 lines of accreted,
ablation-specific action-prompt-building functions (vision-first paths,
GUI vs. text-env branches, multiple "next action" extraction regexes at
different line ranges) -- confirmed by grep during this phase, consistent
with the same accretion pattern the earlier implementation audit already
flagged for this file's router logic. Picking one of several candidates
with no confident way to verify which (if any) corresponds to the reported
ALFWorld headline numbers would be guessing, not sourcing -- so the actor
prompt below is a LOCAL DEFINITION, disclosed as such, not attributed to
the official repo or to Appendix E.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

REFLEXGRAD_PAPER_URL = "https://arxiv.org/html/2511.14584v4"
REFLEXGRAD_PAPER_SECTION = "Appendix E Full LLM prompts (reproducibility)"
REFLEXGRAD_OFFICIAL_REPO = "https://github.com/qpiai/reflexgrad"
REFLEXGRAD_OFFICIAL_REPO_COMMIT = "1fd292a"


@dataclass
class PromptProvenance:
    role: str
    source_type: str  # "paper_appendix_e" | "official_repo" | "local_definition"
    source_ref: str    # URL/anchor, or "repo@commit:file:function"
    verbatim: bool      # False if any wording/structure was adapted
    adaptation_note: Optional[str]
    sha256: str

    def as_dict(self) -> dict:
        return {
            "role": self.role, "source_type": self.source_type, "source_ref": self.source_ref,
            "verbatim": self.verbatim, "adaptation_note": self.adaptation_note,
            "prompt_sha256": self.sha256,
        }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ══════════════════════ Appendix E verbatim (7 roles) ══════════════════════
# Text below is the get_text() extraction of arxiv.org/html/2511.14584v4's
# Appendix E anchors A5.SS0.SSS0.Px1 (loss) through Px8 (evaluator), with
# only the doubled Unicode-then-LaTeX-source math rendering artifact from
# HTML->text extraction cleaned up (e.g. the raw extraction literally reads
# "loss text ℓt\\ell_{t}" -- the math element's rendered glyph followed
# immediately by its own alt-text LaTeX source concatenated by the browser's
# text flattening, not two different things in the original). Variable
# placeholders ({task_description}, {policy_pi}, etc.) are exactly as they
# appear in the paper's own bolded-in-source template text.

LOSS_PROMPT_TEMPLATE = (
    "“You are evaluating an agent’s recent actions on task {task_description}. "
    "The agent’s current policy is: {policy_pi}. The last {k} steps produced these "
    "(observation, action, next-observation, score) tuples: {tuples}. Compare what the "
    "policy expected against what actually happened. Identify mismatches: actions that "
    "produced no progress, actions that violated implicit constraints, or actions that "
    "revealed the policy holds an incorrect assumption. Output a structured loss text "
    "ℓ_t that names each mismatch concretely.”"
)

GRADIENT_PROMPT_TEMPLATE = (
    "“Given the loss text {loss_ell} computed against the current policy {policy_pi}, "
    "propose a textual gradient g_t: a targeted critique describing how the policy should "
    "change to reduce the loss. The gradient should be specific and actionable, not generic "
    "advice. It is analogous to ∂ℓ/∂π in TextGrad’s formal sense. "
    "Output the gradient text only.”"
)

OPTIMIZER_PROMPT_TEMPLATE = (
    "“You will revise the natural-language policy by applying a textual gradient. "
    "The current policy is: {policy_pi}. The gradient (proposed change) is: {gradient_g}. "
    "Produce the revised policy. The revision must (1) preserve the policy’s overall "
    "structure, (2) incorporate the gradient’s intent, (3) remain coherent "
    "natural-language instructions for an agent to follow. Output the revised policy "
    "only.”"
)

TRAJECTORY_ANALYZER_PROMPT_TEMPLATE = (
    "“Review the agent’s recent trajectory on task {task_description}. The window "
    "contains the last {m} steps with (obs, action, next-obs, score) tuples: {window_W}. "
    "Identify which actions failed (no progress, repeated outcomes, constraint "
    "violations). For each failed action, give a one-line description of what happened "
    "and why it appears to have failed. Output a structured failed-action list.”"
)

CAUSAL_DIAGNOSER_PROMPT_TEMPLATE = (
    "“You are diagnosing a stall in an agent’s progress. The trajectory analysis "
    "is: {trajectory_analysis}. The agent’s current policy is: {policy_pi}. Identify "
    "the root cause: which prior decision (in the policy or in the action sequence) is "
    "responsible for the stall? Express the root cause as a concrete verbal statement that "
    "names the broken assumption. This statement is the causal trace d_t. Output the "
    "causal trace only.”"
)

PLAN_GENERATOR_PROMPT_TEMPLATE = (
    "“Given the causal trace {causal_trace_d} and the current policy {policy_pi}, "
    "generate a plan ρ_t consisting of 1–3 corrective sub-goals. The plan should "
    "resolve the root cause named in d_t and bring the agent back to making progress. Each "
    "sub-goal should be a concrete, executable instruction. Output the plan as a numbered "
    "list.”"
)

EVALUATOR_PROMPT_TEMPLATE = (
    "“Score the agent’s progress on task {task_description} for the most recent "
    "step. Inputs: previous observation {o_t}, action taken {a_t}, resulting observation "
    "{o_t_plus_1}. Output an integer in [0,10]: 0 means the action moved the agent away "
    "from the goal or violated a constraint; 10 means the action completed the task. "
    "Output only the integer.”"
)

_APPENDIX_E_TEMPLATES = {
    "loss": LOSS_PROMPT_TEMPLATE,
    "gradient": GRADIENT_PROMPT_TEMPLATE,
    "optimizer": OPTIMIZER_PROMPT_TEMPLATE,
    "trajectory_analyzer": TRAJECTORY_ANALYZER_PROMPT_TEMPLATE,
    "causal_diagnoser": CAUSAL_DIAGNOSER_PROMPT_TEMPLATE,
    "plan_generator": PLAN_GENERATOR_PROMPT_TEMPLATE,
    "evaluator": EVALUATOR_PROMPT_TEMPLATE,
}


# ══════════════════ Official repo, adapted (2 roles + seed policy) ══════════
# external/reflexgrad/task_todo_manager.py, commit 1fd292a.

# TaskTodoManager.initialize_from_task's trial_num==0 branch (lines ~90-140):
# kept the core decomposition instruction and output format verbatim;
# dropped the "SUCCESSFUL APPROACHES FROM SIMILAR TASKS" cross-env-learning
# block (similar_todo_suggestions) since this engine has no cross-episode
# memory feature this phase, and the GUI/computer-task-specific bullets
# (pixel coordinates, dialog-blocking, keyboard shortcuts) since those
# don't apply to ALFWorld text actions.
DECOMPOSER_PROMPT_TEMPLATE = """Decompose this task into sequential subgoals.

TASK: {task_description}
CURRENT STATE: {initial_observation}

Requirements:
1. Use ACTION VERBS (what to DO, not what state to BE IN)
2. Keep goals HIGH-LEVEL (e.g., "Cool the object", not "Put object in fridge")
3. Goals should be UNIVERSAL (work in any environment)
4. Each subgoal must be achievable before moving to next
5. Final subgoal completes the entire task
6. Use ONLY objects/concepts mentioned in the task above

Format: Start each line with "TODO: " followed by the high-level goal

Generate 3-8 subgoals:"""

# TaskTodoManager._verify_subgoal_with_llm's verification_prompt (lines
# ~414-421): wording kept verbatim; argument order remapped from the
# source's (subgoal, prev_obs, curr_obs, action) to this engine's
# todo_verifier_fn(current_todo, prev_obs, action, next_obs) call
# signature -- the textual roles (SUBGOAL/PREVIOUS STATE/ACTION TAKEN/
# CURRENT STATE) are unchanged, only which positional argument fills which
# textual slot was re-mapped to match our call site.
TODO_VERIFIER_PROMPT_TEMPLATE = """Determine if this subgoal is achieved based on observable state changes.

SUBGOAL: {subgoal}
PREVIOUS STATE: {prev_obs}
ACTION TAKEN: {action}
CURRENT STATE: {curr_obs}

Question: Based on the state change from PREVIOUS to CURRENT, is the SUBGOAL now observably achieved?

Answer ONLY with: YES or NO"""

# reflexgrad_trial.py line 4120 (`universal_base_policy = ...`), verbatim.
INITIAL_BASE_POLICY = "Select actions that make measurable progress toward completing the stated task."


# ══════════════════════ Local definition (1 role) ═══════════════════════════
# actor: zero-shot (no ICL example, per reproduction spec section one's
# "zero-shot" requirement for reflexgrad_v4/reflexion_only_reflexgrad_v4).
# Output format (Thought:/Action:) matches this repo's own shared parser
# (alfworld_runs_ae/output_parser.py::parse_agent_output), reused as-is
# rather than inventing a second parser -- see reflexgrad_v4_episode.py.
ACTOR_PROMPT_TEMPLATE = """You are an agent acting in a household environment. Respond with exactly two lines:
Thought: <one concise reasoning step>
Action: <one environment command>

Task: {task_description}
Current subgoal: {active_todo}
Current policy: {policy}
{slow_plan_block}
Observation: {observation}
"""


def build_actor_prompt(task_description: str, active_todo: Optional[str], policy: str,
                        active_slow_plan: Optional[str], observation: str) -> str:
    slow_plan_block = f"Recent corrective plan: {active_slow_plan}\n" if active_slow_plan else ""
    return ACTOR_PROMPT_TEMPLATE.format(
        task_description=task_description,
        active_todo=active_todo or "(none)",
        policy=policy or "(none yet)",
        slow_plan_block=slow_plan_block,
        observation=observation,
    )


# ══════════════════════════ provenance registry ═════════════════════════════

def _provenance_for_role(role: str) -> PromptProvenance:
    if role in _APPENDIX_E_TEMPLATES:
        text = _APPENDIX_E_TEMPLATES[role]
        return PromptProvenance(
            role=role, source_type="paper_appendix_e",
            source_ref=f"{REFLEXGRAD_PAPER_URL}#{REFLEXGRAD_PAPER_SECTION}",
            verbatim=True, adaptation_note=None, sha256=_sha256(text),
        )
    if role == "decomposer":
        return PromptProvenance(
            role=role, source_type="official_repo",
            source_ref=f"{REFLEXGRAD_OFFICIAL_REPO}@{REFLEXGRAD_OFFICIAL_REPO_COMMIT}:"
                       f"task_todo_manager.py::TaskTodoManager.initialize_from_task (trial_num==0 branch)",
            verbatim=False,
            adaptation_note=(
                "Dropped the cross-env 'SUCCESSFUL APPROACHES FROM SIMILAR TASKS' block "
                "(no cross-episode memory feature this phase) and the GUI/computer-task "
                "bullets (not applicable to ALFWorld text actions). Core decomposition "
                "instruction and output format kept verbatim."
            ),
            sha256=_sha256(DECOMPOSER_PROMPT_TEMPLATE),
        )
    if role == "todo_verifier":
        return PromptProvenance(
            role=role, source_type="official_repo",
            source_ref=f"{REFLEXGRAD_OFFICIAL_REPO}@{REFLEXGRAD_OFFICIAL_REPO_COMMIT}:"
                       f"task_todo_manager.py::TaskTodoManager._verify_subgoal_with_llm",
            verbatim=False,
            adaptation_note=(
                "Wording verbatim; argument order remapped from source's "
                "(subgoal, prev_obs, curr_obs, action) to this engine's "
                "todo_verifier_fn(current_todo, prev_obs, action, next_obs) call signature."
            ),
            sha256=_sha256(TODO_VERIFIER_PROMPT_TEMPLATE),
        )
    if role == "initial_base_policy":
        return PromptProvenance(
            role=role, source_type="official_repo",
            source_ref=f"{REFLEXGRAD_OFFICIAL_REPO}@{REFLEXGRAD_OFFICIAL_REPO_COMMIT}:"
                       f"reflexgrad_trial.py:4120 (universal_base_policy)",
            verbatim=True, adaptation_note=None, sha256=_sha256(INITIAL_BASE_POLICY),
        )
    if role == "actor":
        return PromptProvenance(
            role=role, source_type="local_definition",
            source_ref="ae/baselines/reflexgrad_v4_prompts.py::ACTOR_PROMPT_TEMPLATE",
            verbatim=False,
            adaptation_note=(
                "No single canonical actor prompt could be confidently identified in "
                "external/reflexgrad/reflexgrad_trial.py (~15,000 lines of accreted, "
                "ablation-specific action-prompt variants across vision/GUI/text-env "
                "branches) -- see module docstring. Written locally for this phase: "
                "zero-shot, Thought:/Action: two-line format matching this repo's shared "
                "alfworld_runs_ae/output_parser.py parser."
            ),
            sha256=_sha256(ACTOR_PROMPT_TEMPLATE),
        )
    raise ValueError(f"unknown role: {role!r}")


PROMPT_PROVENANCE = {
    role: _provenance_for_role(role)
    for role in (
        "loss", "gradient", "optimizer", "trajectory_analyzer", "causal_diagnoser",
        "plan_generator", "evaluator", "decomposer", "todo_verifier",
        "initial_base_policy", "actor",
    )
}


def manifest_prompt_provenance() -> dict:
    """What run_alfworld_anchor.py writes into each run's manifest -- one
    entry per role, never a blanket 'all from Appendix E' claim."""
    return {role: p.as_dict() for role, p in PROMPT_PROVENANCE.items()}
