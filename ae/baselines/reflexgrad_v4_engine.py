"""Shared inference-time dual-process engine for reflexgrad_v4 and
reflexion_only_reflexgrad_v4 (reproduction spec sections 3/4).

Phase A3 note: `loss_fn` and `trajectory_analyzer_fn` now also receive
`task` as their first argument (previously (policy, last3) / (last5)).
Appendix E's actual verbatim templates for both roles require
{task_description} -- see ae/baselines/reflexgrad_v4_prompts.py -- so the
call sites had to widen to pass it through. All mock tests updated to match.

Built independently of external/reflexgrad/reflexgrad_trial.py -- per spec
section 4.6 and the earlier implementation audit, that vendored file's
`stall_threshold=2` / `low_score_cutoff=5` (score<=5, not strict <4) /
2-step cooldown all diverge from the v4 paper's m=5 / strict-<4 / 5-step
values, and its routing is a keyword-heuristic system layered with
ablation flags, not this pseudocode's pure numeric state machine. This
module is a from-scratch implementation of spec section 4.3's pseudocode,
not a refactor of that file.

Phase A2.2 revision (see git history for A2/A2.1): fixes the TODO
completion *signal* itself, which A2.1 got structurally right (real
pending/active/done/failed status) but semantically wrong --
  (a) env_success now ONLY ends the episode. It is the whole-ALFWorld-task
      completion signal, not a per-TODO one, and no longer marks any TODO
      done -- conflating "the whole task is solved" with "this particular
      subgoal is solved" was exactly the bug (spec: "不得用它判断中间 TODO
      是否完成");
  (b) TODO completion is now driven by an explicitly injected
      todo_verifier_fn, called once per REAL env action (never on a parser
      failure, never when env_success is already True -- global success
      needs no extra verification). todo_max_attempts (default 4, an
      official-repo implementation value, NOT a number the v4 paper states
      in its text -- see ReflexGradV4Config's docstring) gates the
      failure/advance transition instead of a single verifier "no" ending a
      TODO immediately;
  (c) slow no longer auto-fails the active TODO. A stall is a *reason to
      suspect* the current subgoal, not proof it failed -- slow now only
      records the causal diagnosis as a failure_reasons entry, leaving
      status untouched. rollback_todo() remains a separate, manually-
      invokable method, not wired to any automatic keyword/heuristic
      trigger this phase.

Per-episode setup (once, before the first choose_action):
  decompose(task, initial_observation) -> todos
      [LLM call: role="decomposer", gated by config.use_task_decomposer;
       does NOT touch agent_steps, env_actions, or the 15-step budget --
       structurally impossible, since this method never calls choose_action
       or after_env_step]

Per real decision attempt:
  choose_action(task, observation) -> (parsed_action_or_None, raw_output)
      [LLM call: role="actor"; always increments agent_steps -- "every
       decision attempt, including ones that fail to parse into an action"]
  -> if parsed_action is None: caller calls after_parse_failure() --
       does NOT call the environment, evaluator, or todo_verifier; does NOT
       touch memory, env_actions, or any TODO's attempts/status; does NOT
       run cooldown/slow/fast/base routing at all this turn
  -> if parsed_action is not None: caller executes env.step(parsed_action)
       [owned by the future run_episode integration, NOT this engine], then
       calls after_env_step(...) below

after_env_step(task, observation, action, next_observation, env_success):
  1. env_actions += 1
  2. evaluator(task, observation, action, next_observation) -> int 0..10
                                            [LLM call: role="evaluator"]
  3. memory.append((obs, action, next_obs, score)); memory keeps last 10
  4. if there is an active TODO: attempts += 1; last_action = action
     (this happens on every real env action, regardless of env_success)
  5. if env_success: route="terminal"; STOP here -- no todo_verifier call
     (spec: "全局 env_success=True 时可以直接终止，不必额外验证"), no
     TODO status change from this signal, no cooldown/slow/fast/base
     evaluation this step (matches spec's "if env_success: break")
  6. elif there is an active TODO and a todo_verifier_fn is configured:
       verified = todo_verifier_fn(active_todo.content, observation, action,
                                    next_observation)         [LLM call:
                                    role="todo_verifier", counted separately]
       if verified: mark active TODO done, activate the next pending one
       elif active_todo.attempts >= config.todo_max_attempts: mark active
            TODO failed (reason recorded), activate the next pending one
       else: stays active -- more attempts are still allowed
  7. elif cooldown_remaining > 0: decrement it; route="cooldown"
     (slow AND fast are BOTH suppressed for the full cooldown window --
     spec: "cooldown 期间不再触发 slow/fast 更新")
  8. elif last `slow_window_m` scores in memory are ALL strictly <
     low_progress_threshold: SLOW route
       trajectory_analyzer -> causal_diagnoser -> plan_generator
       [LLM calls: role="trajectory_analyzer"/"causal_diagnoser"/"plan_generator"]
       base_policy = merge(base_policy, plan) -- plan is folded into the
       ONE field the actor reads, not stashed in a separate override;
       active_slow_plan = plan (kept only as a historical record -- see
       test_stored_slow_plan_is_not_a_permanent_override);
       if there is an active TODO, the causal_diagnoser's cause is appended
       to its failure_reasons -- status is NOT changed, NOT auto-advanced
       (spec: "slow 触发本身不得自动把当前 TODO 标为 failed");
       cooldown_remaining = cooldown_steps
  9. elif textgrad_enabled and env_actions % gradient_cadence_k == 0: FAST
       loss -> gradient -> optimizer
       [LLM calls: role="loss"/"gradient"/"optimizer"]
       base_policy = optimizer's returned policy (a full replacement, not a
       diff -- matches spec's role table: optimizer's output IS "更新后的
       完整自然语言 policy"). Nothing gates this once cooldown_remaining
       reaches 0, because slow no longer holds a separate always-shown
       override field.
  10. else: BASE route (no role calls at all)

Note step 6 (TODO verification) and steps 7-10 (cooldown/slow/fast/base
routing) are independent decisions on the SAME step -- a step can, e.g.,
both complete a TODO via the verifier AND separately be a "cooldown" step
for the router. They are not mutually exclusive the way slow/fast/base are.

Priority among slow/fast/base is enforced structurally by the elif-chain
order alone: slow and fast can never both fire in the same step, so a slow
plan and a TextGrad gradient are never averaged into one blended update
(spec: "不能把 slow plan 和 gradient 做平均").

rollback_todo() is a separate, manually-invokable method (not auto-called
by after_env_step, not wired to slow, not driven by keyword heuristics this
phase) -- spec: "rollback_todo 只在明确的结构化 rollback 信号下触发，不要
靠模糊关键词猜测。现阶段保留独立可调用接口即可." A future real integration
would decide when to call it based on a structured signal this phase does
not yet define.

reflexion_only_reflexgrad_v4 = reflexgrad_v4 with textgrad_enabled=False:
the FAST branch becomes permanently unreachable (falls through to BASE
instead) whenever textgrad is disabled -- SLOW/cooldown/memory/TODO
tracking are completely unaffected either way. This one config field is the
entire difference between the two published-anchor baselines;
ReflexGradV4Engine itself is never subclassed or duplicated for the two
configs (spec section 二: "使用同一个共享 v4 engine... 不得复制两套状态机代码").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

Transition = Tuple[str, str, str, int]  # (observation, action, next_observation, score)
RoleFn = Callable[..., object]
ActionParserFn = Callable[[str], Optional[str]]


class WorkingMemory:
    """Fixed-capacity FIFO of (observation, action, next_observation, score)
    transitions -- spec's working_memory_size=10. Oldest entry is evicted
    first once capacity is exceeded; append/keep-last-N is the only
    mutation this class supports (no random access mutation)."""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self._items: List[Transition] = []

    def append(self, item: Transition) -> None:
        self._items.append(item)
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity:]

    def scores(self) -> List[int]:
        return [it[3] for it in self._items]

    def last(self, n: int) -> List[Transition]:
        return self._items[-n:]

    def __len__(self) -> int:
        return len(self._items)

    def reset(self) -> None:
        self._items = []


@dataclass
class Todo:
    """One decomposed subgoal. Status lifecycle: pending -> active ->
    (done | failed), driven by todo_verifier_fn + todo_max_attempts (NOT by
    env_success -- see module docstring). A failed TODO's slot is taken
    over by the next pending one; rollback_todo() can reopen a done TODO
    and demote the current one back to pending."""
    content: str
    status: str = "pending"  # pending | active | done | failed
    attempts: int = 0
    last_action: Optional[str] = None
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class ReflexGradV4Config:
    """Field names/defaults mirror the reproduction spec's YAML-like config
    blocks (sections 3.2 and 4.1) verbatim, so a future config-loading layer
    can map 1:1 without renaming anything."""
    working_memory_size: int = 10
    slow_window_m: int = 5
    low_progress_threshold: int = 4       # strictly-< this counts as "low"
    cooldown_steps: int = 5
    gradient_cadence_k: int = 3
    reflexion_enabled: bool = True         # slow process; spec never disables
                                            # this for either published-anchor
                                            # config this phase builds
    textgrad_enabled: bool = True          # False => reflexion_only_reflexgrad_v4
    use_task_decomposer: bool = True       # spec section 3.3's explicit config flag
    # Number of unverified real attempts on a TODO before it's marked
    # failed and skipped. 4 is the CURRENT OFFICIAL ReflexGrad repo's
    # implementation value -- the v4 paper's text does not state this
    # number explicitly. Must be recorded as such in the manifest whenever
    # a real run uses it (not presented as a paper-confirmed hyperparameter).
    todo_max_attempts: int = 4


@dataclass
class RoleCallCounts:
    decomposer: int = 0
    actor: int = 0
    evaluator: int = 0
    loss: int = 0
    gradient: int = 0
    optimizer: int = 0
    trajectory_analyzer: int = 0
    causal_diagnoser: int = 0
    plan_generator: int = 0
    todo_verifier: int = 0

    def as_dict(self) -> dict:
        return {
            "decomposer": self.decomposer, "actor": self.actor, "evaluator": self.evaluator,
            "loss": self.loss, "gradient": self.gradient, "optimizer": self.optimizer,
            "trajectory_analyzer": self.trajectory_analyzer,
            "causal_diagnoser": self.causal_diagnoser, "plan_generator": self.plan_generator,
            "todo_verifier": self.todo_verifier,
        }

    def total(self) -> int:
        return sum(self.as_dict().values())


class ReflexGradV4Engine:
    """Shared dual-process controller for reflexgrad_v4 and
    reflexion_only_reflexgrad_v4. Every LLM "role" is a plain injected
    callable (see the ten `*_fn` constructor args: the nine from spec
    section 4.4's role table plus todo_verifier_fn) -- this phase locks the
    state machine, thresholds, and per-role call counts against
    mock/scripted callables only, never a real LLM, per explicit
    instruction. `action_parser_fn` is NOT an LLM role -- it's the
    deterministic text->action parser (mirrors
    alfworld_runs_ae/output_parser.py's role in react_reflact_anchor);
    defaults to the identity function so tests that never model parse
    failure keep working unchanged. `todo_verifier_fn` defaults to None,
    meaning TODO completion is simply never evaluated (todos stay
    "active" indefinitely) -- tests that don't care about TODO semantics
    at all can omit it without any side effects on the router."""

    def __init__(self, config: ReflexGradV4Config,
                 decomposer_fn: RoleFn, actor_fn: RoleFn, evaluator_fn: RoleFn,
                 loss_fn: RoleFn, gradient_fn: RoleFn, optimizer_fn: RoleFn,
                 trajectory_analyzer_fn: RoleFn, causal_diagnoser_fn: RoleFn,
                 plan_generator_fn: RoleFn,
                 action_parser_fn: Optional[ActionParserFn] = None,
                 todo_verifier_fn: Optional[RoleFn] = None):
        self.config = config
        self._decomposer_fn = decomposer_fn
        self._actor_fn = actor_fn
        self._evaluator_fn = evaluator_fn
        self._loss_fn = loss_fn
        self._gradient_fn = gradient_fn
        self._optimizer_fn = optimizer_fn
        self._trajectory_analyzer_fn = trajectory_analyzer_fn
        self._causal_diagnoser_fn = causal_diagnoser_fn
        self._plan_generator_fn = plan_generator_fn
        self._action_parser_fn: ActionParserFn = action_parser_fn or (lambda raw: raw)
        self._todo_verifier_fn: Optional[RoleFn] = todo_verifier_fn
        self.reset()

    def reset(self) -> None:
        """Full per-episode reset. Every mutable field the state machine
        touches is reset here -- nothing carries over from a previous
        episode (spec's explicit no-cross-episode-leakage requirement)."""
        self.memory = WorkingMemory(self.config.working_memory_size)
        self.todos: List[Todo] = []
        self.base_policy: str = ""
        self.active_slow_plan: Optional[str] = None
        self.cooldown_remaining: int = 0
        self.agent_steps: int = 0
        self.env_actions: int = 0
        self.calls = RoleCallCounts()
        self.route_log: List[str] = []

    # ---- decomposer / TODO tracking -------------------------------------

    def decompose(self, task: str, initial_observation: str) -> List[Todo]:
        """Called once per episode, before the first choose_action. Gated
        by config.use_task_decomposer; when disabled, todos stays empty and
        the decomposer LLM is never called. Structurally cannot touch
        agent_steps/env_actions/the 15-step budget -- this method never
        calls choose_action or after_env_step."""
        if not self.config.use_task_decomposer:
            self.todos = []
            return self.todos
        contents = self._decomposer_fn(task, initial_observation)
        self.calls.decomposer += 1
        self.todos = [Todo(content=c) for c in contents]
        if self.todos:
            self.todos[0].status = "active"
        return self.todos

    def active_todo(self) -> Optional[Todo]:
        for t in self.todos:
            if t.status == "active":
                return t
        return None

    def _activate_next_pending(self) -> Optional[Todo]:
        for t in self.todos:
            if t.status == "pending":
                t.status = "active"
                return t
        return None

    def _fail_and_advance_todo(self, todo: Todo, reason: str) -> None:
        todo.status = "failed"
        todo.failure_reasons.append(reason)
        self._activate_next_pending()

    def rollback_todo(self) -> Optional[Todo]:
        """Reopens the most recently DONE todo (back to active) and demotes
        whatever is currently active or failed back to pending, so it can
        be retried later. Not auto-called by after_env_step, not wired to
        slow -- see this module's docstring. Returns the re-activated Todo,
        or None if there was nothing to roll back to."""
        current = self.active_todo()
        if current is not None:
            current.status = "pending"
            current.last_action = None
        for todo in reversed(self.todos):
            if todo.status == "done":
                todo.status = "active"
                return todo
        return None

    # ---- action selection / parsing -------------------------------------

    def choose_action(self, task: str, observation: str) -> Tuple[Optional[str], str]:
        """Returns (parsed_action_or_None, raw_actor_output). Always
        increments agent_steps and the actor call count -- one decision
        attempt, successful or not. Caller decides which of
        after_env_step/after_parse_failure to call next based on whether
        parsed_action is None."""
        todo = self.active_todo()
        raw_output = self._actor_fn(task, observation, todo, self.base_policy,
                                     self.active_slow_plan, self.memory)
        self.calls.actor += 1
        self.agent_steps += 1
        parsed_action = self._action_parser_fn(raw_output)
        return parsed_action, raw_output

    def after_parse_failure(self) -> str:
        """Called when choose_action's parsed_action was None. No
        env.step() happened, so: evaluator and todo_verifier are not
        called, memory/env_actions/any TODO's attempts are not touched, and
        no cooldown/slow/fast/base routing decision is made this turn."""
        self.route_log.append("parse_failure")
        return "parse_failure"

    def after_env_step(self, task: str, observation: str, action: str,
                        next_observation: str, env_success: bool) -> str:
        """Called only when choose_action returned a real parsed_action and
        the caller has already executed env.step(action). Returns the
        route taken this step: "terminal", "cooldown", "slow", "fast", or
        "base" -- for test assertions and episode logging."""
        self.env_actions += 1

        score = self._evaluator_fn(task, observation, action, next_observation)
        self.calls.evaluator += 1
        self.memory.append((observation, action, next_observation, score))

        todo = self.active_todo()
        if todo is not None:
            todo.attempts += 1
            todo.last_action = action

        if env_success:
            # Whole-ALFWorld-task completion only -- never a per-TODO
            # signal (spec: "不得用它判断中间 TODO 是否完成"). No verifier
            # call needed either (spec: "全局 env_success=True 时可以直接
            # 终止，不必额外验证").
            route = "terminal"
            self.route_log.append(route)
            return route

        if todo is not None and self._todo_verifier_fn is not None:
            verified = self._todo_verifier_fn(todo.content, observation, action, next_observation)
            self.calls.todo_verifier += 1
            if verified:
                todo.status = "done"
                self._activate_next_pending()
            elif todo.attempts >= self.config.todo_max_attempts:
                self._fail_and_advance_todo(
                    todo, f"exceeded todo_max_attempts={self.config.todo_max_attempts} "
                          f"without verifier confirmation",
                )
            # else: stays active, more real attempts are still allowed

        cfg = self.config
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            route = "cooldown"

        elif self._is_low_progress_stall():
            last5 = self.memory.last(cfg.slow_window_m)
            analysis = self._trajectory_analyzer_fn(task, last5)
            self.calls.trajectory_analyzer += 1
            cause = self._causal_diagnoser_fn(analysis, self.base_policy)
            self.calls.causal_diagnoser += 1
            plan = self._plan_generator_fn(cause, self.base_policy)
            self.calls.plan_generator += 1
            self.base_policy = self._merge_policy_with_plan(self.base_policy, plan)
            self.active_slow_plan = plan  # historical record only, see module docstring
            # Slow is a reason to SUSPECT the active TODO, not proof it
            # failed -- record the diagnosis, do not change status or
            # advance (spec: "slow 触发本身不得自动把当前 TODO 标为 failed").
            still_active_todo = self.active_todo()
            if still_active_todo is not None:
                still_active_todo.failure_reasons.append(cause)
            self.cooldown_remaining = cfg.cooldown_steps
            route = "slow"

        elif cfg.textgrad_enabled and self.env_actions % cfg.gradient_cadence_k == 0:
            last3 = self.memory.last(3)
            loss = self._loss_fn(task, self.base_policy, last3)
            self.calls.loss += 1
            gradient = self._gradient_fn(loss, self.base_policy)
            self.calls.gradient += 1
            self.base_policy = self._optimizer_fn(self.base_policy, gradient)
            self.calls.optimizer += 1
            route = "fast"

        else:
            route = "base"

        self.route_log.append(route)
        return route

    def _merge_policy_with_plan(self, base_policy: str, plan: str) -> str:
        """Deterministic, not an LLM role (spec's role table has no "merge"
        entry). Folds the slow plan into the ONE field the actor reads, so
        a later fast/optimizer call (a full replacement) naturally
        supersedes it once cooldown ends -- nothing pins base_policy to the
        slow plan forever."""
        if not base_policy:
            return plan
        return base_policy + "\n" + plan

    def _is_low_progress_stall(self) -> bool:
        cfg = self.config
        if not cfg.reflexion_enabled:
            return False
        scores = self.memory.scores()
        window = cfg.slow_window_m
        if len(scores) < window:
            return False
        return all(s < cfg.low_progress_threshold for s in scores[-window:])
