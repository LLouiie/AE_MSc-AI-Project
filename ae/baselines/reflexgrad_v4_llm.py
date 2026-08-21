"""Real chat-API client + per-role response parsing for reflexgrad_v4 /
reflexion_only_reflexgrad_v4 (Phase A3).

Deliberately separate from ae/baselines/react_reflact_anchor.py's
QwenChatAnchorLLM: that class explicitly sets
chat_template_kwargs={"enable_thinking": False} because react_reflact_anchor's
reproduction target requires it. This module does the OPPOSITE on purpose --
the current official ReflexGrad repo's wrapper uses temperature=0.2 and never
sends any thinking-control parameter at all, so this client:
  - never sends enable_thinking / chat_template_kwargs of any kind;
  - never claims thinking is off (thinking_mode is recorded as
    "backend_default" in the manifest -- an honest "we didn't touch it",
    not a claim about what the server actually did with it);
  - uses temperature=0.2 (matching the official wrapper's behavior, NOT the
    paper's stated "default deterministic" text -- the mismatch itself is a
    disclosed reproduction gap, not something papered over).

No fallback, anywhere:
  - if the chat API call ultimately fails, ReflexGradAPIError propagates --
    callers must not catch-and-substitute a fake completion;
  - if a structured role's response can't be parsed into what that role is
    supposed to produce (evaluator's integer, todo_verifier's YES/NO,
    decomposer's TODO: lines), RoleParseError propagates -- never a guessed
    default value.

logical llm_calls (one per engine role-callback invocation, e.g.
ReflexGradV4Engine.calls.evaluator) vs physical api_attempts (one per actual
HTTP request, including retries) are tracked separately -- see
api_attempts_by_role. Retries are configurable (max_api_retries, default 0 =
no retries) and every attempt is counted, so a retried-then-succeeded call
shows logical_calls=1 but api_attempts>1 for that role.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Optional


class ReflexGradAPIError(RuntimeError):
    """The chat API call ultimately failed (after all configured retries).
    Must propagate -- no fallback completion, no silent retry-forever."""


class RoleParseError(RuntimeError):
    """A structured role's raw completion could not be parsed into what
    that role is supposed to produce. Must propagate -- never guessed."""


class ReflexGradChatLLM:
    """OpenAI-compatible chat.completions client for the seven "free-text or
    structured" LLM roles (evaluator, loss, gradient, optimizer,
    trajectory_analyzer, causal_diagnoser, plan_generator) plus decomposer
    and todo_verifier. actor is handled by ae/baselines/reflexgrad_v4_episode.py
    directly since it reuses alfworld_runs_ae/output_parser.py's existing
    parser (a different failure-handling contract -- see that module)."""

    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY",
                 temperature: float = 0.2, max_api_retries: int = 0, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_api_retries = max_api_retries
        self.client = client
        self.api_attempts_by_role: dict = defaultdict(int)
        self.last_request_payload: Optional[dict] = None
        self.last_raw_generation: Optional[str] = None

    def call(self, role: str, messages: List[dict], max_tokens: int) -> str:
        """One logical call for `role`. May make 1 + max_api_retries
        physical HTTP attempts; every attempt increments
        api_attempts_by_role[role] regardless of outcome."""
        last_err: Optional[Exception] = None
        attempts_made = 0
        for _ in range(1 + self.max_api_retries):
            attempts_made += 1
            self.api_attempts_by_role[role] += 1
            payload = {
                "model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": max_tokens,
            }
            self.last_request_payload = payload
            try:
                response = self.client.chat.completions.create(**payload)
            except Exception as e:  # noqa: BLE001 -- re-raised as ReflexGradAPIError below
                last_err = e
                continue
            text = response.choices[0].message.content or ""
            self.last_raw_generation = text
            return text
        raise ReflexGradAPIError(
            f"role={role!r} chat API call failed after {attempts_made} attempt(s) "
            f"(max_api_retries={self.max_api_retries}): {last_err}"
        ) from last_err


# ── per-role response parsers (fail loudly, never guess) ────────────────

_INT_RE = re.compile(r"-?\d+")


def parse_evaluator_score(raw: str) -> int:
    """Appendix E's evaluator prompt asks for "only the integer" in
    [0,10]. Extracts the first integer found; raises if none, or if it's
    out of the documented range -- never clamps or guesses."""
    m = _INT_RE.search(raw)
    if not m:
        raise RoleParseError(f"evaluator response contained no integer: {raw!r}")
    score = int(m.group(0))
    if not (0 <= score <= 10):
        raise RoleParseError(f"evaluator score {score} outside documented [0,10] range: {raw!r}")
    return score


def parse_todo_verifier_result(raw: str) -> bool:
    """The vendored verification prompt (task_todo_manager.py, adapted)
    asks for "ONLY: YES or NO". Mirrors the source's own fail-loud stance
    (it raises ValueError on an unparseable response, no fallback) --
    same behavior here, just as a RoleParseError."""
    text = raw.strip().upper()
    has_yes = "YES" in text
    has_no = "NO" in text
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    raise RoleParseError(f"todo_verifier response was not an unambiguous YES/NO: {raw!r}")


_TODO_LINE_RE = re.compile(r"^\s*TODO:\s*(.+)$", re.MULTILINE)


def parse_decomposer_todos(raw: str) -> List[str]:
    """The vendored decomposer prompt (task_todo_manager.py, adapted) asks
    for one "TODO: <goal>" line per subgoal. Raises if none are found --
    never silently falls back to a single catch-all TODO."""
    todos = [m.group(1).strip() for m in _TODO_LINE_RE.finditer(raw)]
    todos = [t for t in todos if t]
    if not todos:
        raise RoleParseError(f"decomposer response contained no 'TODO: ...' lines: {raw!r}")
    return todos
