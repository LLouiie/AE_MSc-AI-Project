"""Re-exports hotpotqa_runs/llm.py so every ae/ module gets the same vLLM
client and the same global call_counter, instead of each baseline importing
its own copy via a separate sys.path hack (as alfworld_runs_ae/agents.py
currently does)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hotpotqa_runs"))

import llm as _llm  # noqa: E402
from llm import AnyOpenAILLM  # noqa: E402,F401


def call_counter() -> int:
    return _llm.call_counter
