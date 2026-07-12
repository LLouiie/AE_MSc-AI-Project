"""
test_episode_integration_mock.py — mocked-LLM integration test for the
wikipedia-retrieval episode path (run_episode.py --retrieval wikipedia).

Exercises the REAL production pieces except the LLM:
  - the official hotpot-qa-distractor-sample.joblib (first row only)
  - a REAL WikipediaDocstore hitting the live MediaWiki API
  - a REAL ReactAgent (imported from agents.py, untouched)
a FakeLLM feeds ReactAgent a fixed Search -> Lookup -> Finish script instead
of calling vLLM, so this test never needs a running LLM server.

Run from hotpotqa_runs/:
    WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (jy625@imperial.ac.uk)' \
        python3 test_episode_integration_mock.py

This file only adds a test; it does not modify run_episode.py, agents.py,
or wiki_docstore.py.
"""

import os
import sys

import joblib

from agents import ReactAgent
from wiki_docstore import WikipediaDocstore

DATA_PATH = "/rds/general/user/jy625/home/projects/AE/data/hotpotqa/hotpot-qa-distractor-sample.joblib"

USER_AGENT = os.environ.get("WIKIPEDIA_USER_AGENT", "").strip()
if not USER_AGENT:
    raise SystemExit(
        "ERROR: set WIKIPEDIA_USER_AGENT before running this test, e.g.\n"
        "  export WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (jy625@imperial.ac.uk)'"
    )

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


class FakeLLM:
    """Returns a fixed sequence of Thought/Action strings; never calls vLLM."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self._responses.pop(0)


# ── 1. Read only the first question from the official joblib ───────────────
df = joblib.load(DATA_PATH)
row = df.iloc[0].to_dict()

# Physically remove context/supporting_facts right away: wikipedia-mode code
# must never see them, so they can't leak into anything below even by
# accident (unlike a mere convention, a missing key can't be misused).
row.pop("context")
row.pop("supporting_facts")

question_id = row["id"]
question_text = row["question"]
gold_answer = row["answer"]  # stored only, never read again below

# ── 2. Real WikipediaDocstore (live MediaWiki API) ──────────────────────────
docstore = WikipediaDocstore(user_agent=USER_AGENT)

call_log = []
_orig_search, _orig_lookup = docstore.search, docstore.lookup


def _tracked_search(q):
    call_log.append(("search", q))
    return _orig_search(q)


def _tracked_lookup(k):
    call_log.append(("lookup", k))
    return _orig_lookup(k)


docstore.search = _tracked_search
docstore.lookup = _tracked_lookup

# ── 3. FakeLLM scripted to drive Search -> Lookup -> Finish ─────────────────
# One (Thought, Action) pair per step; ReactAgent.step() calls the LLM twice
# per step, so 3 steps => 6 scripted responses in order.
fake_llm = FakeLLM([
    "I should search Wikipedia for Albert Einstein.",
    "Search[Albert Einstein]",
    "I should look up his birth in the article.",
    "Lookup[born]",
    "I have enough information to answer now.",
    "Finish[Albert Einstein]",
])

# ── 4/5. Real ReactAgent, online_feedback=False, no vLLM/real LLM involved ──
agent = ReactAgent(
    question=question_text,
    key=gold_answer,           # stored only; never read by this test
    docstore=docstore,
    react_llm=fake_llm,
    online_feedback=False,
)
agent.run(reset=True)

# ── 6. Assertions ────────────────────────────────────────────────────────────
check("FakeLLM was driven exactly 6 times (3 steps x Thought+Action)",
      len(fake_llm.calls) == 6, detail=str(len(fake_llm.calls)))

check("agent called WikipediaDocstore.search then .lookup, in order",
      call_log == [("search", "Albert Einstein"), ("lookup", "born")],
      detail=str(call_log))

check("WikipediaDocstore actually resolved a real page (Search succeeded)",
      docstore._last_page_title == "Albert Einstein",
      detail=str(docstore._last_page_title))

traj = agent.scratchpad
check("trajectory contains Thought/Action/Observation markers",
      "Thought 1:" in traj and "Action 1:" in traj and "Observation 1:" in traj
      and "Thought 2:" in traj and "Action 2:" in traj and "Observation 2:" in traj
      and "Thought 3:" in traj and "Action 3:" in traj and "Observation 3:" in traj,
      detail=traj[:200])

obs_blocks = traj.split("Observation ")
search_obs = obs_blocks[1] if len(obs_blocks) > 1 else ""
lookup_obs = obs_blocks[2] if len(obs_blocks) > 2 else ""
check("Search observation looks like real Wikipedia content (not an error)",
      ("einstein" in search_obs.lower() or "physicist" in search_obs.lower())
      and "could not find" not in search_obs.lower(),
      detail=search_obs[:200])
check("Lookup observation is a real sentence containing 'born'",
      "born" in lookup_obs.lower() and "no results" not in lookup_obs.lower(),
      detail=lookup_obs[:200])

check("agent finished normally (Finish reached, not halted)",
      agent.is_finished() and agent.step_n <= agent.max_steps)
check("agent.answer captured from the scripted Finish action",
      agent.answer == "Albert Einstein", detail=agent.answer)

check("scratchpad never contains 'Answer is CORRECT'",
      "Answer is CORRECT" not in traj)
check("scratchpad never contains 'Answer is INCORRECT'",
      "Answer is INCORRECT" not in traj)

check("question text passed to the agent matches the joblib row",
      agent.question == question_text)
check("row no longer has context/supporting_facts (never available to leak)",
      "context" not in row and "supporting_facts" not in row)
check("scripted LLM prompts never contain the gold answer "
      "(gold wasn't used to control the agent's process)",
      all(gold_answer not in p for p in fake_llm.calls))

print()
if failures:
    print(f"===== {len(failures)} CHECK(S) FAILED: {failures} =====")
    sys.exit(1)
print("===== ALL CHECKS PASSED =====")
print(f"question_id={question_id!r}  question={question_text!r}")
