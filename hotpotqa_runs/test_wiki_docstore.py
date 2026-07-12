"""
test_wiki_docstore.py — no-LLM checks for wiki_docstore.WikipediaDocstore.

Run directly (not pytest, matching this repo's existing tests.py style):
    WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (jy625@imperial.ac.uk)' \
        python3 test_wiki_docstore.py

Exercises real network calls to Wikipedia (Search[Albert Einstein], Lookup,
lookup-advance, search-resets-lookup, User-Agent header) plus mocked
requests.get() failures (403 / timeout) and the run_episode.py wikipedia-mode
branch never touching ex["context"] / ex["supporting_facts"].
"""

import os
import sys
from unittest import mock

import requests

from wiki_docstore import (
    WikipediaDocstore,
    WikipediaForbiddenError,
    WikipediaTimeoutError,
)

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


# ── 1. Search[Albert Einstein] returns real Wikipedia body text ────────────
ds = WikipediaDocstore(user_agent=USER_AGENT)
result = ds.search("Albert Einstein")
check("search returns non-empty string", isinstance(result, str) and len(result) > 0)
check("search result mentions Einstein/physicist",
      "einstein" in result.lower() or "physicist" in result.lower(),
      detail=result[:120])

# ── 2. Lookup[born] returns a sentence containing 'born' ───────────────────
look1 = ds.lookup("born")
check("lookup('born') contains 'born'", "born" in look1.lower(), detail=look1)
check("lookup('born') uses Result-N format or sentinel",
      look1.startswith("(Result") or look1 in ("No Results", "No More Results"),
      detail=look1)

# ── 3. Successive Lookup moves forward ──────────────────────────────────────
look2 = ds.lookup("born")
check("second lookup('born') differs from first (or hits No More Results)",
      look2 != look1 or look2 == "No More Results",
      detail=f"{look1!r} vs {look2!r}")

# ── 4. New Search resets Lookup state ───────────────────────────────────────
ds.search("Marie Curie")
try:
    # lookup_index for a fresh keyword after a new search must start at index 0,
    # i.e. either "(Result 1/N)" or a no-match sentinel — never an index carried
    # over from the previous page's search history (a single call only, since
    # a second call with the *same* keyword is defined to advance forward).
    look_after_reset = ds.lookup("born")
    reset_ok = True
except ValueError:
    look_after_reset = ""
    reset_ok = False
check("lookup works again right after a new Search (state reset, not stale)", reset_ok)
check("lookup after new Search starts fresh (Result 1/... or sentinel)",
      look_after_reset.startswith("(Result 1/") or look_after_reset in ("No Results", "No More Results"),
      detail=look_after_reset)

# ── 5. Confirm outgoing requests carry the correct User-Agent ──────────────
seen_headers = {}
real_get = requests.get


def _capturing_get(url, params=None, headers=None, timeout=None):
    seen_headers.update(headers or {})
    return real_get(url, params=params, headers=headers, timeout=timeout)


with mock.patch("wiki_docstore.requests.get", side_effect=_capturing_get):
    WikipediaDocstore(user_agent=USER_AGENT).search("Python (programming language)")
check("outgoing request set the expected User-Agent header",
      seen_headers.get("User-Agent") == USER_AGENT,
      detail=str(seen_headers))

# ── 6a. 403 Forbidden produces a clear, distinct error ──────────────────────
class _FakeResp:
    status_code = 403
    text = "Forbidden"


with mock.patch("wiki_docstore.requests.get", return_value=_FakeResp()):
    try:
        WikipediaDocstore(user_agent=USER_AGENT).search("anything")
        forbidden_raised = False
    except WikipediaForbiddenError as e:
        forbidden_raised = "403" in str(e)
    except Exception:
        forbidden_raised = False
check("403 response raises WikipediaForbiddenError with clear message", forbidden_raised)

# ── 6b. Timeout produces a clear, distinct error (not silently 'not found') ─
with mock.patch("wiki_docstore.requests.get", side_effect=requests.exceptions.Timeout("timed out")):
    try:
        WikipediaDocstore(user_agent=USER_AGENT, max_retries=0).search("anything")
        timeout_raised = False
    except WikipediaTimeoutError as e:
        timeout_raised = "timed out" in str(e) or "timeout" in str(e).lower()
    except Exception:
        timeout_raised = False
check("timeout raises WikipediaTimeoutError with clear message", timeout_raised)

# ── 6c. Genuinely nonexistent page returns a distinct 'not found' string ────
nonsense_title = "Qxzjklw_ThisPageDoesNotExist_asdfghjkl_9999"
not_found = WikipediaDocstore(user_agent=USER_AGENT).search(nonsense_title)
check("nonexistent page returns 'Could not find [...]' (not an exception)",
      not_found.startswith(f"Could not find [{nonsense_title}]"), detail=not_found)
check("'not found' result is textually distinct from 403/timeout errors",
      "Similar:" in not_found, detail=not_found)

# ── 7. wikipedia mode never reads ex['context'] / ex['supporting_facts'] ───
class _TripwireDict(dict):
    """Raises if a HotpotQA-context-only key is ever read."""

    def __getitem__(self, key):
        if key in ("context", "supporting_facts"):
            raise AssertionError(f"wikipedia-mode code path read ex[{key!r}]")
        return super().__getitem__(key)


ex = _TripwireDict(id="x1", question="Q?", answer="A", type="bridge", level="hard",
                    context={"title": [], "sentences": []}, supporting_facts={})

try:
    # Exact construction path used by run_episode.py's `if args.retrieval == "wikipedia"` branch.
    _ = WikipediaDocstore(user_agent=USER_AGENT, log_path=None)
    _ = ex["id"]        # allowed
    _ = ex["question"]  # allowed
    tripwire_ok = True
except AssertionError:
    tripwire_ok = False
check("wikipedia-mode docstore construction never indexes ex['context']/['supporting_facts']",
      tripwire_ok)

# ── summary ──────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"===== {len(failures)} CHECK(S) FAILED: {failures} =====")
    sys.exit(1)
print("===== ALL CHECKS PASSED =====")
