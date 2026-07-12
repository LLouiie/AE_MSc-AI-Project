"""
wiki_docstore.py — live Wikipedia retrieval via the official MediaWiki API.

Reproduces the Search/Lookup contract that hotpotqa_runs/agents.py's
ReactAgent.step() calls directly (self.docstore.search / self.docstore.lookup),
which is the same contract langchain's DocstoreExplorer(Wikipedia()) exposed in
the original Reflexion/ReAct repo (langchain/agents/react/base.py,
langchain/docstore/wikipedia.py, both installed under reflexion_hotpot as
langchain==0.0.162):

  - search(term) -> str
      * on a real page: caches the full page text, returns only the FIRST
        PARAGRAPH (langchain's DocstoreExplorer._summary == page_content
        split on blank lines, index 0) — matches prompts.py's instruction
        "Search[entity] ... returns the first paragraph if it exists."
      * on a missing page: returns "Could not find [X]. Similar: [...]"
        (mirrors langchain.docstore.wikipedia.Wikipedia.search's PageError
        branch, backed here by the MediaWiki opensearch endpoint).
  - lookup(term) -> str
      * raises ValueError if no prior successful search — agents.py's
        Lookup branch catches exactly ValueError.
      * scans every sentence of the FULL page text cached by the last
        successful search (not just the first paragraph shown to the
        agent) — matches DocstoreExplorer.lookup() scanning
        self.document.page_content (the whole article), not the summary.
      * one deliberate deviation from the original: langchain's
        DocstoreExplorer.lookup() matches whole PARAGRAPHS; this returns
        the next matching SENTENCE instead, per prompts.py's own wording
        ("Lookup[keyword], which returns the next sentence...") and this
        repo's existing DistractorDocstore convention.
      * new keyword -> lookup index resets to 0; repeating the same
        keyword advances to the next match; "No Results" / "No More
        Results" sentinels match the original DocstoreExplorer strings.
  - a fresh Search always resets Lookup state, so repeated lookups can
    never leak into a document searched before it.

Real network/API failures (timeout, non-2xx HTTP, connection errors) are
never swallowed into a look-alike "not found" result — they raise a typed
WikipediaAPIError subclass so callers can tell "the page doesn't exist"
apart from "the API call failed". There is no fallback to any local/
distractor data source anywhere in this module.
"""

import json
import os
import re
import time
from typing import List, Optional

import requests

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

_WHITESPACE_RE = re.compile(r"\s+")
_HEADING_RE = re.compile(r"^=+.*=+$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class WikipediaAPIError(Exception):
    """Base class for real MediaWiki API/network failures (not 'page missing')."""


class WikipediaTimeoutError(WikipediaAPIError):
    pass


class WikipediaForbiddenError(WikipediaAPIError):
    pass


class WikipediaHTTPError(WikipediaAPIError):
    pass


def _clean_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    paragraphs = []
    for part in parts:
        part = part.strip()
        if not part or _HEADING_RE.match(part):
            continue
        paragraphs.append(_clean_text(part))
    return paragraphs


def _split_sentences(text: str) -> List[str]:
    cleaned = _clean_text(text)
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]


class WikipediaDocstore:
    """Live, question-scoped Wikipedia docstore. No local/distractor context."""

    def __init__(
        self,
        user_agent: str,
        timeout: float = 15.0,
        max_retries: int = 2,
        log_path: Optional[str] = None,
    ):
        if not user_agent:
            raise ValueError("WikipediaDocstore requires a non-empty user_agent")
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.log_path = log_path

        self._last_page_title: Optional[str] = None
        self._last_page_sentences: List[str] = []
        self._lookup_str: Optional[str] = None
        self._lookup_index: int = -1

    # ── logging ──────────────────────────────────────────────────────────
    def _log(self, action: str, query: str, *, result: Optional[str] = None,
              error: Optional[str] = None, duration: float = 0.0) -> None:
        entry = {
            "ts": time.strftime("%F %T"),
            "action": action,
            "query": query,
            "duration_s": round(duration, 3),
        }
        if result is not None:
            entry["result"] = result[:300]
        if error is not None:
            entry["error"] = error
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── HTTP helpers ─────────────────────────────────────────────────────
    def _get(self, params: dict):
        headers = {"User-Agent": self.user_agent}
        last_exc = None
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = requests.get(
                    WIKI_API_URL, params=params, headers=headers, timeout=self.timeout
                )
            except requests.exceptions.Timeout as e:
                last_exc = e
                if attempt <= self.max_retries:
                    continue
                raise WikipediaTimeoutError(
                    f"Wikipedia API request timed out after {attempt} attempt(s) "
                    f"(timeout={self.timeout}s)"
                ) from e
            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt <= self.max_retries:
                    continue
                raise WikipediaHTTPError(
                    f"Wikipedia API request failed after {attempt} attempt(s): {e}"
                ) from e

            if resp.status_code == 403:
                raise WikipediaForbiddenError(
                    f"Wikipedia API returned 403 Forbidden "
                    f"(User-Agent used: '{self.user_agent}')"
                )
            if resp.status_code >= 400:
                raise WikipediaHTTPError(
                    f"Wikipedia API HTTP {resp.status_code}: {resp.text[:200]}"
                )
            return resp.json()
        raise WikipediaHTTPError(f"Wikipedia API request failed: {last_exc}")

    def _fetch_page(self, title: str) -> Optional[str]:
        """Return full plain-text extract, or None if the page is confirmed missing."""
        data = self._get({
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
        })
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        if "missing" in page:
            return None
        return page.get("extract", "") or ""

    def _opensearch(self, query: str) -> List[str]:
        data = self._get({
            "action": "opensearch",
            "search": query,
            "limit": 5,
            "format": "json",
        })
        return data[1] if isinstance(data, list) and len(data) > 1 else []

    # ── Agent-facing API ─────────────────────────────────────────────────
    def search(self, query: str) -> str:
        t0 = time.time()
        extract = self._fetch_page(query)

        if extract is None:
            similar = self._opensearch(query)
            self._last_page_title = None
            self._last_page_sentences = []
            self._lookup_str = None
            self._lookup_index = -1
            result = f"Could not find [{query}]. Similar: {similar}"
            self._log("search", query, result=result, duration=time.time() - t0)
            return result

        paragraphs = _split_paragraphs(extract)
        first_paragraph = paragraphs[0] if paragraphs else _clean_text(extract)

        self._last_page_title = query
        self._last_page_sentences = _split_sentences(extract)
        self._lookup_str = None
        self._lookup_index = -1

        self._log("search", query, result=first_paragraph, duration=time.time() - t0)
        return first_paragraph

    def lookup(self, keyword: str) -> str:
        if self._last_page_title is None:
            raise ValueError("Cannot lookup without a successful search first")

        t0 = time.time()
        kw = keyword.lower()
        if kw != self._lookup_str:
            self._lookup_str = kw
            self._lookup_index = 0
        else:
            self._lookup_index += 1

        matches = [s for s in self._last_page_sentences if kw in s.lower()]
        if not matches:
            result = "No Results"
        elif self._lookup_index >= len(matches):
            result = "No More Results"
        else:
            result = f"(Result {self._lookup_index + 1}/{len(matches)}) {matches[self._lookup_index]}"

        self._log("lookup", keyword, result=result, duration=time.time() - t0)
        return result
