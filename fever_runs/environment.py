import re, string, time, requests
from typing import List, Optional, Tuple


WIKI_API = "https://en.wikipedia.org/w/api.php"
FEVER_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


class WikiSearchDocstore:
    """Live Wikipedia search via MediaWiki API. No local index needed."""

    def __init__(self):
        self._last_page_sents: List[str] = []
        self._lookup_idx: int = 0

    def search(self, query: str) -> str:
        content = self._fetch_page(query)
        if content is not None:
            self._last_page_sents = _split_sentences(content)
            self._lookup_idx = 0
            return _trunc(content)

        titles = self._opensearch(query)
        if not titles:
            return f"Could not find [{query}]."

        content = self._fetch_page(titles[0])
        if content is not None:
            self._last_page_sents = _split_sentences(content)
            self._lookup_idx = 0
            return _trunc(content)

        return f"Could not find [{query}]. Similar: {titles[:5]}."

    def lookup(self, keyword: str) -> str:
        if not self._last_page_sents:
            return "No page has been searched yet."
        for i in range(self._lookup_idx, len(self._last_page_sents)):
            if keyword.lower() in self._last_page_sents[i].lower():
                self._lookup_idx = i + 1
                n_found = sum(1 for s in self._last_page_sents
                              if keyword.lower() in s.lower())
                result_n = sum(1 for j in range(i + 1)
                               if keyword.lower() in self._last_page_sents[j].lower())
                return f"(Result {result_n} / {n_found}) {self._last_page_sents[i]}"
        return f"No more results for '{keyword}' in current page."

    def _fetch_page(self, title: str) -> Optional[str]:
        params = {
            "action": "query", "titles": title,
            "prop": "extracts", "explaintext": 1,
            "format": "json", "redirects": 1,
        }
        for _ in range(3):
            try:
                r = requests.get(WIKI_API, params=params, timeout=15)
                pages = r.json()["query"]["pages"]
                page = next(iter(pages.values()))
                if "missing" in page:
                    return None
                extract = page.get("extract", "") or ""
                extract = re.sub(r"\n+", " ", extract).strip()
                return extract if extract else None
            except Exception:
                time.sleep(1)
        return None

    def _opensearch(self, query: str) -> List[str]:
        params = {
            "action": "opensearch", "search": query,
            "limit": 5, "format": "json",
        }
        for _ in range(3):
            try:
                r = requests.get(WIKI_API, params=params, timeout=10)
                return r.json()[1]
            except Exception:
                time.sleep(1)
        return []


class FEVEREnv:
    def __init__(self, claim: str, label: str, max_steps: int = 7):
        self.question = claim   # alias for agent compatibility
        self.claim = claim
        self.key = label
        self.max_steps = max_steps
        self.docstore = WikiSearchDocstore()
        self.reset()

    def reset(self):
        self.curr_step = 0
        self.terminated = False
        self.answer = ""
        self.docstore._last_page_sents = []
        self.docstore._lookup_idx = 0

    def step(self, action: str) -> Tuple[str, bool, bool, bool, int]:
        action_type, argument = parse_action(action)

        if action_type == "Finish":
            self.answer = argument
            observation = "Answer is CORRECT" if self.is_correct() else "Answer is INCORRECT"
            self.terminated = True

        elif action_type == "Search":
            try:
                observation = self.docstore.search(argument)
            except Exception as e:
                observation = f"Could not find that page, please try again."

        elif action_type == "Lookup":
            try:
                observation = self.docstore.lookup(argument)
            except Exception as e:
                observation = str(e)

        else:
            observation = ("Invalid Action. "
                           "Valid Actions are Search[<topic>], Lookup[<keyword>], "
                           "Finish[SUPPORTS], Finish[REFUTES], Finish[NOT ENOUGH INFO].")

        self.curr_step += 1
        return observation, self.is_correct(), self.is_terminated(), self.is_truncated(), self.curr_step

    def is_correct(self) -> bool:
        return normalize_label(self.answer) == normalize_label(self.key)

    def is_terminated(self) -> bool:
        return self.terminated

    def is_truncated(self) -> bool:
        return self.curr_step >= self.max_steps


def normalize_label(s: str) -> str:
    return s.strip().upper().replace("NOT_ENOUGH_INFO", "NOT ENOUGH INFO")


def parse_action(action: str):
    m = re.match(r'^(\w[\w\s]*)\[(.+)\]$', action.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _trunc(text: str, max_chars: int = 2000) -> str:
    return text[:max_chars] if len(text) > max_chars else text
