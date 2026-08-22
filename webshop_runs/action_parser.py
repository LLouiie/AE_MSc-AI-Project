"""Parse one WebShop action from a model response."""

import re


_ACTION_RE = re.compile(
    r"^(?:Action:\s*)*((search|click|think)\[.*\])\s*$",
    flags=re.IGNORECASE,
)


def parse_webshop_action(text):
    """Return a canonical command, or the untouched response if none exists."""
    raw = (text or "").strip()
    for line in raw.splitlines():
        match = _ACTION_RE.fullmatch(line.strip())
        if match:
            command = match.group(1)
            verb, rest = command.split("[", 1)
            return f"{verb.lower()}[{rest}"
    return raw
