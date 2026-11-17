"""Split a prompt into attributable units (words / punctuation) and rebuild it.

Ollama is a black box that takes text, so the units we can switch on and off are
words, not sub-word tokens. Punctuation marks are their own units. Words keep
internal apostrophes/hyphens ("don't", "state-of-the-art"), numbers keep their
decimal point ("3.14"), and combining marks (Devanagari, Kannada, ...) stay
attached to their word.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Unit:
    text: str
    lead: str = ""  # whitespace that preceded this unit in the original prompt


_JOINERS = "'’-"


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("P", "S")


def _in_word(s: str, k: int) -> bool:
    ch = s[k]
    if ch.isspace():
        return False
    if not _is_punct(ch):
        return True
    if 0 < k < len(s) - 1:
        p, nx = s[k - 1], s[k + 1]
        p_word = not (p.isspace() or _is_punct(p))
        n_word = not (nx.isspace() or _is_punct(nx))
        if ch in _JOINERS and p_word and n_word:
            return True
        if ch in ".," and p.isdigit() and nx.isdigit():
            return True
    return False


def segment(text: str) -> Tuple[List[Unit], str]:
    """Return (units, trailing_whitespace)."""
    units: List[Unit] = []
    lead: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            lead.append(ch)
            i += 1
            continue
        if _in_word(text, i):
            j = i + 1
            while j < n and _in_word(text, j):
                j += 1
            units.append(Unit(text[i:j], "".join(lead)))
            i = j
        else:
            units.append(Unit(ch, "".join(lead)))
            i += 1
        lead = []
    return units, "".join(lead)


def render(units: Sequence[Unit], keep: Sequence[bool], trailing: str = "") -> str:
    """Rebuild the prompt keeping only the units whose flag is True."""
    out = "".join(u.lead + u.text for u, k in zip(units, keep) if k)
    if out and units and not units[0].lead:
        out = out.lstrip()
    return out + trailing
