"""Tokenisation helpers shared by BM25, abstention and the lexical metrics."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")

STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does doing for from get gets
    had has have having he her here hers him his how i if in into is it its just me my
    no nor not of on or our out over own she should so some such than that the their them
    then there these they this those through to too under until up very was we were what
    when where which while who whom why will with would you your yours am also any each
    much many more most other only same few both again further once all off against about
    above below between during before after s t don per""".split()
)


def _stem(token: str) -> str:
    """A deliberately tiny suffix stripper; good enough for a handbook-sized corpus."""
    for suffix in ("ing", "ies", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping numbers such as 15,000 or 2.5 intact."""
    return [t.replace(",", "") for t in _TOKEN_RE.findall(text.lower())]


def content_tokens(text: str) -> list[str]:
    """Stemmed tokens with stopwords removed."""
    return [_stem(t) for t in tokenize(text) if t not in STOPWORDS]


def numbers(text: str) -> set[str]:
    """Numeric facts in a string (e.g. '24', '15000', '11:00' -> '11', '00')."""
    return {t for t in tokenize(text) if any(ch.isdigit() for ch in t)}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]
