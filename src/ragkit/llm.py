"""Answer generators.

`OpenAICompatibleLLM` talks to any /v1/chat/completions endpoint: OpenAI, Groq,
Together, or a local Ollama / LM Studio / vLLM server.

`ExtractiveGenerator` is an offline stand-in that quotes the best-supported
sentences from the retrieved context. It never invents text, so it gives a
floor for answer quality and lets the whole pipeline run in CI without keys.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Protocol

import httpx

from ragkit.text import content_tokens, split_sentences

ABSTAIN = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context passages.
Rules:
- Every factual sentence must end with a citation like [1] or [2][3] pointing at the passages that support it.
- Keep the answer to 1-3 sentences and keep exact numbers, dates and amounts from the passages.
- If the passages do not contain the answer, reply with exactly: {ABSTAIN}
- Never use outside knowledge."""


def format_context(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))


class Generator(Protocol):
    name: str

    def generate(self, question: str, passages: list[str]) -> str: ...


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ):
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout
        self.name = f"llm[{self.model}]"

    def chat(self, messages: list[dict[str, str]]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={"model": self.model, "messages": messages, "temperature": self.temperature},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def generate(self, question: str, passages: list[str]) -> str:
        return self.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{format_context(passages)}\n\nQuestion: {question}"},
            ]
        )


class ExtractiveGenerator:
    """Pick the context sentences with the highest IDF-weighted overlap with the question."""

    name = "extractive"

    def __init__(self, max_sentences: int = 2, min_overlap: float = 0.15):
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, question: str, passages: list[str]) -> str:
        q = set(content_tokens(question))
        if not q or not passages:
            return ABSTAIN
        candidates = []  # (score, passage_idx, sentence)
        doc_freq: Counter[str] = Counter()
        bodies = [p.split("\n", 1)[-1] for p in passages]  # drop "Title > Section" header line
        for body in bodies:
            doc_freq.update(set(content_tokens(body)))
        idf = {t: math.log(1 + len(bodies) / (1 + doc_freq[t])) + 1.0 for t in q}
        total = sum(idf.values())
        for i, body in enumerate(bodies, start=1):
            for sent in split_sentences(body):
                toks = set(content_tokens(sent))
                score = sum(idf[t] for t in q & toks) / total
                # Small prior for higher-ranked passages.
                candidates.append((score + 0.02 * (len(bodies) - i), i, sent))
        candidates.sort(key=lambda c: -c[0])
        best = candidates[0]
        if best[0] < self.min_overlap:
            return ABSTAIN
        chosen = [best]
        for cand in candidates[1:]:
            if len(chosen) >= self.max_sentences:
                break
            if cand[1] == best[1] and cand[0] >= 0.6 * best[0]:
                chosen.append(cand)
        return " ".join(f"{sent.rstrip()} [{idx}]" for _, idx, sent in chosen)


CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(answer: str) -> list[int]:
    seen: list[int] = []
    for m in CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def is_abstention(answer: str) -> bool:
    a = answer.lower()
    return "i don't know" in a or "i do not know" in a or "not contain the answer" in a


def get_generator(name: str) -> Generator:
    if name == "extractive":
        return ExtractiveGenerator()
    if name == "llm":
        return OpenAICompatibleLLM()
    raise ValueError(f"Unknown generator {name!r}; choose extractive or llm")
