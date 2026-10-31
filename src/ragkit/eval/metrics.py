"""Retrieval, answer-quality, grounding and abstention metrics.

All lexical metrics are deterministic so results are reproducible in CI. An optional
LLM judge (see `llm_judge_faithfulness`) gives a stronger faithfulness signal when a
model endpoint is configured.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ragkit.llm import CITATION_RE
from ragkit.text import content_tokens, numbers, split_sentences, tokenize

# --------------------------------------------------------------------- retrieval


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    return float(any(r in gold for r in retrieved[:k]))


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    gains = [1.0 if r in gold else 0.0 for r in dict.fromkeys(retrieved[:k])]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(set(gold)), k)))
    return dcg / ideal if ideal else 0.0


# --------------------------------------------------------------------- answers


def _strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def token_f1(prediction: str, reference: str) -> float:
    p = Counter(tokenize(_strip_citations(prediction)))
    r = Counter(tokenize(reference))
    overlap = sum((p & r).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def key_fact_recall(prediction: str, reference: str) -> float:
    """Share of the reference's numbers (days, amounts, limits) present in the answer.

    For policy QA the numbers are the facts that matter; a fluent answer with the wrong
    number is the most damaging kind of hallucination. Returns 1.0 if the reference has
    no numbers.
    """
    ref = numbers(reference)
    if not ref:
        return 1.0
    return len(ref & numbers(_strip_citations(prediction))) / len(ref)


def faithfulness(answer: str, contexts: list[str], threshold: float = 0.6) -> float:
    """Lexical grounding: share of answer sentences whose content words (and every number)
    appear in the retrieved contexts. 1.0 means every sentence is supported."""
    sentences = [s for s in split_sentences(_strip_citations(answer)) if content_tokens(s)]
    if not sentences:
        return 1.0
    ctx_tokens = set(content_tokens(" ".join(contexts)))
    ctx_numbers = numbers(" ".join(contexts))
    supported = 0
    for s in sentences:
        toks = set(content_tokens(s))
        coverage = len(toks & ctx_tokens) / len(toks)
        if coverage >= threshold and numbers(s) <= ctx_numbers:
            supported += 1
    return supported / len(sentences)


def citation_precision(cited_sections: list[str], gold: list[str]) -> float:
    if not cited_sections:
        return 0.0
    return sum(c in gold for c in cited_sections) / len(cited_sections)


# --------------------------------------------------------------------- abstention


def abstention_report(answerable: list[bool], abstained: list[bool]) -> dict[str, float]:
    """Treat 'abstain' as the positive class: precision = abstentions that were correct."""
    tp = sum(a and not ans for ans, a in zip(answerable, abstained))
    fp = sum(a and ans for ans, a in zip(answerable, abstained))
    fn = sum((not a) and (not ans) for ans, a in zip(answerable, abstained))
    n_ans = sum(answerable)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "abstain_precision": precision,
        "abstain_recall": recall,
        "abstain_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_abstain_rate": fp / n_ans if n_ans else 0.0,
    }


# --------------------------------------------------------------------- LLM judge


JUDGE_PROMPT = """You are grading whether an answer is fully supported by the context.
Reply with a single number between 0 and 1: the fraction of the answer's claims that the
context directly supports. Reply with the number only.

Context:
{context}

Answer:
{answer}"""


def llm_judge_faithfulness(llm, answer: str, contexts: list[str]) -> float:
    """Ask a model to grade grounding. `llm` must expose .chat(messages) -> str."""
    reply = llm.chat([{"role": "user", "content": JUDGE_PROMPT.format(context="\n\n".join(contexts), answer=answer)}])
    match = re.search(r"\d*\.?\d+", reply)
    return min(1.0, max(0.0, float(match.group()))) if match else 0.0
