"""Retrieve -> gate -> generate -> attach citations."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field

from ragkit.embeddings import Embedder, get_embedder
from ragkit.ingest import Chunk, load_corpus
from ragkit.llm import ABSTAIN, Generator, get_generator, is_abstention, parse_citations
from ragkit.retrievers import Hit, Retriever, build_retriever
from ragkit.text import content_tokens


@dataclass
class Citation:
    n: int
    chunk_id: str
    title: str
    section: str


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    support: float
    citations: list[Citation] = field(default_factory=list)
    contexts: list[Hit] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["contexts"] = [
            {"rank": h.rank, "score": round(h.score, 4), "chunk_id": h.chunk.id, "text": h.chunk.text}
            for h in self.contexts
        ]
        return d


class SupportScorer:
    """IDF-weighted share of the question's content words found in the retrieved passages.

    Rare words count for more, so a question about 'stock options' scores low against a
    corpus that never mentions them, even if 'company' appears everywhere.
    """

    def __init__(self, chunks: list[Chunk]):
        df: Counter[str] = Counter()
        for c in chunks:
            df.update(set(content_tokens(c.render())))
        n = len(chunks)
        self._idf = {t: math.log((n + 1) / (f + 0.5)) for t, f in df.items()}
        self._unseen_idf = math.log((n + 1) / 0.5)

    def __call__(self, question: str, hits: list[Hit]) -> float:
        q = set(content_tokens(question))
        if not q or not hits:
            return 0.0
        found: set[str] = set()
        for h in hits:
            found |= q & set(content_tokens(h.chunk.render()))
        weight = {t: self._idf.get(t, self._unseen_idf) for t in q}
        return sum(weight[t] for t in found) / sum(weight.values())


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        support_scorer: SupportScorer,
        k: int = 3,
        min_support: float = 0.0,
    ):
        self.retriever = retriever
        self.generator = generator
        self.support_scorer = support_scorer
        self.k = k
        self.min_support = min_support

    def answer(self, question: str) -> Answer:
        hits = self.retriever.search(question, self.k)
        support = self.support_scorer(question, hits)
        if support < self.min_support:
            return Answer(question, ABSTAIN, True, support, [], hits)
        text = self.generator.generate(question, [h.chunk.render() for h in hits])
        abstained = is_abstention(text)
        citations = [
            Citation(n, hits[n - 1].chunk.id, hits[n - 1].chunk.title, hits[n - 1].chunk.section)
            for n in parse_citations(text)
            if 1 <= n <= len(hits)
        ]
        return Answer(question, text, abstained, support, [] if abstained else citations, hits)


def build_pipeline(
    corpus: str = "data/corpus",
    retriever: str = "hybrid",
    generator: str = "extractive",
    embedder: str | Embedder = "lsa",
    k: int = 3,
    min_support: float = 0.30,
) -> RAGPipeline:
    chunks = load_corpus(corpus)
    emb = get_embedder(embedder) if isinstance(embedder, str) else embedder
    return RAGPipeline(
        build_retriever(retriever, chunks, emb),
        get_generator(generator),
        SupportScorer(chunks),
        k=k,
        min_support=min_support,
    )
