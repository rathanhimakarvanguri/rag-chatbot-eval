"""Sparse (BM25), dense and hybrid (reciprocal rank fusion) retrievers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from ragkit.embeddings import Embedder, LSAEmbedder
from ragkit.ingest import Chunk
from ragkit.text import content_tokens


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1-based


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[Hit]: ...


def _top_k(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [Hit(chunks[i], float(scores[i]), r) for r, i in enumerate(order, start=1)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self._bm25 = BM25Okapi([content_tokens(c.render()) for c in chunks], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(content_tokens(query)))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return _top_k(self.chunks, self.scores(query), k)


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.embedder = embedder or LSAEmbedder()
        texts = [c.render() for c in chunks]
        self.embedder.fit(texts)
        self._matrix = self.embedder.encode(texts)
        self.name = f"dense[{self.embedder.name}]"

    def scores(self, query: str) -> np.ndarray:
        try:
            q = self.embedder.encode([query], is_query=True)  # type: ignore[call-arg]
        except TypeError:
            q = self.embedder.encode([query])
        return self._matrix @ q[0]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return _top_k(self.chunks, self.scores(query), k)


class HybridRetriever:
    """Reciprocal rank fusion: score(d) = sum_r w_r / (rrf_k + rank_r(d))."""

    def __init__(
        self,
        sparse: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),
        candidates: int = 20,
    ):
        self.sparse, self.dense = sparse, dense
        self.rrf_k, self.weights, self.candidates = rrf_k, weights, candidates
        self.chunks = sparse.chunks
        self.name = f"hybrid[bm25+{dense.embedder.name}]"

    def search(self, query: str, k: int = 5) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for weight, retriever in zip(self.weights, (self.sparse, self.dense)):
            for hit in retriever.search(query, self.candidates):
                fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + weight / (self.rrf_k + hit.rank)
                by_id[hit.chunk.id] = hit.chunk
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(by_id[cid], score, r) for r, (cid, score) in enumerate(ranked, start=1)]


def build_retriever(name: str, chunks: list[Chunk], embedder: Embedder | None = None) -> Retriever:
    if name == "bm25":
        return BM25Retriever(chunks)
    if name == "dense":
        return DenseRetriever(chunks, embedder)
    if name == "hybrid":
        return HybridRetriever(BM25Retriever(chunks), DenseRetriever(chunks, embedder))
    raise ValueError(f"Unknown retriever {name!r}; choose bm25, dense or hybrid")
