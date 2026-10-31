"""Embedding backends behind one small interface.

`SentenceTransformerEmbedder` is the real thing (e.g. BAAI/bge-small-en-v1.5).
`LSAEmbedder` (TF-IDF + truncated SVD) needs no model download, so tests and CI run
offline and deterministically. Both return L2-normalised vectors.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ragkit.text import content_tokens


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> Embedder: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class LSAEmbedder:
    """TF-IDF (word + char n-grams) projected with SVD. Offline, fast, deterministic."""

    name = "lsa"

    def __init__(self, dim: int = 128, seed: int = 0):
        self.dim = dim
        self.seed = seed
        self._word = TfidfVectorizer(
            tokenizer=content_tokens,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._svd: TruncatedSVD | None = None

    def _tfidf(self, texts: list[str], fit: bool):
        from scipy.sparse import hstack

        if fit:
            return hstack([self._word.fit_transform(texts), self._char.fit_transform(texts)])
        return hstack([self._word.transform(texts), self._char.transform(texts)])

    def fit(self, texts: list[str]) -> LSAEmbedder:
        x = self._tfidf(texts, fit=True)
        k = max(2, min(self.dim, x.shape[0] - 1, x.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(x)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("Call fit() before encode().")
        return _normalise(self._svd.transform(self._tfidf(texts, fit=False)))


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. Install with `pip install -e .[models]`."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", query_prefix: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        # bge models recommend an instruction prefix on queries only.
        self.query_prefix = (
            query_prefix
            if query_prefix is not None
            else ("Represent this sentence for searching relevant passages: " if "bge" in model_name else "")
        )

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        return self

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))


def get_embedder(name: str) -> Embedder:
    if name == "lsa":
        return LSAEmbedder()
    return SentenceTransformerEmbedder(name)
