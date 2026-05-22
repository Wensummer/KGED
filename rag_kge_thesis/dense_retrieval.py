from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import re


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _lexical_similarity(query_text: str, doc_text: str) -> float:
    q_tokens = set(_tokenize(query_text))
    d_tokens = set(_tokenize(doc_text))
    if not q_tokens:
        return 0.0
    return len(q_tokens.intersection(d_tokens)) / max(1, len(q_tokens))


def _truncate(text: str, max_chars: int) -> str:
    clean = " ".join((text or "").strip().split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars]


@dataclass
class DenseRetrievalRuntime:
    available: bool
    model_name: str
    load_error: str


class DenseRetriever:
    """
    Dense retriever with graceful fallback.

    - Preferred backend: sentence-transformers
    - Fallback: lexical overlap score
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 24,
        max_chars: int = 1200,
    ):
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_chars = max(64, int(max_chars))

        self._model = None
        self._available: bool | None = None
        self._load_error: str = ""

    def _lazy_init(self) -> None:
        if self._available is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._available = True
            self._load_error = ""
        except Exception as exc:  # pragma: no cover - runtime dependency
            self._model = None
            self._available = False
            self._load_error = str(exc)

    @property
    def available(self) -> bool:
        self._lazy_init()
        return bool(self._available)

    @property
    def load_error(self) -> str:
        self._lazy_init()
        return self._load_error

    def runtime(self) -> DenseRetrievalRuntime:
        return DenseRetrievalRuntime(
            available=self.available,
            model_name=self.model_name,
            load_error=self.load_error,
        )

    def rank(self, query_text: str, docs: Iterable, top_k: int) -> List[Tuple[object, float]]:
        candidates = list(docs)
        if not candidates or top_k <= 0:
            return []

        if self.available and self._model is not None:
            doc_texts = [
                _truncate(f"{getattr(doc, 'title', '')}. {getattr(doc, 'text', '')}", self.max_chars)
                for doc in candidates
            ]
            query = _truncate(query_text, self.max_chars)
            try:
                q_emb = self._model.encode(
                    [query],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=1,
                )
                d_emb = self._model.encode(
                    doc_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=self.batch_size,
                )
                scores = d_emb @ q_emb[0]
                ranked = [(doc, float(score)) for doc, score in zip(candidates, scores)]
                ranked.sort(key=lambda x: x[1], reverse=True)
                return ranked[:top_k]
            except Exception:  # pragma: no cover - runtime dependency
                pass

        ranked = [
            (
                doc,
                _lexical_similarity(
                    query_text,
                    f"{getattr(doc, 'title', '')} {getattr(doc, 'text', '')}",
                ),
            )
            for doc in candidates
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
