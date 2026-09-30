from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from recon_rag.config import get_settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Local ONNX embeddings (no API key). bge models expect a query instruction, which fastembed adds."""

    def __init__(self, model: str, dim: int):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model)
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts, batch_size=128)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


class HashEmbedder:
    """Deterministic signed feature-hashing embedder for tests and offline runs. Not semantic."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
        for tok in toks + [f"{a} {b}" for a, b in zip(toks, toks[1:], strict=False)]:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == "hash":
        return HashEmbedder(s.embedding_dim)
    return FastEmbedder(s.embedding_model, s.embedding_dim)
