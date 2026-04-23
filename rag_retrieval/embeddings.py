"""Embedding providers with a deterministic local fallback for tests and demos."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from .config import EmbeddingConfig
from .text import search_tokens


class Embedder(Protocol):
    dimension: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class HashingEmbedder:
    """Small dependency-free embedder.

    This is not meant to beat modern neural embeddings. It keeps the scaffold
    runnable in restricted environments and provides lexical-semantic signal
    for tests, local demos, and CI.
    """

    def __init__(self, dimension: int = 384, normalize: bool = True):
        self.dimension = dimension
        self.normalize = normalize

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _tokens(text)
        features = [*tokens, *[f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]]
        for token in features:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            integer = int.from_bytes(digest, "big")
            index = integer % self.dimension
            sign = 1.0 if integer & 1 else -1.0
            vector[index] += sign
        if self.normalize:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
        return vector


class SentenceTransformerEmbedder:
    """Adapter for sentence-transformers models."""

    def __init__(self, config: EmbeddingConfig):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc

        self.config = config
        self.model = SentenceTransformer(config.model_name)
        dimension = getattr(self.model, "get_sentence_embedding_dimension", lambda: None)()
        self.dimension = int(dimension or config.fallback_dimension)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in embeddings]


def build_embedder(config: EmbeddingConfig | None = None) -> Embedder:
    config = config or EmbeddingConfig()
    if config.provider in {"sentence-transformers", "auto"}:
        try:
            return SentenceTransformerEmbedder(config)
        except RuntimeError:
            if config.provider == "sentence-transformers":
                raise
    return HashingEmbedder(config.fallback_dimension, config.normalize)


def _tokens(text: str) -> list[str]:
    return search_tokens(text)
