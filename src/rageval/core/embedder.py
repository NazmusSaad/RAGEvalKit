from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rageval.core.config import EmbeddingConfig


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DummyEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Converts each text to a SHA-256 hash, samples bytes cyclically to
    produce ``dim`` floats, then L2-normalises the result.  Same text
    always produces the same unit vector; different texts almost always
    produce different vectors.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes
        vals = [(digest[i % len(digest)] / 127.5) - 1.0 for i in range(self._dim)]
        norm = sum(v * v for v in vals) ** 0.5
        return [v / norm for v in vals] if norm > 0.0 else [0.0] * self._dim


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model.

    ``sentence_transformers`` is imported lazily inside ``__init__`` so
    merely importing this module never triggers a model download.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        # sentence-transformers ≥3.x renamed the method; fall back for older versions
        _get_dim = getattr(
            self._model, "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self._dim: int = _get_dim()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()  # type: ignore[return-value]


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Build the appropriate :class:`Embedder` from an ``EmbeddingConfig``."""
    if config.provider == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=config.model)
    if config.provider == "dummy":
        return DummyEmbedder(dim=16)
    raise ValueError(f"Unknown embedding provider: {config.provider!r}")
