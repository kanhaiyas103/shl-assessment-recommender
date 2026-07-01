"""Embedding services for deterministic catalog index builds."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from shl_agent.retrieval.text_builder import SemanticDocument

FloatMatrix = NDArray[np.float32]


class EmbeddingBackend(Protocol):
    """Low-level embedding backend contract."""

    @property
    def model_name(self) -> str:
        """Return the embedding model identifier."""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Return the fixed vector dimension."""
        raise NotImplementedError

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed text in stable input order."""
        raise NotImplementedError


class SentenceTransformerModel(Protocol):
    """Runtime subset used from SentenceTransformer."""

    def get_embedding_dimension(self) -> int:
        """Return the model's embedding dimension."""
        raise NotImplementedError

    def get_sentence_embedding_dimension(self) -> int:
        """Return the model's embedding dimension."""
        raise NotImplementedError

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object:
        """Return encoded vectors."""
        raise NotImplementedError


class SentenceTransformerBackend:
    """SentenceTransformers-backed embedding backend."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformerModel | None = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        """Return the embedding model identifier."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the fixed vector dimension."""
        if self._dimension is None:
            model = self._load_model()
            if hasattr(model, "get_embedding_dimension"):
                dimension = int(model.get_embedding_dimension())
            else:
                dimension = int(model.get_sentence_embedding_dimension())
            if dimension <= 0:
                raise EmbeddingError("embedding model returned an invalid dimension")
            self._dimension = dimension
        return self._dimension

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed text with normalized vectors for cosine/IP FAISS search."""
        model = self._load_model()
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _load_model(self) -> SentenceTransformerModel:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self._model_name)
        return self._model


class EmbeddingError(ValueError):
    """Raised when embedding generation produces unusable vectors."""


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResult:
    """Embeddings and counters produced by an embedding run."""

    vectors: FloatMatrix
    failed_embeddings: int
    skipped_embeddings: int


class EmbeddingService:
    """Generate and validate embeddings for semantic documents."""

    def __init__(self, backend: EmbeddingBackend) -> None:
        self._backend = backend

    @property
    def model_name(self) -> str:
        """Return the configured embedding model."""
        return self._backend.model_name

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._backend.dimension

    async def embed_documents(self, documents: Sequence[SemanticDocument]) -> EmbeddingBatchResult:
        """Embed every semantic document in deterministic order."""
        texts = [document.text for document in documents if document.text.strip()]
        skipped = len(documents) - len(texts)
        if skipped:
            raise EmbeddingError("semantic documents must not be empty")
        try:
            vectors = await asyncio.to_thread(self._backend.encode, texts)
        except Exception as exc:
            raise EmbeddingError("embedding backend failed") from exc

        if vectors.shape != (len(documents), self.dimension):
            raise EmbeddingError(
                "embedding matrix shape does not match documents and model dimension"
            )
        if not np.isfinite(vectors).all():
            raise EmbeddingError("embedding matrix contains non-finite values")

        return EmbeddingBatchResult(
            vectors=np.ascontiguousarray(vectors, dtype=np.float32),
            failed_embeddings=0,
            skipped_embeddings=skipped,
        )

    async def embed_texts(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed arbitrary query texts in stable input order."""
        if any(not text.strip() for text in texts):
            raise EmbeddingError("query text must not be blank")
        vectors = await asyncio.to_thread(self._backend.encode, texts)
        if vectors.shape != (len(texts), self.dimension):
            raise EmbeddingError("query embedding matrix shape does not match model dimension")
        return np.ascontiguousarray(vectors, dtype=np.float32)
