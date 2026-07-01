"""Tests for embedding service validation."""

from collections.abc import Sequence

import numpy as np
import pytest

from shl_agent.retrieval.embedding_service import EmbeddingError, EmbeddingService, FloatMatrix
from shl_agent.retrieval.text_builder import SemanticDocument


class FakeBackend:
    """Deterministic embedding backend for unit tests."""

    def __init__(self, vectors: FloatMatrix | None = None) -> None:
        self._vectors = vectors

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimension(self) -> int:
        return 3

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        if self._vectors is not None:
            return self._vectors
        rows = [[float(index), 0.0, 1.0] for index, _ in enumerate(texts)]
        return np.asarray(rows, dtype=np.float32)


@pytest.mark.asyncio
async def test_embedding_service_preserves_document_order() -> None:
    service = EmbeddingService(FakeBackend())
    documents = (
        SemanticDocument("a", "alpha"),
        SemanticDocument("b", "beta"),
    )

    result = await service.embed_documents(documents)

    assert service.model_name == "fake-model"
    assert service.dimension == 3
    assert result.failed_embeddings == 0
    assert result.skipped_embeddings == 0
    assert result.vectors.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]


@pytest.mark.asyncio
async def test_embedding_service_rejects_shape_mismatch() -> None:
    service = EmbeddingService(FakeBackend(np.asarray([[1.0, 2.0]], dtype=np.float32)))

    with pytest.raises(EmbeddingError, match="shape"):
        await service.embed_documents((SemanticDocument("a", "alpha"),))


@pytest.mark.asyncio
async def test_embedding_service_rejects_empty_documents() -> None:
    service = EmbeddingService(FakeBackend())

    with pytest.raises(EmbeddingError, match="must not be empty"):
        await service.embed_documents((SemanticDocument("a", " "),))
