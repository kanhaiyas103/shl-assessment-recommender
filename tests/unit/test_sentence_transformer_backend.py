"""Tests for SentenceTransformer backend without downloading real models."""

import sys
from types import ModuleType

import numpy as np
import pytest

from shl_agent.retrieval.embedding_service import EmbeddingError, SentenceTransformerBackend


class FakeSentenceTransformer:
    """Small fake SentenceTransformer implementation."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def get_embedding_dimension(self) -> int:
        return 2

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        assert batch_size == 4
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        assert show_progress_bar is False
        return np.asarray([[1.0, float(index)] for index, _ in enumerate(sentences)])


class InvalidDimensionSentenceTransformer(FakeSentenceTransformer):
    """Fake model with an invalid dimension."""

    def get_embedding_dimension(self) -> int:
        return 0

    def get_sentence_embedding_dimension(self) -> int:
        return 0


def install_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
    model_class: type[FakeSentenceTransformer],
) -> None:
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = model_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_sentence_transformer_backend_encodes_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sentence_transformer(monkeypatch, FakeSentenceTransformer)
    backend = SentenceTransformerBackend("fake-model", batch_size=4)

    vectors = backend.encode(["alpha", "beta"])

    assert backend.model_name == "fake-model"
    assert backend.dimension == 2
    assert vectors.dtype == np.float32
    assert vectors.tolist() == [[1.0, 0.0], [1.0, 1.0]]


def test_sentence_transformer_backend_rejects_invalid_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sentence_transformer(monkeypatch, InvalidDimensionSentenceTransformer)

    with pytest.raises(EmbeddingError, match="invalid dimension"):
        _ = SentenceTransformerBackend("fake-model").dimension
