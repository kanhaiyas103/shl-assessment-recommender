"""Tests for semantic retriever orchestration."""

from collections.abc import Sequence

import numpy as np
import pytest

from shl_agent.models.retrieval import ExpandedQuery
from shl_agent.retrieval.embedding_service import EmbeddingService, FloatMatrix
from shl_agent.retrieval.semantic_retriever import CandidateSignal, SemanticRetriever


class Backend:
    """Fake embedding backend."""

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 2

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        return np.asarray([[1.0, float(index)] for index, _ in enumerate(texts)], dtype=np.float32)


class SearchIndex:
    """Fake vector index."""

    def search(self, vector: Sequence[float], *, limit: int) -> tuple[CandidateSignal, ...]:
        assert limit == 5
        return (
            CandidateSignal("a", float(vector[0]), 1),
            CandidateSignal("b", 0.5, 2),
        )


@pytest.mark.asyncio
async def test_semantic_retriever_weights_expanded_queries() -> None:
    retriever = SemanticRetriever(
        embedding_service=EmbeddingService(Backend()),
        search_index=SearchIndex(),
        per_query_limit=5,
    )

    results = await retriever.retrieve(
        (
            ExpandedQuery("python", 1.0, "original"),
            ExpandedQuery("developer", 0.5, "role"),
        )
    )

    assert results["a"].score == 1.0
    assert results["a"].matched_terms == ("python",)
