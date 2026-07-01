"""Semantic FAISS retrieval without reranking."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from shl_agent.models.retrieval import ExpandedQuery
from shl_agent.retrieval.embedding_service import EmbeddingService
from shl_agent.retrieval.runtime_store import CatalogMetadataStore


@dataclass(frozen=True, slots=True)
class CandidateSignal:
    """Raw candidate signal emitted by retrieval components."""

    assessment_id: str
    score: float
    rank: int
    matched_terms: tuple[str, ...] = ()


class FaissSearchIndex:
    """Reusable FAISS index wrapper."""

    def __init__(self, index_path: Path, store: CatalogMetadataStore) -> None:
        import faiss  # noqa: PLC0415

        self._index = faiss.read_index(str(index_path))
        self._store = store

    def search(self, vector: Sequence[float], *, limit: int) -> tuple[CandidateSignal, ...]:
        """Search the FAISS index and map vector IDs to catalog IDs."""
        matrix = np.asarray([vector], dtype=np.float32)
        scores, indices = self._index.search(matrix, limit)
        results: list[CandidateSignal] = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0], strict=False), start=1):
            if int(index) < 0:
                continue
            results.append(
                CandidateSignal(
                    assessment_id=self._store.id_for_vector_index(int(index)),
                    score=max(0.0, min(1.0, float(score))),
                    rank=rank,
                )
            )
        return tuple(results)


class SearchIndex(Protocol):
    """Vector-search capability required by SemanticRetriever."""

    def search(self, vector: Sequence[float], *, limit: int) -> tuple[CandidateSignal, ...]:
        """Return candidate signals for a query vector."""
        raise NotImplementedError


class SemanticRetriever:
    """Run semantic search over expanded retrieval views."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        search_index: SearchIndex,
        per_query_limit: int = 50,
    ) -> None:
        self._embedding_service = embedding_service
        self._search_index = search_index
        self._per_query_limit = per_query_limit

    async def retrieve(self, queries: Sequence[ExpandedQuery]) -> dict[str, CandidateSignal]:
        """Return best semantic signal per assessment ID."""
        vectors = await self._embedding_service.embed_texts(tuple(query.text for query in queries))
        best: dict[str, CandidateSignal] = {}
        for query, vector in zip(queries, vectors, strict=True):
            for signal in self._search_index.search(vector, limit=self._per_query_limit):
                weighted = CandidateSignal(
                    assessment_id=signal.assessment_id,
                    score=signal.score * query.weight,
                    rank=signal.rank,
                    matched_terms=(query.text,),
                )
                previous = best.get(weighted.assessment_id)
                if previous is None or weighted.score > previous.score:
                    best[weighted.assessment_id] = weighted
        return best
