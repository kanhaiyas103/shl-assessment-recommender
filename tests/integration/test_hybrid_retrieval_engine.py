"""Integration tests for the deterministic hybrid retrieval engine."""

from collections.abc import Sequence

import numpy as np
import pytest

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import ConversationIntent
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.models.recommendation import ResolvedRequirements
from shl_agent.retrieval.candidate_ranker import CandidateRanker
from shl_agent.retrieval.embedding_service import EmbeddingService, FloatMatrix
from shl_agent.retrieval.engine import HybridRetrievalEngine
from shl_agent.retrieval.evidence_builder import RetrievalEvidenceBuilder
from shl_agent.retrieval.lexical_retriever import LexicalRetriever
from shl_agent.retrieval.metadata_filter import MetadataFilter
from shl_agent.retrieval.query_expansion import QueryExpansionService
from shl_agent.retrieval.rank_fusion import RankFusionService
from shl_agent.retrieval.requirements import RequirementResolver
from shl_agent.retrieval.semantic_retriever import CandidateSignal, SemanticRetriever


class Store:
    """Tiny retrieval catalog."""

    def __init__(self) -> None:
        self._items = {
            "python": Assessment(
                "python",
                "Python New",
                "https://www.shl.com/products/product-catalog/view/python-new/",
                (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
                "Measures Python programming skills.",
                duration_minutes=20,
                job_levels=("Mid-Professional",),
            ),
            "leader": Assessment(
                "leader",
                "Leadership Assessment",
                "https://www.shl.com/products/product-catalog/view/leadership/",
                (AssessmentTestType.COMPETENCIES,),
                "Measures leadership competencies.",
                duration_minutes=35,
            ),
        }

    def assessment(self, assessment_id: str) -> Assessment:
        return self._items[assessment_id]

    def contains(self, assessment_id: str) -> bool:
        return assessment_id in self._items

    def all_assessments(self) -> tuple[Assessment, ...]:
        return tuple(self._items.values())


class Backend:
    """Fake query embedding backend."""

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 2

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        return np.asarray(
            [[1.0, 0.0] if "python" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


class SearchIndex:
    """Semantic search fake biased toward Python for Python queries."""

    def search(self, vector: Sequence[float], *, limit: int) -> tuple[CandidateSignal, ...]:
        assert limit > 0
        if vector[0] >= vector[1]:
            return (CandidateSignal("python", 0.95, 1), CandidateSignal("leader", 0.2, 2))
        return (CandidateSignal("leader", 0.9, 1), CandidateSignal("python", 0.2, 2))


def engine() -> HybridRetrievalEngine:
    store = Store()
    return HybridRetrievalEngine(
        requirement_resolver=RequirementResolver(),
        query_expander=QueryExpansionService(),
        semantic_retriever=SemanticRetriever(
            embedding_service=EmbeddingService(Backend()),
            search_index=SearchIndex(),
        ),
        lexical_retriever=LexicalRetriever(store),
        metadata_filter=MetadataFilter(),
        rank_fusion=RankFusionService(),
        candidate_ranker=CandidateRanker(store),
        evidence_builder=RetrievalEvidenceBuilder(),
        store=store,
    )


@pytest.mark.asyncio
async def test_end_to_end_skill_lookup_is_deterministic() -> None:
    requirements = ResolvedRequirements(
        intent=ConversationIntent.RECOMMEND,
        skills=("Python",),
        test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
        max_duration_minutes=30,
    )

    first = await engine().retrieve(requirements)
    second = await engine().retrieve(requirements)

    assert first.matched_catalog_ids == second.matched_catalog_ids
    assert first.results[0].assessment.assessment_id == "python"
    assert first.results[0].metadata_match_score == 1.0
    assert first.retrieval_confidence > 0.5


@pytest.mark.asyncio
async def test_category_lookup_returns_competency_candidate() -> None:
    evidence = await engine().retrieve(
        ResolvedRequirements(
            intent=ConversationIntent.RECOMMEND,
            competencies=("Leadership",),
            test_types=(AssessmentTestType.COMPETENCIES,),
        )
    )

    assert "leader" in evidence.matched_catalog_ids


@pytest.mark.asyncio
async def test_exact_assessment_lookup_prioritizes_name_match() -> None:
    evidence = await engine().retrieve(
        ResolvedRequirements(
            intent=ConversationIntent.RECOMMEND,
            assessment_names=("Python New",),
        )
    )

    assert evidence.results[0].assessment.name == "Python New"
