"""Tests for final candidate ranking."""

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.retrieval.candidate_ranker import CandidateRanker
from shl_agent.retrieval.rank_fusion import FusedCandidate
from shl_agent.retrieval.requirements import CanonicalRequirements


class Store:
    """Small candidate store."""

    def __init__(self) -> None:
        self._assessment = Assessment(
            "python",
            "Python New",
            "https://www.shl.com/products/product-catalog/view/python-new/",
            (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
            "Python skills.",
        )

    def assessment(self, assessment_id: str) -> Assessment:
        assert assessment_id == "python"
        return self._assessment

    def contains(self, assessment_id: str) -> bool:
        return assessment_id == "python"


def test_candidate_ranker_returns_reranked_results_with_explanations() -> None:
    requirements = CanonicalRequirements("python", skills=("python",))
    fused = (FusedCandidate("python", 0.9, 0.8, 1.0, 1.0, 1, 1, ("python", "duration")),)

    results = CandidateRanker(Store()).rank(fused, requirements)

    assert len(results) == 1
    assert results[0].rerank_score > 0.8
    assert "semantic=" in results[0].explanation


class MultiStore:
    """Candidate store for ordering checks."""

    def __init__(self) -> None:
        self._assessments = {
            "exact": Assessment(
                "exact",
                "Python New",
                "https://www.shl.com/products/product-catalog/view/python-new/",
                (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
                "Python skills.",
            ),
            "generic": Assessment(
                "generic",
                "Programming Concepts",
                "https://www.shl.com/products/product-catalog/view/programming-concepts/",
                (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
                "General programming skills.",
            ),
        }

    def assessment(self, assessment_id: str) -> Assessment:
        return self._assessments[assessment_id]

    def contains(self, assessment_id: str) -> bool:
        return assessment_id in self._assessments


def test_candidate_ranker_boosts_dual_channel_exact_skill_matches() -> None:
    requirements = CanonicalRequirements("python", skills=("python",))
    fused = (
        FusedCandidate("generic", 0.88, 0.8, 0.0, 1.0, 1, None, ("programming",)),
        FusedCandidate("exact", 0.7, 0.7, 0.8, 1.0, 4, 4, ("python",)),
    )

    results = CandidateRanker(MultiStore()).rank(fused, requirements)

    assert results[0].assessment.assessment_id == "exact"
    assert "support=" in results[0].explanation
