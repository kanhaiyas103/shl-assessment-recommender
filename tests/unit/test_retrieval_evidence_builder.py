"""Tests for retrieval evidence construction."""

import pytest

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.models.retrieval import RetrievalResult
from shl_agent.retrieval.evidence_builder import RetrievalEvidenceBuilder
from shl_agent.retrieval.requirements import CanonicalRequirements


def result(assessment_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        assessment=Assessment(
            assessment_id,
            f"Assessment {assessment_id}",
            f"https://www.shl.com/products/product-catalog/view/{assessment_id}/",
            (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
            "Python skills.",
        ),
        semantic_score=0.8,
        lexical_score=0.7,
        metadata_match_score=1.0,
        fused_score=0.9,
        rerank_score=score,
        matched_requirements=("python",),
        explanation="Matched python.",
    )


def test_evidence_builder_computes_confidence_and_distribution() -> None:
    evidence = RetrievalEvidenceBuilder().build(
        results=(result("a", 0.9), result("b", 0.7)),
        requirements=CanonicalRequirements("python", skills=("python",)),
    )

    assert evidence.top_score == 0.9
    assert evidence.top_score_margin == 0.2
    assert evidence.required_skill_coverage == 1.0
    assert evidence.matched_catalog_ids == ("a", "b")
    assert evidence.candidate_explanations[0] == "Matched python."


def test_evidence_builder_rejects_duplicates() -> None:
    duplicate = result("a", 0.9)
    with pytest.raises(ValueError, match="duplicate"):
        RetrievalEvidenceBuilder().build(
            results=(duplicate, duplicate),
            requirements=CanonicalRequirements("python"),
        )
