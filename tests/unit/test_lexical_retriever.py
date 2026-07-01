"""Tests for lexical candidate retrieval."""

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import ConversationIntent
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.models.recommendation import ResolvedRequirements
from shl_agent.retrieval.lexical_retriever import LexicalRetriever
from shl_agent.retrieval.requirements import RequirementResolver


class Store:
    """Small lexical catalog."""

    def all_assessments(self) -> tuple[Assessment, ...]:
        return (
            Assessment(
                "python",
                "Python New",
                "https://www.shl.com/products/product-catalog/view/python-new/",
                (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
                "Measures Python programming skills.",
            ),
            Assessment(
                "leader",
                "Leadership Report",
                "https://www.shl.com/products/product-catalog/view/leadership-report/",
                (AssessmentTestType.COMPETENCIES,),
                "Measures leadership competencies.",
            ),
        )


def test_lexical_retriever_finds_exact_skill_and_name() -> None:
    canonical = RequirementResolver().resolve(
        ResolvedRequirements(
            intent=ConversationIntent.RECOMMEND,
            skills=("Python",),
            assessment_names=("Python New",),
        )
    )

    results = LexicalRetriever(Store()).retrieve(canonical)

    assert "python" in results
    assert "leader" not in results
    assert "python" in results["python"].matched_terms
