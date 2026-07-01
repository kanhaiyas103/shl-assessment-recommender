"""Tests for grounded assessment comparison."""

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.services.conversation_engine import ComparisonEngine


class Store:
    """Small comparison catalog."""

    def assessment(self, assessment_id: str) -> Assessment:
        raise KeyError(assessment_id)

    def all_assessments(self) -> tuple[Assessment, ...]:
        return (
            Assessment(
                "python",
                "Python New",
                "https://www.shl.com/products/product-catalog/view/python-new/",
                (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
                "Python skills.",
                duration_minutes=20,
                remote_testing=True,
            ),
            Assessment(
                "java",
                "Java New",
                "https://www.shl.com/products/product-catalog/view/java-new/",
                (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
                "Java skills.",
                duration_minutes=25,
                remote_testing=True,
            ),
        )


def test_comparison_engine_uses_only_catalog_matches() -> None:
    engine = ComparisonEngine(Store())

    assessments = engine.compare(("Python New", "Java New"))
    reply = engine.compose(assessments)

    assert [assessment.name for assessment in assessments] == ["Python New", "Java New"]
    assert "duration" not in reply.casefold() or "minutes" in reply
    assert "catalog fields only" in reply
