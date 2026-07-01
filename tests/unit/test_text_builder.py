"""Tests for deterministic semantic document construction."""

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.retrieval.text_builder import AssessmentTextBuilder


def make_assessment() -> Assessment:
    return Assessment(
        assessment_id="python-new",
        name="Python New",
        url="https://www.shl.com/products/product-catalog/view/python-new/",
        test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS, AssessmentTestType.SIMULATIONS),
        description="Measures Python programming and debugging skills.",
        duration_minutes=22,
        remote_testing=True,
        adaptive_irt=False,
        job_levels=("Entry-Level", "Mid-Professional"),
        languages=("English",),
    )


def test_text_builder_is_deterministic_and_labeled() -> None:
    builder = AssessmentTextBuilder()
    assessment = make_assessment()

    first = builder.build(assessment)
    second = builder.build(assessment)

    assert first == second
    assert first.assessment_id == "python-new"
    assert first.text.split("\n\n")[0] == "Assessment Name:\nPython New"
    assert "Description:\nMeasures Python programming and debugging skills." in first.text
    assert "Test Type:\nKnowledge and Skills, Simulations" in first.text
    assert "Remote Testing:\nYes" in first.text
    assert "Adaptive / IRT Support:\nNo" in first.text
    assert "Duration:\n22 minutes" in first.text
    assert "Keywords:" in first.text


def test_text_builder_omits_empty_optional_sections() -> None:
    builder = AssessmentTextBuilder()
    assessment = Assessment(
        assessment_id="ability",
        name="General Ability",
        url="https://www.shl.com/products/product-catalog/view/general-ability/",
        test_types=(AssessmentTestType.ABILITY_AND_APTITUDE,),
        description="Measures reasoning.",
    )

    document = builder.build(assessment)

    assert "Job Levels:" not in document.text
    assert "Languages:" not in document.text
    assert "Duration:" not in document.text
    assert "Remote Testing:" not in document.text
