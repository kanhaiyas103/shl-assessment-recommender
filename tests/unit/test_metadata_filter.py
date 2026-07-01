"""Tests for recall-friendly metadata scoring."""

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import ConversationIntent
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.models.recommendation import ResolvedRequirements
from shl_agent.retrieval.metadata_filter import MetadataFilter
from shl_agent.retrieval.requirements import RequirementResolver


def test_metadata_filter_scores_constraints_without_unnecessary_removal() -> None:
    assessment = Assessment(
        "python",
        "Python New",
        "https://www.shl.com/products/product-catalog/view/python-new/",
        (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
        "Python skills.",
        duration_minutes=25,
        job_levels=("Mid-Professional",),
    )
    requirements = RequirementResolver().resolve(
        ResolvedRequirements(
            intent=ConversationIntent.RECOMMEND,
            seniority="mid",
            test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
            max_duration_minutes=30,
        )
    )

    signal = MetadataFilter().score(assessment, requirements)

    assert signal.score == 1.0
    assert signal.constraints_satisfied == signal.constraint_checks
    assert signal.matched_requirements == ("duration", "test_type", "seniority")


def test_metadata_filter_scores_excluded_candidate_as_zero() -> None:
    assessment = Assessment(
        "python",
        "Python New",
        "https://www.shl.com/products/product-catalog/view/python-new/",
        (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
        "Python skills.",
    )
    requirements = RequirementResolver().resolve(
        ResolvedRequirements(
            intent=ConversationIntent.RECOMMEND,
            excluded_assessment_ids=("python",),
        )
    )

    assert MetadataFilter().score(assessment, requirements).score == 0.0
