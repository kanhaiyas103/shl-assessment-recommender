"""Tests for retrieval requirement normalization."""

from shl_agent.models.enums import ConversationIntent
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.models.recommendation import ResolvedRequirements
from shl_agent.retrieval.requirements import RequirementResolver


def test_requirement_resolver_normalizes_terms_and_synonyms() -> None:
    requirements = ResolvedRequirements(
        intent=ConversationIntent.RECOMMEND,
        role="SWE",
        seniority="mid",
        skills=("JS, NodeJS", "Python"),
        competencies=("Situational Judgement",),
        test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
        max_duration_minutes=30,
        assessment_names=(" Java New ",),
    )

    canonical = RequirementResolver().resolve(requirements)

    assert canonical.role == "software engineer"
    assert canonical.seniority == "mid-professional"
    assert canonical.skills == ("javascript", "node.js", "python")
    assert canonical.competencies == ("situational judgment",)
    assert canonical.max_duration_minutes == 30
    assert "java new" in canonical.assessment_names
    assert "python" in canonical.original_query
