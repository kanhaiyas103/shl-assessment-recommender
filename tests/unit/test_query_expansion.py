"""Tests for deterministic retrieval query expansion."""

from shl_agent.models.enums import ConversationIntent
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.models.recommendation import ResolvedRequirements
from shl_agent.retrieval.query_expansion import QueryExpansionService
from shl_agent.retrieval.requirements import RequirementResolver


def test_query_expansion_is_bounded_deterministic_and_non_inventive() -> None:
    canonical = RequirementResolver().resolve(
        ResolvedRequirements(
            intent=ConversationIntent.RECOMMEND,
            role="Developer",
            skills=("JavaScript",),
            test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
        )
    )
    service = QueryExpansionService(max_expansions=5)

    first = service.expand(canonical)
    second = service.expand(canonical)

    assert first == second
    assert len(first) == 5
    assert first[0].source == "original"
    assert any(query.text == "js" for query in first)
    assert all(query.weight <= 1 for query in first)
