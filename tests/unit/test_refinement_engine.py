"""Tests for requirement refinement merging."""

from shl_agent.models.enums import ConversationIntent
from shl_agent.models.recommendation import ResolvedRequirements
from shl_agent.services.conversation_engine import RefinementEngine


def test_refinement_engine_adds_constraints_without_discarding_previous() -> None:
    previous = ResolvedRequirements(
        intent=ConversationIntent.RECOMMEND,
        role="developer",
        skills=("python",),
    )
    update = ResolvedRequirements(
        intent=ConversationIntent.REFINE,
        max_duration_minutes=30,
        skills=("sql",),
    )

    merged = RefinementEngine().merge(previous, update)

    assert merged.role == "developer"
    assert merged.skills == ("python", "sql")
    assert merged.max_duration_minutes == 30
