"""Tests for framework-independent domain invariants."""

import pytest

from shl_agent.models import (
    Assessment,
    Conversation,
    ConversationMessage,
    ConversationRole,
    DecisionAction,
    ExpandedQuery,
    RecommendationDecision,
    RecommendationReadiness,
    ResolvedRequirements,
    RetrievalEvidence,
    RetrievalResult,
)
from shl_agent.models import (
    TestType as AssessmentTestType,
)


def make_assessment() -> Assessment:
    return Assessment(
        assessment_id="java-8",
        name="Java 8",
        url="https://www.shl.com/solutions/products/product-catalog/view/java-8/",
        test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
        description="Java knowledge assessment.",
    )


def test_assessment_requires_shl_url() -> None:
    with pytest.raises(ValueError, match="SHL catalog URL"):
        Assessment(
            assessment_id="java-8",
            name="Java 8",
            url="https://example.com/java",
            test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
            description="Java knowledge assessment.",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"assessment_id": " "}, "assessment_id"),
        ({"name": " "}, "assessment name"),
        ({"test_types": ()}, "test type"),
        ({"duration_minutes": 0}, "duration_minutes"),
    ],
)
def test_assessment_rejects_invalid_fields(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "assessment_id": "java-8",
        "name": "Java 8",
        "url": "https://www.shl.com/catalog/java-8/",
        "test_types": (AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
        "description": "Java knowledge assessment.",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        Assessment(**values)  # type: ignore[arg-type]


def test_conversation_reports_remaining_budget() -> None:
    conversation = Conversation(
        messages=(
            ConversationMessage(
                role=ConversationRole.USER,
                content="I need a Java assessment.",
            ),
        )
    )

    assert conversation.remaining_message_budget == 7


@pytest.mark.parametrize(
    "messages",
    [
        (),
        tuple(
            ConversationMessage(role=ConversationRole.USER, content=f"message-{index}")
            for index in range(9)
        ),
        (ConversationMessage(role=ConversationRole.ASSISTANT, content="Hello"),),
        (
            ConversationMessage(role=ConversationRole.USER, content="Java"),
            ConversationMessage(role=ConversationRole.ASSISTANT, content="Seniority?"),
        ),
        (
            ConversationMessage(role=ConversationRole.USER, content="Java"),
            ConversationMessage(role=ConversationRole.USER, content="Mid-level"),
        ),
    ],
)
def test_conversation_rejects_invalid_history(
    messages: tuple[ConversationMessage, ...],
) -> None:
    with pytest.raises(ValueError):
        Conversation(messages=messages)


def test_message_rejects_blank_content() -> None:
    with pytest.raises(ValueError, match="content"):
        ConversationMessage(role=ConversationRole.USER, content=" ")


def test_expanded_query_requires_normalized_weight() -> None:
    with pytest.raises(ValueError, match="weight"):
        ExpandedQuery(text="Java skills", weight=1.1, source="deterministic")


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": " "},
        {"source": " "},
    ],
)
def test_expanded_query_rejects_blank_fields(overrides: dict[str, str]) -> None:
    values = {"text": "Java", "weight": 1.0, "source": "original"}
    values.update(overrides)
    with pytest.raises(ValueError):
        ExpandedQuery(**values)  # type: ignore[arg-type]


def test_clarification_readiness_requires_question() -> None:
    with pytest.raises(ValueError, match="require a question"):
        RecommendationReadiness(
            action=DecisionAction.CLARIFY,
            reason="The role is missing.",
        )


def test_non_clarification_readiness_rejects_question() -> None:
    with pytest.raises(ValueError, match="only clarification"):
        RecommendationReadiness(
            action=DecisionAction.RECOMMEND,
            reason="Enough context.",
            clarification_question="What role?",
        )


def test_resolved_requirements_reports_actionable_context() -> None:
    empty = ResolvedRequirements(intent="clarify")  # type: ignore[arg-type]
    actionable = ResolvedRequirements(intent="recommend", role="developer")  # type: ignore[arg-type]

    assert empty.has_actionable_context is False
    assert actionable.has_actionable_context is True


def test_resolved_requirements_rejects_invalid_duration() -> None:
    with pytest.raises(ValueError, match="max_duration"):
        ResolvedRequirements(intent="recommend", max_duration_minutes=0)  # type: ignore[arg-type]


def test_recommendation_decision_rejects_duplicates() -> None:
    assessment = make_assessment()
    with pytest.raises(ValueError, match="duplicates"):
        RecommendationDecision(
            reply="Results",
            recommendations=(assessment, assessment),
            end_of_conversation=False,
        )


def test_recommendation_decision_rejects_blank_reply_and_excess_results() -> None:
    with pytest.raises(ValueError, match="reply"):
        RecommendationDecision(reply=" ", recommendations=(), end_of_conversation=False)
    with pytest.raises(ValueError, match="ten"):
        RecommendationDecision(
            reply="Results",
            recommendations=tuple(
                Assessment(
                    assessment_id=f"id-{index}",
                    name=f"Assessment {index}",
                    url=f"https://www.shl.com/catalog/{index}/",
                    test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
                    description="Description",
                )
                for index in range(11)
            ),
            end_of_conversation=False,
        )


def test_retrieval_models_validate_normalized_scores() -> None:
    assessment = make_assessment()
    with pytest.raises(ValueError, match="scores"):
        RetrievalResult(
            assessment=assessment,
            semantic_score=1.1,
            lexical_score=0.5,
            metadata_match_score=0.5,
            fused_score=0.5,
        )
    with pytest.raises(ValueError, match="confidence"):
        RetrievalEvidence(
            results=(),
            retrieval_confidence=-0.1,
            required_skill_coverage=0.5,
            constraint_coverage=0.5,
            top_score_margin=0.1,
        )
    with pytest.raises(ValueError, match="top_score_margin"):
        RetrievalEvidence(
            results=(),
            retrieval_confidence=0.5,
            required_skill_coverage=0.5,
            constraint_coverage=0.5,
            top_score_margin=1.1,
        )
