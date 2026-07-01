"""Tests for recommendation readiness policy."""

from shl_agent.models.enums import ConversationIntent, DecisionAction
from shl_agent.models.recommendation import ResolvedRequirements
from shl_agent.models.retrieval import RetrievalEvidence
from shl_agent.services.conversation_engine import RecommendationReadinessPolicy


def evidence(confidence: float, coverage: float) -> RetrievalEvidence:
    return RetrievalEvidence(
        results=(),
        retrieval_confidence=confidence,
        required_skill_coverage=coverage,
        constraint_coverage=0.5,
        top_score_margin=0.0,
    )


def test_readiness_clarifies_when_request_has_no_anchor() -> None:
    decision = RecommendationReadinessPolicy().decide(
        ResolvedRequirements(intent=ConversationIntent.CLARIFY),
        evidence(0.0, 0.0),
        can_clarify=True,
    )

    assert decision.action is DecisionAction.CLARIFY


def test_readiness_recommends_when_budget_or_evidence_is_sufficient() -> None:
    decision = RecommendationReadinessPolicy().decide(
        ResolvedRequirements(intent=ConversationIntent.RECOMMEND, skills=("python",)),
        evidence(0.8, 1.0),
        can_clarify=True,
    )

    assert decision.action is DecisionAction.RECOMMEND


def test_readiness_asks_one_material_technical_focus_question() -> None:
    decision = RecommendationReadinessPolicy().decide(
        ResolvedRequirements(
            intent=ConversationIntent.RECOMMEND,
            skills=("java", "spring", "angular", "aws", "docker", "sql"),
        ),
        evidence(0.8, 1.0),
        can_clarify=True,
    )

    assert decision.action is DecisionAction.CLARIFY
    assert decision.clarification_question is not None
    assert "backend" in decision.clarification_question


def test_readiness_recommends_when_material_clarification_already_used() -> None:
    decision = RecommendationReadinessPolicy().decide(
        ResolvedRequirements(
            intent=ConversationIntent.RECOMMEND,
            skills=("java", "spring", "angular", "aws", "docker", "sql"),
        ),
        evidence(0.8, 1.0),
        can_clarify=False,
    )

    assert decision.action is DecisionAction.RECOMMEND


def test_readiness_does_not_clarify_when_safety_request_is_specific() -> None:
    decision = RecommendationReadinessPolicy().decide(
        ResolvedRequirements(
            intent=ConversationIntent.RECOMMEND,
            skills=("safety", "dependability", "industrial", "plant operator"),
        ),
        evidence(0.8, 1.0),
        can_clarify=True,
    )

    assert decision.action is DecisionAction.RECOMMEND
