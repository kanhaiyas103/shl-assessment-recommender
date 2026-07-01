"""Requirement, readiness, and recommendation domain models."""

from dataclasses import dataclass

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import ConversationIntent, DecisionAction, TestType


@dataclass(frozen=True, slots=True)
class ResolvedRequirements:
    """Current requirements resolved from the full conversation history."""

    intent: ConversationIntent
    role: str | None = None
    seniority: str | None = None
    skills: tuple[str, ...] = ()
    competencies: tuple[str, ...] = ()
    test_types: tuple[TestType, ...] = ()
    max_duration_minutes: int | None = None
    languages: tuple[str, ...] = ()
    assessment_names: tuple[str, ...] = ()
    excluded_assessment_ids: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject impossible constraints before retrieval."""
        if self.max_duration_minutes is not None and self.max_duration_minutes <= 0:
            raise ValueError("max_duration_minutes must be positive when present")

    @property
    def has_actionable_context(self) -> bool:
        """Return whether the requirements contain a useful retrieval anchor."""
        return any(
            (
                self.role,
                self.skills,
                self.competencies,
                self.test_types,
                self.assessment_names,
            )
        )


@dataclass(frozen=True, slots=True)
class RecommendationReadiness:
    """Auditable policy decision about clarifying or recommending."""

    action: DecisionAction
    reason: str
    clarification_question: str | None = None

    def __post_init__(self) -> None:
        """Require a question only for clarification decisions."""
        if self.action is DecisionAction.CLARIFY and not self.clarification_question:
            raise ValueError("clarification decisions require a question")
        if self.action is not DecisionAction.CLARIFY and self.clarification_question is not None:
            raise ValueError("only clarification decisions may include a question")


@dataclass(frozen=True, slots=True)
class RecommendationDecision:
    """Framework-independent result of one conversation turn."""

    reply: str
    recommendations: tuple[Assessment, ...]
    end_of_conversation: bool

    def __post_init__(self) -> None:
        """Enforce response invariants before API serialization."""
        if not self.reply.strip():
            raise ValueError("reply must not be blank")
        if len(self.recommendations) > 10:
            raise ValueError("at most ten recommendations are allowed")
        ids = [assessment.assessment_id for assessment in self.recommendations]
        if len(ids) != len(set(ids)):
            raise ValueError("recommendations must not contain duplicates")
