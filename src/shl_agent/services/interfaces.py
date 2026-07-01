"""Protocols defining application use cases and external boundaries."""

from collections.abc import Sequence
from typing import Protocol

from shl_agent.models.assessment import Assessment
from shl_agent.models.conversation import Conversation
from shl_agent.models.recommendation import (
    RecommendationDecision,
    RecommendationReadiness,
    ResolvedRequirements,
)
from shl_agent.models.retrieval import ExpandedQuery, RetrievalEvidence, RetrievalResult


class ConversationService(Protocol):
    """Orchestrate one stateless conversational turn."""

    async def respond(self, conversation: Conversation) -> RecommendationDecision:
        """Return the next grounded decision for a complete history."""
        ...


class RequirementExtractor(Protocol):
    """Resolve current intent and constraints from conversation history."""

    async def extract(self, conversation: Conversation) -> ResolvedRequirements:
        """Return requirements with later corrections taking precedence."""
        ...


class QueryExpansionService(Protocol):
    """Build bounded retrieval views without adding unsupported facts."""

    async def expand(
        self,
        requirements: ResolvedRequirements,
    ) -> Sequence[ExpandedQuery]:
        """Return weighted original and expanded queries."""
        ...


class Retriever(Protocol):
    """Retrieve catalog candidates and calibrated evidence."""

    async def retrieve(
        self,
        requirements: ResolvedRequirements,
        queries: Sequence[ExpandedQuery],
        *,
        limit: int,
    ) -> RetrievalEvidence:
        """Return broad candidates and query-level confidence signals."""
        ...


class RecommendationRanker(Protocol):
    """Filter and rerank retrieved catalog candidates."""

    async def rank(
        self,
        requirements: ResolvedRequirements,
        candidates: Sequence[RetrievalResult],
        *,
        limit: int,
    ) -> Sequence[Assessment]:
        """Return a constraint-aware, diverse shortlist."""
        ...


class RecommendationReadinessPolicy(Protocol):
    """Decide whether another user fact can materially improve retrieval."""

    def decide(
        self,
        requirements: ResolvedRequirements,
        evidence: RetrievalEvidence,
        *,
        remaining_message_budget: int,
    ) -> RecommendationReadiness:
        """Return an auditable clarify, recommend, refuse, or close decision."""
        ...


class ResponseComposer(Protocol):
    """Compose grounded conversational prose from trusted domain data."""

    async def compose(
        self,
        conversation: Conversation,
        requirements: ResolvedRequirements,
        readiness: RecommendationReadiness,
        recommendations: Sequence[Assessment],
    ) -> str:
        """Return reply text without creating catalog identities."""
        ...
