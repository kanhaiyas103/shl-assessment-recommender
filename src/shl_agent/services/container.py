"""Typed dependency composition for application-scoped infrastructure."""

import logging
from dataclasses import dataclass

from shl_agent.retrieval.candidate_ranker import CandidateRanker
from shl_agent.retrieval.embedding_service import EmbeddingService, SentenceTransformerBackend
from shl_agent.retrieval.engine import HybridRetrievalEngine
from shl_agent.retrieval.evidence_builder import RetrievalEvidenceBuilder
from shl_agent.retrieval.lexical_retriever import LexicalRetriever
from shl_agent.retrieval.metadata_filter import MetadataFilter
from shl_agent.retrieval.query_expansion import QueryExpansionService
from shl_agent.retrieval.rank_fusion import RankFusionService
from shl_agent.retrieval.requirements import RequirementResolver
from shl_agent.retrieval.runtime_store import CatalogMetadataStore
from shl_agent.retrieval.semantic_retriever import FaissSearchIndex, SemanticRetriever
from shl_agent.services.conversation_engine import (
    ComparisonEngine,
    ConversationEngine,
    ConversationHistoryResolver,
    ConversationPolicy,
    GroundedResponseComposer,
    OutputValidator,
    RecommendationReadinessPolicy,
)
from shl_agent.utils.logging import configure_logging
from shl_agent.utils.settings import Settings


@dataclass(frozen=True, slots=True)
class FoundationContainer:
    """Dependencies available before application services are implemented."""

    settings: Settings
    logger: logging.Logger


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    """Fully wired runtime dependency graph."""

    settings: Settings
    logger: logging.Logger
    catalog_store: CatalogMetadataStore
    embedding_service: EmbeddingService
    retrieval_engine: HybridRetrievalEngine
    conversation_engine: ConversationEngine
    provider_status: str


def build_foundation_container(settings: Settings | None = None) -> FoundationContainer:
    """Build application-scoped foundation dependencies in one composition root."""
    resolved_settings = settings or Settings()
    configure_logging(resolved_settings.log_level)
    return FoundationContainer(
        settings=resolved_settings,
        logger=logging.getLogger("shl_agent"),
    )


def build_application_container(settings: Settings | None = None) -> ApplicationContainer:
    """Build and validate the complete runtime object graph once at startup."""
    foundation = build_foundation_container(settings)
    resolved_settings = foundation.settings

    store = CatalogMetadataStore(
        resolved_settings.catalog_path,
        resolved_settings.catalog_manifest_path,
        resolved_settings.index_metadata_path,
    )
    embedding_service = EmbeddingService(
        SentenceTransformerBackend(
            model_name=resolved_settings.embedding_model,
            batch_size=resolved_settings.embedding_batch_size,
        )
    )
    _ = embedding_service.dimension
    search_index = FaissSearchIndex(resolved_settings.faiss_index_path, store)
    semantic_retriever = SemanticRetriever(
        embedding_service=embedding_service,
        search_index=search_index,
        per_query_limit=resolved_settings.retrieval_candidate_limit,
    )
    lexical_retriever = LexicalRetriever(store)
    metadata_filter = MetadataFilter()
    rank_fusion = RankFusionService()
    candidate_ranker = CandidateRanker(store)
    retrieval_engine = HybridRetrievalEngine(
        requirement_resolver=RequirementResolver(),
        query_expander=QueryExpansionService(),
        semantic_retriever=semantic_retriever,
        lexical_retriever=lexical_retriever,
        metadata_filter=metadata_filter,
        rank_fusion=rank_fusion,
        candidate_ranker=candidate_ranker,
        evidence_builder=RetrievalEvidenceBuilder(),
        store=store,
    )
    conversation_engine = ConversationEngine(
        history_resolver=ConversationHistoryResolver(),
        retrieval_engine=retrieval_engine,
        readiness_policy=RecommendationReadinessPolicy(),
        conversation_policy=ConversationPolicy(
            max_messages=resolved_settings.max_conversation_messages
        ),
        comparison_engine=ComparisonEngine(store),
        composer=GroundedResponseComposer(),
        output_validator=OutputValidator(),
    )
    return ApplicationContainer(
        settings=resolved_settings,
        logger=foundation.logger,
        catalog_store=store,
        embedding_service=embedding_service,
        retrieval_engine=retrieval_engine,
        conversation_engine=conversation_engine,
        provider_status=resolved_settings.llm_provider.value,
    )
