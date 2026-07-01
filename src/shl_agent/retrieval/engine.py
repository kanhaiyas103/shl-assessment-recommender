"""Hybrid retrieval engine orchestration."""

from shl_agent.models.recommendation import ResolvedRequirements
from shl_agent.models.retrieval import RetrievalEvidence
from shl_agent.retrieval.candidate_ranker import CandidateRanker, CandidateStore
from shl_agent.retrieval.evidence_builder import RetrievalEvidenceBuilder
from shl_agent.retrieval.lexical_retriever import LexicalRetriever
from shl_agent.retrieval.metadata_filter import MetadataFilter
from shl_agent.retrieval.query_expansion import QueryExpansionService
from shl_agent.retrieval.rank_fusion import RankFusionService
from shl_agent.retrieval.requirements import RequirementResolver
from shl_agent.retrieval.semantic_retriever import SemanticRetriever


class HybridRetrievalEngine:
    """Produce RetrievalEvidence from resolved requirements."""

    def __init__(
        self,
        *,
        requirement_resolver: RequirementResolver,
        query_expander: QueryExpansionService,
        semantic_retriever: SemanticRetriever,
        lexical_retriever: LexicalRetriever,
        metadata_filter: MetadataFilter,
        rank_fusion: RankFusionService,
        candidate_ranker: CandidateRanker,
        evidence_builder: RetrievalEvidenceBuilder,
        store: CandidateStore,
    ) -> None:
        self._requirement_resolver = requirement_resolver
        self._query_expander = query_expander
        self._semantic_retriever = semantic_retriever
        self._lexical_retriever = lexical_retriever
        self._metadata_filter = metadata_filter
        self._rank_fusion = rank_fusion
        self._candidate_ranker = candidate_ranker
        self._evidence_builder = evidence_builder
        self._store = store

    async def retrieve(self, requirements: ResolvedRequirements) -> RetrievalEvidence:
        """Run the full deterministic hybrid retrieval pipeline."""
        canonical = self._requirement_resolver.resolve(requirements)
        queries = self._query_expander.expand(canonical)
        semantic = await self._semantic_retriever.retrieve(queries)
        lexical = self._lexical_retriever.retrieve(canonical)
        candidate_ids = set(semantic) | set(lexical)
        metadata = {
            assessment_id: self._metadata_filter.score(
                self._store.assessment(assessment_id),
                canonical,
            )
            for assessment_id in candidate_ids
            if self._store.contains(assessment_id)
        }
        fused = self._rank_fusion.fuse(semantic=semantic, lexical=lexical, metadata=metadata)
        ranked = self._candidate_ranker.rank(fused, canonical)
        return self._evidence_builder.build(results=ranked, requirements=canonical)
