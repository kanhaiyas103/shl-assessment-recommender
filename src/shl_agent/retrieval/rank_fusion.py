"""Reciprocal Rank Fusion for hybrid retrieval signals."""

from dataclasses import dataclass

from shl_agent.retrieval.metadata_filter import MetadataSignal
from shl_agent.retrieval.semantic_retriever import CandidateSignal


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """Candidate after rank fusion, before final ranking."""

    assessment_id: str
    fused_score: float
    semantic_score: float
    lexical_score: float
    metadata_score: float
    semantic_rank: int | None
    lexical_rank: int | None
    matched_terms: tuple[str, ...]


class RankFusionService:
    """Fuse semantic, lexical, and metadata signals with RRF."""

    def __init__(self, rrf_k: int = 60) -> None:
        self._rrf_k = rrf_k

    def fuse(
        self,
        *,
        semantic: dict[str, CandidateSignal],
        lexical: dict[str, CandidateSignal],
        metadata: dict[str, MetadataSignal],
    ) -> tuple[FusedCandidate, ...]:
        """Return candidates ordered by reciprocal rank fusion score."""
        ids = set(semantic) | set(lexical) | set(metadata)
        candidates: list[FusedCandidate] = []
        max_rrf = (1 / (self._rrf_k + 1)) * 3
        for assessment_id in ids:
            semantic_signal = semantic.get(assessment_id)
            lexical_signal = lexical.get(assessment_id)
            metadata_signal = metadata.get(assessment_id)
            raw = 0.0
            if semantic_signal is not None:
                raw += 1 / (self._rrf_k + semantic_signal.rank)
            if lexical_signal is not None:
                raw += 1 / (self._rrf_k + lexical_signal.rank)
            if metadata_signal is not None:
                metadata_rank = max(1, round((1 - metadata_signal.score) * 100))
                raw += 1 / (self._rrf_k + metadata_rank)
            matched = tuple(
                sorted(
                    {
                        *(semantic_signal.matched_terms if semantic_signal else ()),
                        *(lexical_signal.matched_terms if lexical_signal else ()),
                        *(metadata_signal.matched_requirements if metadata_signal else ()),
                    }
                )
            )
            candidates.append(
                FusedCandidate(
                    assessment_id=assessment_id,
                    fused_score=min(1.0, raw / max_rrf),
                    semantic_score=semantic_signal.score if semantic_signal else 0.0,
                    lexical_score=lexical_signal.score if lexical_signal else 0.0,
                    metadata_score=metadata_signal.score if metadata_signal else 0.0,
                    semantic_rank=semantic_signal.rank if semantic_signal else None,
                    lexical_rank=lexical_signal.rank if lexical_signal else None,
                    matched_terms=matched,
                )
            )
        return tuple(sorted(candidates, key=lambda item: (-item.fused_score, item.assessment_id)))
