"""Final deterministic candidate ranking before LLM-independent evidence."""

from typing import Protocol

from shl_agent.models.assessment import Assessment
from shl_agent.models.retrieval import RetrievalResult
from shl_agent.retrieval.rank_fusion import FusedCandidate
from shl_agent.retrieval.requirements import CanonicalRequirements


class CandidateStore(Protocol):
    """Catalog lookup capability required by ranking."""

    def assessment(self, assessment_id: str) -> Assessment:
        """Return one assessment."""
        raise NotImplementedError

    def contains(self, assessment_id: str) -> bool:
        """Return whether an assessment exists."""
        raise NotImplementedError


class CandidateRanker:
    """Compute rerank scores and retain the Top-30 candidate set."""

    def __init__(self, store: CandidateStore, internal_limit: int = 30) -> None:
        self._store = store
        self._internal_limit = internal_limit

    def rank(
        self,
        candidates: tuple[FusedCandidate, ...],
        requirements: CanonicalRequirements,
    ) -> tuple[RetrievalResult, ...]:
        """Return top candidates with transparent score components."""
        results: list[RetrievalResult] = []
        seen_families: set[str] = set()
        for candidate in candidates:
            if not self._store.contains(candidate.assessment_id):
                continue
            assessment = self._store.assessment(candidate.assessment_id)
            family = self._family(assessment.name)
            diversity = 0.85 if family in seen_families else 1.0
            coverage = self._coverage(candidate.matched_terms, requirements)
            balance = self._balance(candidate.semantic_score, candidate.lexical_score)
            support = self._support_boost(assessment, candidate, requirements)
            rerank = min(
                1.0,
                (
                    candidate.fused_score * 0.35
                    + coverage * 0.2
                    + candidate.metadata_score * 0.15
                    + balance * 0.1
                    + diversity * 0.05
                    + support * 0.25
                ),
            )
            seen_families.add(family)
            explanation = self._explanation(candidate, coverage, support)
            results.append(
                RetrievalResult(
                    assessment=assessment,
                    semantic_score=candidate.semantic_score,
                    lexical_score=candidate.lexical_score,
                    metadata_match_score=candidate.metadata_score,
                    fused_score=candidate.fused_score,
                    rerank_score=rerank,
                    matched_requirements=candidate.matched_terms,
                    explanation=explanation,
                )
            )
        return tuple(
            sorted(results, key=lambda item: (-item.rerank_score, item.assessment.name.casefold()))[
                : self._internal_limit
            ]
        )

    @staticmethod
    def _coverage(matched_terms: tuple[str, ...], requirements: CanonicalRequirements) -> float:
        anchors = requirements.anchors
        if not anchors:
            return 0.5
        covered = sum(
            1
            for anchor in anchors
            if any(anchor in matched or matched in anchor for matched in matched_terms)
        )
        return min(1.0, covered / len(anchors))

    @staticmethod
    def _balance(semantic_score: float, lexical_score: float) -> float:
        if semantic_score and lexical_score:
            return 1.0
        if semantic_score or lexical_score:
            return 0.7
        return 0.3

    @classmethod
    def _support_boost(
        cls,
        assessment: Assessment,
        candidate: FusedCandidate,
        requirements: CanonicalRequirements,
    ) -> float:
        """Return deterministic boost for exact and dual-channel support."""
        normalized_name = cls._normalize(assessment.name)
        normalized_text = cls._normalize(
            " ".join((assessment.name, assessment.description, *assessment.job_levels))
        )
        score = 0.0
        if any(cls._normalize(name) in normalized_name for name in requirements.assessment_names):
            score += 0.45
        exact_skill_matches = sum(
            1
            for skill in requirements.skills
            if cls._contains_term(normalized_text, cls._normalize(skill))
        )
        if exact_skill_matches:
            score += min(0.35, 0.12 * exact_skill_matches)
        if candidate.semantic_score >= 0.2 and candidate.lexical_score >= 0.5:
            score += 0.25
        if any(
            cls._contains_term(normalized_name, cls._normalize(term))
            for term in candidate.matched_terms
        ):
            score += 0.15
        return min(1.0, score)

    @staticmethod
    def _family(name: str) -> str:
        return name.casefold().replace("(new)", "").split()[0]

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().replace("&", " and ").split())

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        if not term:
            return False
        return f" {term} " in f" {text} "

    @staticmethod
    def _explanation(candidate: FusedCandidate, coverage: float, support: float) -> str:
        matched = ", ".join(candidate.matched_terms[:5]) or "semantic similarity"
        return (
            f"Matched {matched}; semantic={candidate.semantic_score:.2f}, "
            f"lexical={candidate.lexical_score:.2f}, metadata={candidate.metadata_score:.2f}, "
            f"coverage={coverage:.2f}, support={support:.2f}."
        )
