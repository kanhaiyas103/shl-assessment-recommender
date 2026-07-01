"""Build calibrated RetrievalEvidence from ranked candidates."""

from shl_agent.models.retrieval import RetrievalEvidence, RetrievalResult
from shl_agent.retrieval.requirements import CanonicalRequirements


class RetrievalEvidenceBuilder:
    """Create evidence signals for the future recommendation readiness policy."""

    def build(
        self,
        *,
        results: tuple[RetrievalResult, ...],
        requirements: CanonicalRequirements,
    ) -> RetrievalEvidence:
        """Return retrieval evidence with confidence and coverage metrics."""
        self._validate(results)
        scores = tuple(round(result.rerank_score, 6) for result in results)
        top_score = scores[0] if scores else 0.0
        margin = round(top_score - scores[1], 6) if len(scores) > 1 else top_score
        skill_coverage = self._requirement_coverage(results, requirements)
        constraint_coverage = self._constraint_coverage(results)
        confidence = min(
            1.0,
            top_score * 0.55
            + skill_coverage * 0.25
            + constraint_coverage * 0.15
            + max(0, margin) * 0.05,
        )
        return RetrievalEvidence(
            results=results,
            retrieval_confidence=round(confidence, 6),
            required_skill_coverage=round(skill_coverage, 6),
            constraint_coverage=round(constraint_coverage, 6),
            top_score_margin=margin,
            top_score=top_score,
            score_distribution=scores,
            matched_catalog_ids=tuple(result.assessment.assessment_id for result in results),
            candidate_explanations=tuple(result.explanation for result in results),
        )

    @staticmethod
    def _validate(results: tuple[RetrievalResult, ...]) -> None:
        ids = [result.assessment.assessment_id for result in results]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieval evidence must not contain duplicate candidates")
        for result in results:
            if not result.assessment.url.startswith("https://www.shl.com/"):
                raise ValueError("retrieval evidence candidate URL must exist in SHL catalog")

    @staticmethod
    def _requirement_coverage(
        results: tuple[RetrievalResult, ...],
        requirements: CanonicalRequirements,
    ) -> float:
        anchors = requirements.anchors
        if not anchors:
            return 0.5 if results else 0.0
        matched = " ".join(term for result in results[:10] for term in result.matched_requirements)
        covered = sum(1 for anchor in anchors if anchor in matched)
        return min(1.0, covered / len(anchors))

    @staticmethod
    def _constraint_coverage(results: tuple[RetrievalResult, ...]) -> float:
        if not results:
            return 0.0
        return sum(result.metadata_match_score for result in results[:10]) / min(10, len(results))
