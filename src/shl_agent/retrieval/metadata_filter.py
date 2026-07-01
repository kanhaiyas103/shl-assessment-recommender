"""Recall-friendly metadata scoring and constraint handling."""

from dataclasses import dataclass

from shl_agent.models.assessment import Assessment
from shl_agent.retrieval.requirements import CanonicalRequirements


@dataclass(frozen=True, slots=True)
class MetadataSignal:
    """Metadata relevance and matched deterministic constraints."""

    assessment_id: str
    score: float
    matched_requirements: tuple[str, ...]
    constraint_checks: int
    constraints_satisfied: int


class MetadataFilter:
    """Apply soft deterministic constraints while preserving recall."""

    def score(self, assessment: Assessment, requirements: CanonicalRequirements) -> MetadataSignal:
        """Return metadata match signal; excluded IDs are scored as zero."""
        if assessment.assessment_id in requirements.excluded_assessment_ids:
            return MetadataSignal(assessment.assessment_id, 0.0, ("excluded",), 1, 0)

        checks = 0
        satisfied = 0
        matched: list[str] = []
        checks, satisfied = self._duration(assessment, requirements, checks, satisfied, matched)
        checks, satisfied = self._test_types(assessment, requirements, checks, satisfied, matched)
        checks, satisfied = self._seniority(assessment, requirements, checks, satisfied, matched)
        checks, satisfied = self._languages(assessment, requirements, checks, satisfied, matched)

        if checks == 0:
            return MetadataSignal(assessment.assessment_id, 0.5, tuple(matched), 0, 0)
        return MetadataSignal(
            assessment.assessment_id,
            max(0.05, satisfied / checks),
            tuple(matched),
            checks,
            satisfied,
        )

    @staticmethod
    def _duration(
        assessment: Assessment,
        requirements: CanonicalRequirements,
        checks: int,
        satisfied: int,
        matched: list[str],
    ) -> tuple[int, int]:
        if requirements.max_duration_minutes is None:
            return checks, satisfied
        checks += 1
        if (
            assessment.duration_minutes is None
            or assessment.duration_minutes <= requirements.max_duration_minutes
        ):
            satisfied += 1
            matched.append("duration")
        return checks, satisfied

    @staticmethod
    def _test_types(
        assessment: Assessment,
        requirements: CanonicalRequirements,
        checks: int,
        satisfied: int,
        matched: list[str],
    ) -> tuple[int, int]:
        if not requirements.test_types:
            return checks, satisfied
        checks += 1
        if set(assessment.test_types).intersection(requirements.test_types):
            satisfied += 1
            matched.append("test_type")
        return checks, satisfied

    @staticmethod
    def _seniority(
        assessment: Assessment,
        requirements: CanonicalRequirements,
        checks: int,
        satisfied: int,
        matched: list[str],
    ) -> tuple[int, int]:
        if requirements.seniority is None:
            return checks, satisfied
        checks += 1
        levels = " ".join(assessment.job_levels).casefold()
        if requirements.seniority in levels or not levels:
            satisfied += 1
            matched.append("seniority")
        return checks, satisfied

    @staticmethod
    def _languages(
        assessment: Assessment,
        requirements: CanonicalRequirements,
        checks: int,
        satisfied: int,
        matched: list[str],
    ) -> tuple[int, int]:
        if not requirements.languages:
            return checks, satisfied
        checks += 1
        languages = " ".join(assessment.languages).casefold()
        if any(language in languages for language in requirements.languages) or not languages:
            satisfied += 1
            matched.append("language")
        return checks, satisfied
