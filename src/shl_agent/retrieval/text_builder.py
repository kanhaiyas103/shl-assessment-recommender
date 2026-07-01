"""Semantic document construction for catalog embedding generation."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.-]{2,}")
_STOPWORDS = frozenset(
    {
        "and",
        "are",
        "assessment",
        "designed",
        "for",
        "from",
        "knowledge",
        "measure",
        "measures",
        "of",
        "the",
        "this",
        "test",
        "that",
        "with",
    }
)

TEST_TYPE_LABELS: dict[TestType, str] = {
    TestType.ABILITY_AND_APTITUDE: "Ability and Aptitude",
    TestType.BIODATA_AND_SITUATIONAL_JUDGEMENT: "Biodata and Situational Judgment",
    TestType.COMPETENCIES: "Competencies",
    TestType.DEVELOPMENT_AND_360: "Development and 360",
    TestType.ASSESSMENT_EXERCISES: "Assessment Exercises",
    TestType.KNOWLEDGE_AND_SKILLS: "Knowledge and Skills",
    TestType.PERSONALITY_AND_BEHAVIOR: "Personality and Behavior",
    TestType.SIMULATIONS: "Simulations",
}


@dataclass(frozen=True, slots=True)
class SemanticDocument:
    """One deterministic embedding document mapped to one assessment."""

    assessment_id: str
    text: str


class AssessmentTextBuilder:
    """Build deterministic semantic documents from canonical assessments."""

    def build(self, assessment: Assessment) -> SemanticDocument:
        """Build a labeled document that maximizes semantic search quality."""
        sections: list[tuple[str, str]] = []
        self._append(sections, "Assessment Name", assessment.name)
        self._append(sections, "Description", assessment.description)
        self._append(sections, "Test Type", self._join(self._test_type_labels(assessment)))
        self._append(sections, "Job Levels", self._join(assessment.job_levels))
        self._append(sections, "Languages", self._join(assessment.languages))
        self._append(sections, "Remote Testing", self._format_bool(assessment.remote_testing))
        self._append(sections, "Adaptive / IRT Support", self._format_bool(assessment.adaptive_irt))
        self._append(sections, "Duration", self._format_duration(assessment.duration_minutes))
        self._append(sections, "Keywords", self._join(self._keywords(assessment)))

        text = "\n\n".join(f"{label}:\n{value}" for label, value in sections)
        return SemanticDocument(assessment_id=assessment.assessment_id, text=text)

    @staticmethod
    def _append(sections: list[tuple[str, str]], label: str, value: str | None) -> None:
        if value is not None and value.strip():
            sections.append((label, value.strip()))

    @staticmethod
    def _join(values: Iterable[str]) -> str | None:
        normalized = tuple(value.strip() for value in values if value.strip())
        if not normalized:
            return None
        return ", ".join(normalized)

    @staticmethod
    def _format_bool(value: bool | None) -> str | None:
        if value is None:
            return None
        return "Yes" if value else "No"

    @staticmethod
    def _format_duration(value: int | None) -> str | None:
        if value is None:
            return None
        return f"{value} minutes"

    @staticmethod
    def _test_type_labels(assessment: Assessment) -> tuple[str, ...]:
        return tuple(TEST_TYPE_LABELS[test_type] for test_type in assessment.test_types)

    def _keywords(self, assessment: Assessment) -> tuple[str, ...]:
        source = " ".join(
            (
                assessment.name,
                assessment.description,
                " ".join(self._test_type_labels(assessment)),
                " ".join(assessment.job_levels),
            )
        )
        seen: set[str] = set()
        keywords: list[str] = []
        for match in _WORD_RE.finditer(source):
            token = match.group(0).strip(".,;:").casefold()
            if token not in _STOPWORDS and token not in seen:
                seen.add(token)
                keywords.append(token)
        return tuple(keywords[:24])
