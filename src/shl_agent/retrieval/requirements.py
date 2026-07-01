"""Canonical requirement normalization for retrieval."""

import re
from dataclasses import dataclass

from shl_agent.models.enums import TestType
from shl_agent.models.recommendation import ResolvedRequirements

_TOKEN_SPLIT_RE = re.compile(r"[/,;|]+")
_DURATION_RE = re.compile(r"\d+")

SYNONYMS: dict[str, str] = {
    "js": "javascript",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "sjt": "situational judgment",
    "situational judgement": "situational judgment",
}

ROLE_SYNONYMS: dict[str, str] = {
    "dev": "developer",
    "engineer": "software engineer",
    "swe": "software engineer",
    "manager": "manager",
}

SENIORITY_SYNONYMS: dict[str, str] = {
    "junior": "entry-level",
    "entry": "entry-level",
    "entry level": "entry-level",
    "mid": "mid-professional",
    "senior": "senior",
    "lead": "lead",
    "graduate": "graduate",
}


@dataclass(frozen=True, slots=True)
class CanonicalRequirements:
    """Retrieval-ready representation of conversation requirements."""

    original_query: str
    role: str | None = None
    seniority: str | None = None
    skills: tuple[str, ...] = ()
    competencies: tuple[str, ...] = ()
    test_types: tuple[TestType, ...] = ()
    max_duration_minutes: int | None = None
    languages: tuple[str, ...] = ()
    assessment_names: tuple[str, ...] = ()
    excluded_assessment_ids: tuple[str, ...] = ()

    @property
    def anchors(self) -> tuple[str, ...]:
        """Return normalized terms that should be covered by candidates."""
        return tuple(
            item
            for item in (
                *(self.skills),
                *(self.competencies),
                *(self.assessment_names),
                *(self.test_types_to_labels()),
            )
            if item
        )

    def test_types_to_labels(self) -> tuple[str, ...]:
        """Return compact labels for requested test types."""
        return tuple(test_type.value.casefold() for test_type in self.test_types)


class RequirementResolver:
    """Normalize raw resolved requirements into deterministic retrieval inputs."""

    def resolve(self, requirements: ResolvedRequirements) -> CanonicalRequirements:
        """Return canonical requirements for retrieval services."""
        role = self._normalize_optional(requirements.role, ROLE_SYNONYMS)
        seniority = self._normalize_optional(requirements.seniority, SENIORITY_SYNONYMS)
        skills = self._normalize_terms(requirements.skills)
        competencies = self._normalize_terms(requirements.competencies)
        names = self._normalize_terms(requirements.assessment_names, preserve_symbols=True)
        query = self._query(requirements, role, seniority, skills, competencies, names)
        return CanonicalRequirements(
            original_query=query,
            role=role,
            seniority=seniority,
            skills=skills,
            competencies=competencies,
            test_types=tuple(dict.fromkeys(requirements.test_types)),
            max_duration_minutes=self._normalize_duration(requirements.max_duration_minutes),
            assessment_names=names,
            excluded_assessment_ids=tuple(
                sorted(
                    {item.strip() for item in requirements.excluded_assessment_ids if item.strip()}
                )
            ),
        )

    @classmethod
    def _normalize_terms(
        cls,
        values: tuple[str, ...],
        *,
        preserve_symbols: bool = False,
    ) -> tuple[str, ...]:
        terms: list[str] = []
        seen: set[str] = set()
        for value in values:
            for part in _TOKEN_SPLIT_RE.split(value):
                normalized = cls._normalize_text(part, preserve_symbols=preserve_symbols)
                normalized = SYNONYMS.get(normalized, normalized)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    terms.append(normalized)
        return tuple(terms)

    @classmethod
    def _normalize_optional(cls, value: str | None, synonyms: dict[str, str]) -> str | None:
        if value is None:
            return None
        normalized = cls._normalize_text(value)
        return synonyms.get(normalized, normalized) or None

    @staticmethod
    def _normalize_text(value: str, *, preserve_symbols: bool = False) -> str:
        text = value.casefold().replace("&", " and ").strip()
        allowed = r"[^a-z0-9+#.\s-]" if preserve_symbols else r"[^a-z0-9\s-]"
        text = re.sub(allowed, " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_duration(value: int | None) -> int | None:
        if value is None:
            return None
        match = _DURATION_RE.search(str(value))
        if match is None:
            return None
        minutes = int(match.group(0))
        return minutes if minutes > 0 else None

    @staticmethod
    def _query(
        requirements: ResolvedRequirements,
        role: str | None,
        seniority: str | None,
        skills: tuple[str, ...],
        competencies: tuple[str, ...],
        names: tuple[str, ...],
    ) -> str:
        parts = [
            *(names),
            *(skills),
            *(competencies),
            *(test_type.value for test_type in requirements.test_types),
            role or "",
            seniority or "",
        ]
        return " ".join(part for part in parts if part).strip() or "general assessment"
