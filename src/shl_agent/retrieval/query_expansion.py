"""Deterministic query expansion for recall-oriented retrieval."""

from shl_agent.models.retrieval import ExpandedQuery
from shl_agent.retrieval.requirements import CanonicalRequirements

CATEGORY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "K": ("knowledge skills technical assessment", "programming software skills"),
    "A": ("ability aptitude cognitive reasoning",),
    "B": ("situational judgment biodata workplace judgment",),
    "C": ("competencies behavior workplace competency",),
    "D": ("development 360 feedback leadership",),
    "E": ("assessment exercise simulation work sample",),
    "P": ("personality behavior motivation preferences",),
    "S": ("simulation interactive job simulation",),
}

TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "javascript": ("js", "ecmascript", "frontend web development"),
    "node.js": ("nodejs", "node javascript backend"),
    "python": ("python programming", "python developer"),
    "java": ("java programming", "java developer"),
    "sql": ("database sql queries",),
    "leadership": ("people management", "team leadership"),
    "communication": ("verbal communication", "written communication"),
}


class QueryExpansionService:
    """Create bounded deterministic retrieval views without inventing requirements."""

    def __init__(self, max_expansions: int = 12) -> None:
        self._max_expansions = max_expansions

    def expand(self, requirements: CanonicalRequirements) -> tuple[ExpandedQuery, ...]:
        """Return stable expanded query views."""
        candidates: list[ExpandedQuery] = []
        self._append(candidates, requirements.original_query, 1.0, "original")
        if requirements.role:
            self._append(candidates, requirements.role, 0.85, "normalized_role")
        if requirements.seniority:
            self._append(candidates, requirements.seniority, 0.55, "seniority")
        for skill in requirements.skills:
            self._append(candidates, skill, 0.9, "technical_skill")
            for expansion in TERM_EXPANSIONS.get(skill, ()):
                self._append(candidates, expansion, 0.7, "skill_synonym")
        for competency in requirements.competencies:
            self._append(candidates, competency, 0.85, "behavioral_competency")
            for expansion in TERM_EXPANSIONS.get(competency, ()):
                self._append(candidates, expansion, 0.65, "competency_synonym")
        for name in requirements.assessment_names:
            self._append(candidates, name, 1.0, "assessment_name")
        for test_type in requirements.test_types:
            for expansion in CATEGORY_EXPANSIONS.get(test_type.value, ()):
                self._append(candidates, expansion, 0.65, "assessment_category")
        return tuple(candidates[: self._max_expansions])

    @staticmethod
    def _append(
        candidates: list[ExpandedQuery],
        text: str,
        weight: float,
        source: str,
    ) -> None:
        normalized = " ".join(text.split()).strip()
        if normalized and all(query.text != normalized for query in candidates):
            candidates.append(ExpandedQuery(normalized, weight, source))
