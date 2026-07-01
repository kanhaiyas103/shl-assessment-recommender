"""Query expansion and retrieval evidence models."""

from dataclasses import dataclass

from shl_agent.models.assessment import Assessment


@dataclass(frozen=True, slots=True)
class ExpandedQuery:
    """A bounded retrieval view derived from supported user requirements."""

    text: str
    weight: float
    source: str

    def __post_init__(self) -> None:
        """Validate rank-fusion inputs."""
        if not self.text.strip():
            raise ValueError("expanded query text must not be blank")
        if not 0 < self.weight <= 1:
            raise ValueError("expanded query weight must be in (0, 1]")
        if not self.source.strip():
            raise ValueError("expanded query source must not be blank")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One catalog candidate with transparent component scores."""

    assessment: Assessment
    semantic_score: float
    lexical_score: float
    metadata_match_score: float
    fused_score: float
    rerank_score: float = 0.0
    matched_requirements: tuple[str, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        """Keep normalized scores comparable and calibratable."""
        scores = (
            self.semantic_score,
            self.lexical_score,
            self.metadata_match_score,
            self.fused_score,
            self.rerank_score,
        )
        if any(not 0 <= score <= 1 for score in scores):
            raise ValueError("retrieval scores must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RetrievalEvidence:
    """Candidate set and query-level signals used by readiness policy."""

    results: tuple[RetrievalResult, ...]
    retrieval_confidence: float
    required_skill_coverage: float
    constraint_coverage: float
    top_score_margin: float
    top_score: float = 0.0
    score_distribution: tuple[float, ...] = ()
    matched_catalog_ids: tuple[str, ...] = ()
    candidate_explanations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate calibrated query-level evidence."""
        normalized = (
            self.retrieval_confidence,
            self.required_skill_coverage,
            self.constraint_coverage,
        )
        if any(not 0 <= value <= 1 for value in normalized):
            raise ValueError("coverage and confidence values must be in [0, 1]")
        if not -1 <= self.top_score_margin <= 1:
            raise ValueError("top_score_margin must be in [-1, 1]")
