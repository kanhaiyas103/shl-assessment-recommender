"""Typed evaluation models used by the replay harness."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExpectedRecommendation:
    """One expected catalog recommendation parsed from a public trace."""

    name: str
    url: str
    test_type: str


@dataclass(frozen=True, slots=True)
class TraceTurn:
    """One user turn plus reference metadata from a public trace."""

    turn_number: int
    user_message: str
    expected_reply: str
    expected_recommendations: tuple[ExpectedRecommendation, ...]
    expected_end_of_conversation: bool


@dataclass(frozen=True, slots=True)
class ConversationTrace:
    """A complete public conversation trace."""

    trace_id: str
    source_path: str
    turns: tuple[TraceTurn, ...]

    @property
    def expected_final_recommendations(self) -> tuple[ExpectedRecommendation, ...]:
        """Return the final non-empty expected recommendation set."""
        for turn in reversed(self.turns):
            if turn.expected_recommendations:
                return turn.expected_recommendations
        return ()


@dataclass(frozen=True, slots=True)
class ApiRecommendation:
    """One recommendation returned by the evaluated API."""

    name: str
    url: str
    test_type: str


@dataclass(frozen=True, slots=True)
class ApiTurnResult:
    """Observed API response for one replayed turn."""

    turn_number: int
    request_messages: int
    reply: str
    recommendations: tuple[ApiRecommendation, ...]
    end_of_conversation: bool
    latency_ms: float
    schema_compliant: bool
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class TraceReplayResult:
    """Observed replay result for one trace."""

    trace: ConversationTrace
    turns: tuple[ApiTurnResult, ...]
    stopped_reason: str
    failures: tuple[str, ...] = ()

    @property
    def final_recommendations(self) -> tuple[ApiRecommendation, ...]:
        """Return the final observed recommendation set."""
        for turn in reversed(self.turns):
            if turn.recommendations:
                return turn.recommendations
        return ()


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    """Per-trace evaluation metrics."""

    trace_id: str
    recall_at_10: float
    expected_recommendation_count: int
    returned_recommendation_count: int
    matched_recommendation_count: int
    turn_count: int
    completed: bool
    completion_within_limit: bool
    clarification_count: int
    unnecessary_clarification_count: int
    schema_compliant: bool
    recommendation_count_valid: bool
    hallucination_count: int
    hallucination_rate: float
    refusal_accuracy: float | None
    refinement_accuracy: float | None
    comparison_accuracy: float | None
    average_latency_ms: float
    failures: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Complete evaluation report."""

    trace_count: int
    mean_recall_at_10: float
    pass_rate: float
    schema_compliance_rate: float
    hallucination_rate: float
    average_turn_count: float
    average_clarification_count: float
    unnecessary_clarification_rate: float
    average_latency_ms: float
    traces: tuple[TraceMetrics, ...]
