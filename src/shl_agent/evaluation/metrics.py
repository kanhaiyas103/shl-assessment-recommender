"""Metric calculations for SHL conversation replay results."""

from __future__ import annotations

from collections.abc import Iterable

from shl_agent.evaluation.models import (
    ApiRecommendation,
    EvaluationReport,
    TraceMetrics,
    TraceReplayResult,
)

_REFUSAL_TERMS = (
    "legal",
    "compliance",
    "outside",
    "cannot advise",
    "can't advise",
    "can only help",
    "not interpret",
)
_REFINEMENT_TERMS = ("add", "drop", "remove", "replace", "instead", "updated")
_COMPARISON_TERMS = ("difference", "compare", " vs ", " versus ")


class EvaluationMetricsCalculator:
    """Compute aggregate and per-trace evaluation metrics."""

    def __init__(self, catalog_urls: Iterable[str], max_messages: int) -> None:
        self._catalog_urls = frozenset(_normalize_url(url) for url in catalog_urls)
        self._max_messages = max_messages

    def trace_metrics(self, result: TraceReplayResult) -> TraceMetrics:
        """Compute metrics for one replayed trace."""
        expected_urls = {
            _normalize_url(item.url) for item in result.trace.expected_final_recommendations
        }
        returned_urls = [_normalize_url(item.url) for item in result.final_recommendations[:10]]
        matched = expected_urls.intersection(returned_urls)
        recall = len(matched) / len(expected_urls) if expected_urls else 1.0
        hallucinations = self._hallucinations(result.final_recommendations)
        schema_compliant = all(turn.schema_compliant for turn in result.turns)
        recommendation_count_valid = self._recommendation_counts_are_valid(result)
        clarification_count = sum(
            1 for turn in result.turns if not turn.recommendations and "?" in turn.reply
        )
        unnecessary_clarifications = self._unnecessary_clarifications(result)
        latencies = [turn.latency_ms for turn in result.turns]
        failures = (*result.failures, *self._turn_failures(result))
        return TraceMetrics(
            trace_id=result.trace.trace_id,
            recall_at_10=recall,
            expected_recommendation_count=len(expected_urls),
            returned_recommendation_count=len(returned_urls),
            matched_recommendation_count=len(matched),
            turn_count=len(result.turns),
            completed=bool(result.turns and result.turns[-1].end_of_conversation),
            completion_within_limit=all(
                turn.request_messages <= self._max_messages for turn in result.turns
            ),
            clarification_count=clarification_count,
            unnecessary_clarification_count=unnecessary_clarifications,
            schema_compliant=schema_compliant,
            recommendation_count_valid=recommendation_count_valid,
            hallucination_count=hallucinations,
            hallucination_rate=hallucinations / len(returned_urls) if returned_urls else 0.0,
            refusal_accuracy=self._refusal_accuracy(result),
            refinement_accuracy=self._refinement_accuracy(result, recall),
            comparison_accuracy=self._comparison_accuracy(result),
            average_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            failures=failures,
        )

    def report(self, results: Iterable[TraceReplayResult]) -> EvaluationReport:
        """Compute the complete aggregate report."""
        traces = tuple(self.trace_metrics(result) for result in results)
        trace_count = len(traces)
        pass_count = sum(1 for trace in traces if _trace_passed(trace))
        total_hallucinations = sum(trace.hallucination_count for trace in traces)
        total_recommendations = sum(trace.returned_recommendation_count for trace in traces)
        total_clarifications = sum(trace.clarification_count for trace in traces)
        total_unnecessary = sum(trace.unnecessary_clarification_count for trace in traces)
        return EvaluationReport(
            trace_count=trace_count,
            mean_recall_at_10=_mean(trace.recall_at_10 for trace in traces),
            pass_rate=pass_count / trace_count if trace_count else 0.0,
            schema_compliance_rate=_mean(
                1.0 if trace.schema_compliant else 0.0 for trace in traces
            ),
            hallucination_rate=(
                total_hallucinations / total_recommendations if total_recommendations else 0.0
            ),
            average_turn_count=_mean(float(trace.turn_count) for trace in traces),
            average_clarification_count=(
                total_clarifications / trace_count if trace_count else 0.0
            ),
            unnecessary_clarification_rate=(
                total_unnecessary / total_clarifications if total_clarifications else 0.0
            ),
            average_latency_ms=_mean(trace.average_latency_ms for trace in traces),
            traces=traces,
        )

    def _hallucinations(self, recommendations: Iterable[ApiRecommendation]) -> int:
        return sum(
            1
            for recommendation in recommendations
            if _normalize_url(recommendation.url) not in self._catalog_urls
        )

    @staticmethod
    def _recommendation_counts_are_valid(result: TraceReplayResult) -> bool:
        for observed, expected in zip(result.turns, result.trace.turns, strict=False):
            count = len(observed.recommendations)
            if expected.expected_recommendations:
                if not 1 <= count <= 10:
                    return False
            elif count != 0:
                return False
        return True

    @staticmethod
    def _unnecessary_clarifications(result: TraceReplayResult) -> int:
        count = 0
        for observed, expected in zip(result.turns, result.trace.turns, strict=False):
            if (
                expected.expected_recommendations
                and not observed.recommendations
                and "?" in observed.reply
            ):
                count += 1
        return count

    @staticmethod
    def _turn_failures(result: TraceReplayResult) -> tuple[str, ...]:
        return tuple(
            f"turn {turn.turn_number}: {turn.failure}" for turn in result.turns if turn.failure
        )

    @staticmethod
    def _refusal_accuracy(result: TraceReplayResult) -> float | None:
        refusal_turns = [
            turn
            for turn in result.trace.turns
            if any(term in turn.user_message.casefold() for term in ("legally", "required under"))
        ]
        if not refusal_turns:
            return None
        observed_by_turn = {turn.turn_number: turn for turn in result.turns}
        correct = 0
        for expected in refusal_turns:
            observed = observed_by_turn.get(expected.turn_number)
            if (
                observed
                and not observed.recommendations
                and any(term in observed.reply.casefold() for term in _REFUSAL_TERMS)
            ):
                correct += 1
        return correct / len(refusal_turns)

    @staticmethod
    def _refinement_accuracy(result: TraceReplayResult, recall: float) -> float | None:
        has_refinement = any(
            any(term in turn.user_message.casefold() for term in _REFINEMENT_TERMS)
            for turn in result.trace.turns
        )
        return recall if has_refinement else None

    @staticmethod
    def _comparison_accuracy(result: TraceReplayResult) -> float | None:
        comparison_turns = [
            turn
            for turn in result.trace.turns
            if any(term in turn.user_message.casefold() for term in _COMPARISON_TERMS)
        ]
        if not comparison_turns:
            return None
        observed_by_turn = {turn.turn_number: turn for turn in result.turns}
        correct = 0
        for expected in comparison_turns:
            observed = observed_by_turn.get(expected.turn_number)
            if observed and observed.schema_compliant and observed.failure is None:
                correct += 1
        return correct / len(comparison_turns)


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    return sum(items) / len(items) if items else 0.0


def _trace_passed(trace: TraceMetrics) -> bool:
    return (
        trace.schema_compliant
        and trace.recommendation_count_valid
        and trace.hallucination_count == 0
        and trace.completion_within_limit
        and not trace.failures
    )
