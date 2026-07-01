"""Tests for evaluation metric calculations."""

from shl_agent.evaluation.metrics import EvaluationMetricsCalculator
from shl_agent.evaluation.models import (
    ApiRecommendation,
    ApiTurnResult,
    ConversationTrace,
    ExpectedRecommendation,
    TraceReplayResult,
    TraceTurn,
)


def test_recall_at_10_uses_expected_final_urls() -> None:
    trace = ConversationTrace(
        trace_id="C1",
        source_path="C1.md",
        turns=(
            TraceTurn(
                turn_number=1,
                user_message="Need Java",
                expected_reply="",
                expected_recommendations=(
                    ExpectedRecommendation("Java", "https://www.shl.com/a/", "K"),
                    ExpectedRecommendation("SQL", "https://www.shl.com/b/", "K"),
                ),
                expected_end_of_conversation=True,
            ),
        ),
    )
    result = TraceReplayResult(
        trace=trace,
        turns=(
            ApiTurnResult(
                turn_number=1,
                request_messages=1,
                reply="Done",
                recommendations=(ApiRecommendation("Java", "https://www.shl.com/a/", "K"),),
                end_of_conversation=True,
                latency_ms=10,
                schema_compliant=True,
            ),
        ),
        stopped_reason="end_of_conversation",
    )

    metrics = EvaluationMetricsCalculator(
        catalog_urls=("https://www.shl.com/a/", "https://www.shl.com/b/"),
        max_messages=8,
    ).trace_metrics(result)

    assert metrics.recall_at_10 == 0.5
    assert metrics.matched_recommendation_count == 1
    assert metrics.hallucination_count == 0


def test_hallucination_counts_urls_outside_catalog() -> None:
    trace = ConversationTrace(
        trace_id="C2",
        source_path="C2.md",
        turns=(
            TraceTurn(
                turn_number=1,
                user_message="Need Java",
                expected_reply="",
                expected_recommendations=(),
                expected_end_of_conversation=True,
            ),
        ),
    )
    result = TraceReplayResult(
        trace=trace,
        turns=(
            ApiTurnResult(
                turn_number=1,
                request_messages=1,
                reply="Done",
                recommendations=(ApiRecommendation("Fake", "https://example.com/fake", "K"),),
                end_of_conversation=True,
                latency_ms=10,
                schema_compliant=True,
            ),
        ),
        stopped_reason="end_of_conversation",
    )

    metrics = EvaluationMetricsCalculator(catalog_urls=(), max_messages=8).trace_metrics(result)

    assert metrics.hallucination_count == 1
    assert metrics.hallucination_rate == 1.0


def test_report_aggregates_behavior_metrics() -> None:
    trace = ConversationTrace(
        trace_id="C7",
        source_path="C7.md",
        turns=(
            TraceTurn(
                turn_number=1,
                user_message="Are we legally required under HIPAA?",
                expected_reply="",
                expected_recommendations=(),
                expected_end_of_conversation=False,
            ),
            TraceTurn(
                turn_number=2,
                user_message="Add Docker",
                expected_reply="",
                expected_recommendations=(
                    ExpectedRecommendation("Docker", "https://www.shl.com/docker/", "K"),
                ),
                expected_end_of_conversation=True,
            ),
        ),
    )
    result = TraceReplayResult(
        trace=trace,
        turns=(
            ApiTurnResult(
                turn_number=1,
                request_messages=1,
                reply="That is a legal compliance question outside what I can advise on.",
                recommendations=(),
                end_of_conversation=False,
                latency_ms=5,
                schema_compliant=True,
            ),
            ApiTurnResult(
                turn_number=2,
                request_messages=3,
                reply="Which Docker skill?",
                recommendations=(),
                end_of_conversation=False,
                latency_ms=7,
                schema_compliant=True,
            ),
        ),
        stopped_reason="trace_exhausted",
    )

    report = EvaluationMetricsCalculator(
        catalog_urls=("https://www.shl.com/docker/",),
        max_messages=8,
    ).report((result,))

    trace_metrics = report.traces[0]
    assert report.trace_count == 1
    assert trace_metrics.refusal_accuracy == 1.0
    assert trace_metrics.refinement_accuracy == 0.0
    assert trace_metrics.unnecessary_clarification_count == 1
