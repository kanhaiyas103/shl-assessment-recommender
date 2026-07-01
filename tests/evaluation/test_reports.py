"""Tests for evaluation report generation."""

from pathlib import Path

from shl_agent.evaluation.models import EvaluationReport, TraceMetrics
from shl_agent.evaluation.reports import EvaluationReportWriter


def test_report_writer_creates_json_and_markdown(tmp_path: Path) -> None:
    report = EvaluationReport(
        trace_count=1,
        mean_recall_at_10=1.0,
        pass_rate=1.0,
        schema_compliance_rate=1.0,
        hallucination_rate=0.0,
        average_turn_count=1.0,
        average_clarification_count=0.0,
        unnecessary_clarification_rate=0.0,
        average_latency_ms=3.0,
        traces=(
            TraceMetrics(
                trace_id="C1",
                recall_at_10=1.0,
                expected_recommendation_count=1,
                returned_recommendation_count=1,
                matched_recommendation_count=1,
                turn_count=1,
                completed=True,
                completion_within_limit=True,
                clarification_count=0,
                unnecessary_clarification_count=0,
                schema_compliant=True,
                recommendation_count_valid=True,
                hallucination_count=0,
                hallucination_rate=0.0,
                refusal_accuracy=None,
                refinement_accuracy=None,
                comparison_accuracy=None,
                average_latency_ms=3.0,
            ),
        ),
    )

    json_path, markdown_path = EvaluationReportWriter().write(report, tmp_path)

    assert json_path.read_text(encoding="utf-8").startswith("{")
    assert "Overall Mean Recall@10: 1.000" in markdown_path.read_text(encoding="utf-8")
