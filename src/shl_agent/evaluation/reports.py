"""Evaluation report serialization."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from shl_agent.evaluation.models import EvaluationReport, TraceMetrics


class EvaluationReportWriter:
    """Write machine-readable and human-readable evaluation reports."""

    def write(self, report: EvaluationReport, output_dir: Path) -> tuple[Path, Path]:
        """Write JSON and Markdown reports."""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "evaluation_report.json"
        summary_path = output_dir / "evaluation_summary.md"
        json_path.write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary_path.write_text(self._summary_markdown(report), encoding="utf-8")
        return json_path, summary_path

    @staticmethod
    def _summary_markdown(report: EvaluationReport) -> str:
        worst = sorted(report.traces, key=lambda trace: trace.recall_at_10)[:3]
        best = sorted(report.traces, key=lambda trace: trace.recall_at_10, reverse=True)[:3]
        lines = [
            "# SHL Evaluation Summary",
            "",
            f"- Trace count: {report.trace_count}",
            f"- Overall Mean Recall@10: {report.mean_recall_at_10:.3f}",
            f"- Pass rate: {report.pass_rate:.3f}",
            f"- Schema compliance: {report.schema_compliance_rate:.3f}",
            f"- Hallucination rate: {report.hallucination_rate:.3f}",
            f"- Average turn count: {report.average_turn_count:.2f}",
            f"- Average clarification count: {report.average_clarification_count:.2f}",
            f"- Unnecessary clarification rate: {report.unnecessary_clarification_rate:.3f}",
            f"- Average latency: {report.average_latency_ms:.2f} ms",
            "",
            "## Behavior summary",
            "",
            "The harness replays each public trace through POST /chat using complete stateless "
            "conversation history and scores the final recommendation set against the reference "
            "catalog URLs in the trace.",
            "",
            "## Hallucination summary",
            "",
            (
                "No hallucinated recommendations were detected."
                if report.hallucination_rate == 0
                else "One or more returned URLs were not present in the catalog."
            ),
            "",
            "## Worst-performing traces",
            "",
            *_trace_lines(worst),
            "",
            "## Best-performing traces",
            "",
            *_trace_lines(best),
            "",
            "## Suggestions for improvement",
            "",
            "- Inspect low-recall traces and compare missing expected URLs against "
            "retrieval evidence.",
            "- Reduce unnecessary clarifications when the reference trace already recommends.",
            "- Keep hallucination rate at zero by preserving catalog-only output validation.",
            "",
        ]
        return "\n".join(lines)


def _trace_lines(traces: list[TraceMetrics]) -> list[str]:
    lines: list[str] = []
    for trace in traces:
        suffix = f" failures={len(trace.failures)}" if trace.failures else ""
        lines.append(f"- {trace.trace_id}: Recall@10={trace.recall_at_10:.3f}{suffix}")
    return lines or ["- No traces."]
