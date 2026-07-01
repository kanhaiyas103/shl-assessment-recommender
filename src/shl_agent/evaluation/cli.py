"""Command line entry point for the SHL evaluation harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from shl_agent.api.app import create_app
from shl_agent.evaluation.metrics import EvaluationMetricsCalculator
from shl_agent.evaluation.replay import TraceReplayer
from shl_agent.evaluation.reports import EvaluationReportWriter
from shl_agent.evaluation.trace_parser import MarkdownTraceParser
from shl_agent.utils.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Replay SHL public conversation traces.")
    parser.add_argument(
        "--traces-zip",
        type=Path,
        required=True,
        help="Path to SHL public conversation traces ZIP.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Directory for evaluation_report.json and evaluation_summary.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the complete evaluation suite."""
    args = build_parser().parse_args(argv)
    settings = Settings()
    traces = MarkdownTraceParser().parse_zip(args.traces_zip)
    catalog_urls = _load_catalog_urls(settings.catalog_path)

    with TestClient(create_app()) as client:
        replay_results = TraceReplayer(
            client,
            max_messages=settings.max_conversation_messages,
        ).replay_many(traces)

    report = EvaluationMetricsCalculator(
        catalog_urls=catalog_urls,
        max_messages=settings.max_conversation_messages,
    ).report(replay_results)
    json_path, summary_path = EvaluationReportWriter().write(report, args.output_dir)
    sys.stdout.write(
        "Evaluation complete: "
        f"traces={report.trace_count}, "
        f"mean_recall_at_10={report.mean_recall_at_10:.3f}, "
        f"pass_rate={report.pass_rate:.3f}, "
        f"report={json_path}, "
        f"summary={summary_path}\n"
    )
    return 0


def _load_catalog_urls(catalog_path: Path) -> tuple[str, ...]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    assessments = payload.get("assessments", [])
    if not isinstance(assessments, list):
        return ()
    return tuple(
        item["url"]
        for item in assessments
        if isinstance(item, dict) and isinstance(item.get("url"), str)
    )


if __name__ == "__main__":
    raise SystemExit(main())
