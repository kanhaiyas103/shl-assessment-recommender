"""Tests for the evaluation CLI helpers."""

import json
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from shl_agent.evaluation import cli
from shl_agent.evaluation.cli import _load_catalog_urls, build_parser

TRACE_MARKDOWN = "\n".join(
    [
        "## Conversation",
        "",
        "### Turn 1",
        "",
        "**User**",
        "",
        "> Need Java",
        "",
        "**Agent**",
        "",
        "| # | Name | Test Type | Keys | Duration | Languages | URL |",
        "|---|------|-----------|------|----------|-----------|-----|",
        (
            "| 1 | Core Java | K | Knowledge & Skills | 10 minutes | English (USA) | "
            "<https://www.shl.com/products/product-catalog/view/core-java-new/> |"
        ),
        "",
        "_`end_of_conversation`: **true**_",
    ]
)


def test_cli_parser_accepts_required_trace_zip() -> None:
    args = build_parser().parse_args(["--traces-zip", "sample.zip"])

    assert args.traces_zip == Path("sample.zip")
    assert args.output_dir == Path("data/evaluation")


def test_load_catalog_urls_reads_catalog_artifact(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({"assessments": [{"url": "https://www.shl.com/a/"}, {"name": "bad"}]}),
        encoding="utf-8",
    )

    assert _load_catalog_urls(catalog_path) == ("https://www.shl.com/a/",)


def test_main_replays_trace_and_writes_reports(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "assessments": [
                    {"url": "https://www.shl.com/products/product-catalog/view/core-java-new/"}
                ]
            }
        ),
        encoding="utf-8",
    )
    traces_zip = tmp_path / "traces.zip"
    with zipfile.ZipFile(traces_zip, "w") as archive:
        archive.writestr("GenAI_SampleConversations/C1.md", TRACE_MARKDOWN)

    class FakeSettings:
        def __init__(self) -> None:
            self.catalog_path = catalog_path
            self.max_conversation_messages = 8

    def fake_create_app() -> FastAPI:
        app = FastAPI()

        @app.post("/chat")
        async def chat(_payload: dict[str, list[dict[str, str]]]) -> dict[str, object]:
            return {
                "reply": "done",
                "recommendations": [
                    {
                        "name": "Core Java",
                        "url": "https://www.shl.com/products/product-catalog/view/core-java-new/",
                        "test_type": "K",
                    }
                ],
                "end_of_conversation": True,
            }

        return app

    monkeypatch.setattr(cli, "Settings", FakeSettings)
    monkeypatch.setattr(cli, "create_app", fake_create_app)

    status = cli.main(
        [
            "--traces-zip",
            str(traces_zip),
            "--output-dir",
            str(tmp_path / "evaluation"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "Evaluation complete" in captured.out
    assert (tmp_path / "evaluation" / "evaluation_report.json").exists()
