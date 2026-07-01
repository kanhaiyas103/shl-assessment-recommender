"""Tests for Markdown-in-ZIP trace parsing."""

import zipfile
from pathlib import Path

import pytest

from shl_agent.evaluation.trace_parser import MarkdownTraceParser, TraceParseError

TRACE_MARKDOWN = "\n".join(
    [
        "## Conversation",
        "",
        "### Turn 1",
        "",
        "**User**",
        "",
        "> Hiring a Java developer.",
        "",
        "**Agent**",
        "",
        "Here are options.",
        "",
        "| # | Name | Test Type | Keys | Duration | Languages | URL |",
        "|---|------|-----------|------|----------|-----------|-----|",
        (
            "| 1 | Core Java (New) | K | Knowledge & Skills | 10 minutes | "
            "English (USA) | <https://www.shl.com/products/product-catalog/view/"
            "core-java-new/> |"
        ),
        "",
        "_`end_of_conversation`: **true**_",
    ]
)


def test_parse_markdown_trace_extracts_turn_and_recommendation() -> None:
    trace = MarkdownTraceParser().parse_markdown("GenAI_SampleConversations/C1.md", TRACE_MARKDOWN)

    assert trace.trace_id == "C1"
    assert len(trace.turns) == 1
    assert trace.turns[0].user_message == "Hiring a Java developer."
    assert trace.turns[0].expected_end_of_conversation is True
    assert trace.expected_final_recommendations[0].name == "Core Java (New)"
    assert (
        trace.expected_final_recommendations[0].url
        == "https://www.shl.com/products/product-catalog/view/core-java-new/"
    )


def test_parse_zip_reads_all_markdown_files(tmp_path: Path) -> None:
    zip_path = tmp_path / "traces.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("GenAI_SampleConversations/C1.md", TRACE_MARKDOWN)
        archive.writestr("GenAI_SampleConversations/readme.txt", "ignored")

    traces = MarkdownTraceParser().parse_zip(zip_path)

    assert [trace.trace_id for trace in traces] == ["C1"]


def test_parse_zip_rejects_empty_archives(tmp_path: Path) -> None:
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("readme.txt", "no traces")

    with pytest.raises(TraceParseError):
        MarkdownTraceParser().parse_zip(zip_path)


def test_parse_markdown_rejects_missing_end_flag() -> None:
    markdown = """## Conversation

### Turn 1

**User**

> Hi

**Agent**

Hello
"""

    with pytest.raises(TraceParseError):
        MarkdownTraceParser().parse_markdown("C1.md", markdown)
