"""Parser for SHL public Markdown conversation traces packaged as a ZIP."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from shl_agent.evaluation.models import (
    ConversationTrace,
    ExpectedRecommendation,
    TraceTurn,
)

_TURN_SPLIT_RE = re.compile(r"^### Turn\s+(\d+)\s*$", re.MULTILINE)
_END_RE = re.compile(r"`end_of_conversation`:\s*\*\*(true|false)\*\*", re.IGNORECASE)
_URL_RE = re.compile(r"https://www\.shl\.com/products/product-catalog/view/[^>\s|]+")


class TraceParseError(ValueError):
    """Raised when a public trace cannot be parsed safely."""


class MarkdownTraceParser:
    """Parse the actual public trace format: Markdown files inside a ZIP."""

    def parse_zip(self, zip_path: Path) -> tuple[ConversationTrace, ...]:
        """Parse every Markdown trace in a ZIP archive."""
        if not zip_path.exists():
            raise TraceParseError(f"trace ZIP does not exist: {zip_path}")
        traces: list[ConversationTrace] = []
        with zipfile.ZipFile(zip_path) as archive:
            for name in sorted(archive.namelist()):
                if not name.casefold().endswith(".md"):
                    continue
                raw_text = archive.read(name).decode("utf-8-sig")
                traces.append(self.parse_markdown(name, raw_text))
        if not traces:
            raise TraceParseError("trace ZIP contains no Markdown conversation files")
        return tuple(traces)

    def parse_markdown(self, source_path: str, text: str) -> ConversationTrace:
        """Parse one Markdown trace into typed turns."""
        parts = _TURN_SPLIT_RE.split(text)
        if len(parts) < 3:
            raise TraceParseError(f"{source_path} contains no turn sections")
        turns: list[TraceTurn] = []
        for index in range(1, len(parts), 2):
            turn_number = int(parts[index])
            body = parts[index + 1]
            turns.append(
                TraceTurn(
                    turn_number=turn_number,
                    user_message=self._extract_user_message(source_path, turn_number, body),
                    expected_reply=self._extract_agent_reply(body),
                    expected_recommendations=self._extract_recommendations(body),
                    expected_end_of_conversation=self._extract_end_flag(
                        source_path, turn_number, body
                    ),
                )
            )
        trace_id = Path(source_path).stem
        return ConversationTrace(trace_id=trace_id, source_path=source_path, turns=tuple(turns))

    @staticmethod
    def _extract_user_message(source_path: str, turn_number: int, body: str) -> str:
        marker = re.search(r"\*\*User\*\*\s*(?P<content>.*?)\s*\*\*Agent\*\*", body, re.DOTALL)
        if marker is None:
            raise TraceParseError(f"{source_path} turn {turn_number} has no user block")
        return _clean_blockquote(marker.group("content"))

    @staticmethod
    def _extract_agent_reply(body: str) -> str:
        marker = re.search(r"\*\*Agent\*\*\s*(?P<content>.*)", body, re.DOTALL)
        if marker is None:
            return ""
        content = marker.group("content")
        metadata_start = content.find("_`end_of_conversation`")
        if metadata_start >= 0:
            content = content[:metadata_start]
        return content.strip()

    @staticmethod
    def _extract_end_flag(source_path: str, turn_number: int, body: str) -> bool:
        match = _END_RE.search(body)
        if match is None:
            raise TraceParseError(f"{source_path} turn {turn_number} has no end flag")
        return match.group(1).casefold() == "true"

    @staticmethod
    def _extract_recommendations(body: str) -> tuple[ExpectedRecommendation, ...]:
        recommendations: list[ExpectedRecommendation] = []
        for line in body.splitlines():
            if not line.startswith("|") or "shl.com/products/product-catalog/view" not in line:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 7:
                continue
            url_match = _URL_RE.search(cells[-1])
            if url_match is None:
                continue
            recommendations.append(
                ExpectedRecommendation(
                    name=cells[1],
                    test_type=cells[2],
                    url=url_match.group(0),
                )
            )
        return tuple(recommendations)


def _clean_blockquote(text: str) -> str:
    lines: list[str] = []
    for line in text.strip().splitlines():
        cleaned = line.strip()
        if cleaned.startswith(">"):
            cleaned = cleaned[1:].strip()
        lines.append(cleaned)
    return "\n".join(lines).strip()
