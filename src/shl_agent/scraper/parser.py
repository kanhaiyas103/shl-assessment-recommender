"""Parsing and normalization for SHL catalog records."""

import re
from collections.abc import Iterable, Mapping, Sequence
from html import unescape
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType
from shl_agent.scraper.constants import TEST_TYPE_CODES, TEST_TYPE_LABELS

_WHITESPACE_RE = re.compile(r"\s+")
_DURATION_RE = re.compile(r"(?P<minutes>\d+)")


class CatalogParseError(ValueError):
    """Raised when a source record cannot become a valid assessment."""


def normalize_text(value: object) -> str:
    """Return compact text with HTML entities and repeated whitespace normalized."""
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", unescape(str(value))).strip()


def parse_bool(value: object) -> bool | None:
    """Normalize common SHL yes/no values."""
    normalized = normalize_text(value).casefold()
    if normalized in {"yes", "y", "true", "1"}:
        return True
    if normalized in {"no", "n", "false", "0"}:
        return False
    return None


def parse_duration_minutes(*values: object) -> int | None:
    """Extract the first positive integer duration from source fields."""
    for value in values:
        match = _DURATION_RE.search(normalize_text(value))
        if match is not None:
            minutes = int(match.group("minutes"))
            if minutes > 0:
                return minutes
    return None


def normalize_string_sequence(value: object) -> tuple[str, ...]:
    """Normalize a list-like or comma-separated source field into unique strings."""
    if isinstance(value, str):
        candidates: Iterable[object] = value.split(",")
    elif isinstance(value, Iterable):
        candidates = value
    else:
        candidates = ()

    seen: set[str] = set()
    items: list[str] = []
    for candidate in candidates:
        text = normalize_text(candidate)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            items.append(text)
    return tuple(items)


def normalize_test_types(value: object) -> tuple[TestType, ...]:
    """Map SHL labels or one-letter codes to canonical test-type enums."""
    raw_items = normalize_string_sequence(value)
    if len(raw_items) == 1:
        code_parts = tuple(part for part in raw_items[0].split(" ") if part)
        if code_parts and all(part.upper() in TEST_TYPE_CODES for part in code_parts):
            raw_items = code_parts

    seen: set[TestType] = set()
    test_types: list[TestType] = []
    for item in raw_items:
        normalized = item.casefold()
        test_type = TEST_TYPE_LABELS.get(normalized) or TEST_TYPE_CODES.get(item.upper())
        if test_type is not None and test_type not in seen:
            seen.add(test_type)
            test_types.append(test_type)
    return tuple(test_types)


class OfficialCatalogRecordParser:
    """Parse assignment-provided catalog JSON records into canonical assessments."""

    def parse_record(self, record: Mapping[str, object]) -> Assessment:
        """Return a validated assessment from one official catalog JSON record."""
        return Assessment(
            assessment_id=normalize_text(record.get("entity_id")),
            name=normalize_text(record.get("name")),
            url=normalize_text(record.get("link")),
            test_types=normalize_test_types(record.get("keys")),
            description=normalize_text(record.get("description")),
            duration_minutes=parse_duration_minutes(
                record.get("duration"),
                record.get("duration_raw"),
            ),
            remote_testing=parse_bool(record.get("remote")),
            adaptive_irt=parse_bool(record.get("adaptive")),
            job_levels=normalize_string_sequence(
                record.get("job_levels") or record.get("job_levels_raw"),
            ),
            languages=normalize_string_sequence(
                record.get("languages") or record.get("languages_raw"),
            ),
        )


class BeautifulSoupAssessmentPageParser:
    """Parse an SHL Individual Test Solution detail page with BeautifulSoup."""

    def parse(self, *, url: str, html: str) -> Assessment:
        """Return a validated assessment from source HTML."""
        soup = BeautifulSoup(html, "html.parser")

        name = self._extract_name(soup)
        description = self._extract_description(soup)
        facts = self._extract_fact_table(soup)
        test_types = normalize_test_types(facts.get("test type") or facts.get("test types"))

        if not test_types:
            raise CatalogParseError(f"No test types found for {url}")

        return Assessment(
            assessment_id=self._slug_from_url(url),
            name=name,
            url=url,
            test_types=test_types,
            description=description,
            duration_minutes=parse_duration_minutes(facts.get("duration")),
            remote_testing=parse_bool(facts.get("remote testing")),
            adaptive_irt=parse_bool(facts.get("adaptive/irt")),
            job_levels=normalize_string_sequence(facts.get("job levels")),
            languages=normalize_string_sequence(facts.get("languages")),
        )

    def discover_links(self, *, base_url: str, html: str) -> tuple[str, ...]:
        """Discover catalog detail links from an HTML catalog page."""
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = normalize_text(anchor.get("href"))
            absolute = urljoin(base_url, href)
            if "/product-catalog/view/" in absolute and absolute not in seen:
                seen.add(absolute)
                links.append(absolute)
        return tuple(links)

    @staticmethod
    def _extract_name(soup: BeautifulSoup) -> str:
        heading = soup.find("h1")
        if heading is not None:
            return normalize_text(heading.get_text(" "))
        title = soup.find("title")
        if title is not None:
            return normalize_text(title.get_text(" ")).removesuffix("| SHL").strip()
        raise CatalogParseError("No assessment name found")

    @staticmethod
    def _extract_description(soup: BeautifulSoup) -> str:
        candidates: Sequence[str] = (
            ".product-description",
            ".description",
            "[data-testid='product-description']",
            "main p",
        )
        for selector in candidates:
            element = soup.select_one(selector)
            if element is not None:
                text = normalize_text(element.get_text(" "))
                if text:
                    return text
        raise CatalogParseError("No assessment description found")

    @staticmethod
    def _extract_fact_table(soup: BeautifulSoup) -> dict[str, str]:
        facts: dict[str, str] = {}
        for row in soup.select("tr"):
            cells = [normalize_text(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
            if len(cells) >= 2:
                facts[cells[0].casefold().rstrip(":")] = cells[1]

        for item in soup.select("li"):
            text = normalize_text(item.get_text(" "))
            if ":" in text:
                key, value = text.split(":", 1)
                facts[key.casefold().strip()] = value.strip()
        return facts

    @staticmethod
    def _slug_from_url(url: str) -> str:
        slug = url.rstrip("/").rsplit("/", maxsplit=1)[-1]
        if not slug:
            raise CatalogParseError(f"Cannot derive assessment ID from {url}")
        return slug
