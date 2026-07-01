"""Tests for SHL catalog parsing and normalization."""

from pathlib import Path

import pytest

from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.scraper.parser import (
    BeautifulSoupAssessmentPageParser,
    OfficialCatalogRecordParser,
    normalize_string_sequence,
    normalize_test_types,
    parse_bool,
    parse_duration_minutes,
)


def test_official_catalog_record_parser_normalizes_source_record() -> None:
    parser = OfficialCatalogRecordParser()

    assessment = parser.parse_record(
        {
            "entity_id": "  123  ",
            "name": " Python New ",
            "link": "https://www.shl.com/products/product-catalog/view/python-new/",
            "keys": ["Knowledge & Skills", "Simulations", "Knowledge & Skills"],
            "description": " Measures Python skills. ",
            "duration": "22 minutes",
            "remote": "yes",
            "adaptive": "no",
            "job_levels": ["Entry-Level", "Entry-Level", "Mid-Professional"],
            "languages_raw": "English, Spanish,",
        }
    )

    assert assessment.assessment_id == "123"
    assert assessment.test_types == (
        AssessmentTestType.KNOWLEDGE_AND_SKILLS,
        AssessmentTestType.SIMULATIONS,
    )
    assert assessment.duration_minutes == 22
    assert assessment.remote_testing is True
    assert assessment.adaptive_irt is False
    assert assessment.job_levels == ("Entry-Level", "Mid-Professional")


def test_beautiful_soup_parser_reads_detail_page_fixture() -> None:
    parser = BeautifulSoupAssessmentPageParser()
    html = Path("tests/fixtures/shl_assessment_detail.html").read_text(encoding="utf-8")

    assessment = parser.parse(
        url="https://www.shl.com/products/product-catalog/view/python-new/",
        html=html,
    )
    links = parser.discover_links(
        base_url="https://www.shl.com/products/product-catalog/",
        html=html,
    )

    assert assessment.assessment_id == "python-new"
    assert assessment.name == "Python New"
    assert assessment.test_types == (
        AssessmentTestType.KNOWLEDGE_AND_SKILLS,
        AssessmentTestType.SIMULATIONS,
    )
    assert assessment.duration_minutes == 22
    assert links == (
        "https://www.shl.com/products/product-catalog/view/java-new/",
        "https://www.shl.com/products/product-catalog/view/python-new/",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("yes", True),
        ("No", False),
        ("unknown", None),
    ],
)
def test_parse_bool(value: str, expected: bool | None) -> None:
    assert parse_bool(value) is expected


def test_normalizers_handle_empty_and_mixed_values() -> None:
    assert normalize_string_sequence("English, Spanish, English") == ("English", "Spanish")
    assert normalize_test_types("K S") == (
        AssessmentTestType.KNOWLEDGE_AND_SKILLS,
        AssessmentTestType.SIMULATIONS,
    )
    assert parse_duration_minutes("Variable", "Approx. 35 minutes") == 35
    assert parse_duration_minutes("Variable") is None
