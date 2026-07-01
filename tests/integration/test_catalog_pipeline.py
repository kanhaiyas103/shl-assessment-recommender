"""Integration tests for the offline catalog pipeline."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from shl_agent.scraper.official_catalog import OfficialCatalogSource
from shl_agent.scraper.parser import OfficialCatalogRecordParser
from shl_agent.scraper.pipeline import CatalogPipeline, CatalogUrlPolicy


class FakeFetcher:
    """In-memory fetcher for deterministic pipeline tests."""

    def __init__(self, records: Sequence[Mapping[str, object]]) -> None:
        self._records = records

    async def fetch_text(self, url: str) -> str:
        assert url == "https://example.com/catalog.json"
        return json.dumps(self._records)


def valid_record(*, entity_id: str, name: str, link: str) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "name": name,
        "link": link,
        "keys": ["Knowledge & Skills"],
        "description": f"{name} description.",
        "duration": "15 minutes",
        "remote": "yes",
        "adaptive": "no",
        "job_levels": ["Entry-Level"],
        "languages": ["English"],
    }


@pytest.mark.asyncio
async def test_catalog_pipeline_writes_validated_deduplicated_outputs(tmp_path: Path) -> None:
    records: list[Mapping[str, object]] = [
        valid_record(
            entity_id="1",
            name="Python New",
            link="https://www.shl.com/products/product-catalog/view/python-new/",
        ),
        valid_record(
            entity_id="1b",
            name="Python New Duplicate",
            link="https://www.shl.com/products/product-catalog/view/python-new/",
        ),
        valid_record(entity_id="2", name="Invalid Host", link="https://example.com/not-shl/"),
        {"entity_id": "3", "name": "Missing Fields"},
    ]
    source = OfficialCatalogSource(
        source_url="https://example.com/catalog.json",
        raw_output_path=tmp_path / "raw.json",
        fetcher=FakeFetcher(records),
    )
    pipeline = CatalogPipeline(
        source=source,
        record_parser=OfficialCatalogRecordParser(),
        url_policy=CatalogUrlPolicy(),
        catalog_path=tmp_path / "catalog.json",
        manifest_path=tmp_path / "manifest.json",
        report_path=tmp_path / "report.json",
        catalog_version="test-version",
    )

    assessments, report, manifest = await pipeline.build()

    assert len(assessments) == 1
    assert report.total_assessments_discovered == 4
    assert report.duplicate_count == 1
    assert report.missing_field_count == 1
    assert report.final_catalog_size == 1
    assert manifest.catalog_version == "test-version"
    assert manifest.record_count == 1
    assert json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))["record_count"] == 1
    manifest_payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["checksum"].startswith("sha256:")


def test_catalog_url_policy_accepts_only_shl_catalog_detail_urls() -> None:
    policy = CatalogUrlPolicy()

    assert policy.is_individual_test_url(
        "https://www.shl.com/products/product-catalog/view/python-new/"
    )
    assert policy.is_individual_test_url(
        "https://www.shl.com/solutions/products/product-catalog/view/python-new/"
    )
    assert not policy.is_individual_test_url("https://www.shl.com/products/")
    assert not policy.is_individual_test_url("http://www.shl.com/products/product-catalog/view/x/")
