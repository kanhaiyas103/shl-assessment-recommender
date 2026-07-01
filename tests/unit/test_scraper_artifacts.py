"""Tests for catalog artifact serialization."""

import json
from pathlib import Path

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.scraper.artifacts import (
    CatalogBuildReport,
    CatalogManifest,
    assessment_to_json,
    build_catalog_payload,
    checksum_payload,
    write_catalog_outputs,
)


def make_assessment() -> Assessment:
    return Assessment(
        assessment_id="python-new",
        name="Python New",
        url="https://www.shl.com/products/product-catalog/view/python-new/",
        test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
        description="Python skills.",
    )


def test_catalog_payload_and_checksum_are_stable() -> None:
    assessment = make_assessment()
    payload = build_catalog_payload(
        catalog_version="v1",
        generated_at="2026-06-30T00:00:00Z",
        source_url="https://example.com/catalog.json",
        assessments=(assessment,),
    )

    assert assessment_to_json(assessment)["test_types"] == ["K"]
    assert payload["record_count"] == 1
    assert checksum_payload(payload).startswith("sha256:")
    assert checksum_payload(payload) == checksum_payload(dict(reversed(payload.items())))


def test_write_catalog_outputs_persists_all_files(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    payload = {"record_count": 1, "assessments": []}

    write_catalog_outputs(
        catalog_path=catalog_path,
        manifest_path=manifest_path,
        report_path=report_path,
        catalog_payload=payload,
        manifest=CatalogManifest("v1", 1, "2026-06-30T00:00:00Z", "sha256:abc"),
        report=CatalogBuildReport("https://example.com", 1, (), 0, 0, 1),
    )

    assert json.loads(catalog_path.read_text(encoding="utf-8")) == payload
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["record_count"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["final_catalog_size"] == 1
