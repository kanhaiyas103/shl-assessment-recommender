"""Tests for catalog artifact loading and checksum validation."""

import json
from pathlib import Path

import pytest

from shl_agent.retrieval.catalog_artifact import CatalogArtifactError, CatalogArtifactRepository
from shl_agent.scraper.artifacts import checksum_payload, write_json


def catalog_payload() -> dict[str, object]:
    return {
        "catalog_version": "v1",
        "generated_at": "2026-07-01T00:00:00Z",
        "source_url": "https://example.com/catalog.json",
        "record_count": 1,
        "assessments": [
            {
                "assessment_id": "python-new",
                "name": "Python New",
                "url": "https://www.shl.com/products/product-catalog/view/python-new/",
                "test_types": ["K"],
                "description": "Python skills.",
                "duration_minutes": 22,
                "remote_testing": True,
                "adaptive_irt": False,
                "job_levels": ["Entry-Level"],
                "languages": ["English"],
            }
        ],
    }


def test_catalog_repository_loads_and_validates_manifest_checksum(tmp_path: Path) -> None:
    payload = catalog_payload()
    catalog_path = tmp_path / "catalog.json"
    manifest_path = tmp_path / "catalog_manifest.json"
    write_json(catalog_path, payload)
    write_json(manifest_path, {"checksum": checksum_payload(payload)})

    assessments, catalog_version, catalog_checksum = CatalogArtifactRepository(
        catalog_path,
        manifest_path,
    ).load()

    assert len(assessments) == 1
    assert assessments[0].assessment_id == "python-new"
    assert catalog_version == "v1"
    assert catalog_checksum == checksum_payload(payload)


def test_catalog_repository_rejects_count_mismatch(tmp_path: Path) -> None:
    payload = catalog_payload()
    payload["record_count"] = 2
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CatalogArtifactError, match="record_count"):
        CatalogArtifactRepository(catalog_path).load()


def test_catalog_repository_rejects_manifest_checksum_mismatch(tmp_path: Path) -> None:
    payload = catalog_payload()
    catalog_path = tmp_path / "catalog.json"
    manifest_path = tmp_path / "catalog_manifest.json"
    write_json(catalog_path, payload)
    write_json(manifest_path, {"checksum": "sha256:wrong"})

    with pytest.raises(CatalogArtifactError, match="checksum"):
        CatalogArtifactRepository(catalog_path, manifest_path).load()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"catalog_version": "v1", "record_count": 0},
        {"catalog_version": "v1", "record_count": 0, "assessments": {}},
    ],
)
def test_catalog_repository_rejects_invalid_payloads(tmp_path: Path, payload: object) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CatalogArtifactError):
        CatalogArtifactRepository(catalog_path).load()
