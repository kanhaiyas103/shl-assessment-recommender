"""Tests for embedding index metadata and manifest helpers."""

import json
from pathlib import Path

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.retrieval.index_artifacts import (
    EmbeddingManifest,
    EmbeddingReport,
    build_metadata_record,
    checksum_metadata,
    document_checksum,
    metadata_payload,
    write_index_artifacts,
)


def make_assessment() -> Assessment:
    return Assessment(
        assessment_id="python-new",
        name="Python New",
        url="https://www.shl.com/products/product-catalog/view/python-new/",
        test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
        description="Python skills.",
    )


def test_metadata_record_maps_vector_index_to_catalog_record() -> None:
    record = build_metadata_record(
        vector_index=0,
        assessment=make_assessment(),
        semantic_text="Assessment Name:\nPython New",
    )

    assert record.vector_index == 0
    assert record.assessment_id == "python-new"
    assert record.test_types == ("K",)
    assert record.document_checksum == document_checksum("Assessment Name:\nPython New")


def test_metadata_payload_checksum_is_stable() -> None:
    record = build_metadata_record(
        vector_index=0,
        assessment=make_assessment(),
        semantic_text="Assessment Name:\nPython New",
    )
    payload = metadata_payload(
        catalog_version="v1",
        catalog_checksum="sha256:catalog",
        embedding_model="fake-model",
        embedding_dimension=3,
        records=(record,),
    )

    assert payload["vector_count"] == 1
    assert checksum_metadata(payload).startswith("sha256:")


def test_write_index_artifacts_persists_json(tmp_path: Path) -> None:
    write_index_artifacts(
        metadata_path=tmp_path / "metadata.json",
        manifest_path=tmp_path / "manifest.json",
        report_path=tmp_path / "report.json",
        metadata={"vector_count": 1, "records": []},
        manifest=EmbeddingManifest(
            "fake-model",
            3,
            "v1",
            "sha256:catalog",
            1,
            "2026-07-01T00:00:00Z",
            "sha256:index",
        ),
        report=EmbeddingReport(1, 0, 0, 0.25, 0.25),
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert metadata["vector_count"] == 1
    assert manifest["vector_count"] == 1
    assert report["vectors_created"] == 1
