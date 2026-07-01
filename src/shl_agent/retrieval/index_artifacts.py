"""Serialization helpers for embedding index artifacts."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shl_agent.models.assessment import Assessment
from shl_agent.scraper.artifacts import checksum_payload


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    """Metadata for one FAISS vector position."""

    vector_index: int
    assessment_id: str
    name: str
    url: str
    test_types: tuple[str, ...]
    document_checksum: str


@dataclass(frozen=True, slots=True)
class EmbeddingManifest:
    """Manifest proving catalog-to-index compatibility."""

    embedding_model: str
    embedding_dimension: int
    catalog_version: str
    catalog_checksum: str
    vector_count: int
    generation_timestamp: str
    index_checksum: str


@dataclass(frozen=True, slots=True)
class EmbeddingReport:
    """Operational report for one index build."""

    vectors_created: int
    failed_embeddings: int
    skipped_embeddings: int
    embedding_duration_seconds: float
    average_latency_seconds: float


def utc_now_iso() -> str:
    """Return a UTC timestamp suitable for manifests."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_checksum(path: Path) -> str:
    """Return the sha256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def document_checksum(text: str) -> str:
    """Return the deterministic checksum of a semantic document."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def build_metadata_record(
    *,
    vector_index: int,
    assessment: Assessment,
    semantic_text: str,
) -> MetadataRecord:
    """Build metadata that deterministically maps a vector to a catalog record."""
    return MetadataRecord(
        vector_index=vector_index,
        assessment_id=assessment.assessment_id,
        name=assessment.name,
        url=assessment.url,
        test_types=tuple(test_type.value for test_type in assessment.test_types),
        document_checksum=document_checksum(semantic_text),
    )


def metadata_payload(
    *,
    catalog_version: str,
    catalog_checksum: str,
    embedding_model: str,
    embedding_dimension: int,
    records: tuple[MetadataRecord, ...],
) -> dict[str, Any]:
    """Build the metadata.json payload."""
    return {
        "catalog_version": catalog_version,
        "catalog_checksum": catalog_checksum,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "vector_count": len(records),
        "records": [asdict(record) for record in records],
    }


def write_json(path: Path, payload: object) -> None:
    """Write stable JSON artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_index_artifacts(
    *,
    metadata_path: Path,
    manifest_path: Path,
    report_path: Path,
    metadata: dict[str, Any],
    manifest: EmbeddingManifest,
    report: EmbeddingReport,
) -> None:
    """Persist metadata, manifest, and report JSON artifacts."""
    write_json(metadata_path, metadata)
    write_json(manifest_path, asdict(manifest))
    write_json(report_path, asdict(report))


def checksum_metadata(payload: object) -> str:
    """Expose deterministic JSON checksum for metadata tests."""
    return checksum_payload(payload)
