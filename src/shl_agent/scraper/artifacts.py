"""Catalog artifact serialization, checksums, and reports."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shl_agent.models.assessment import Assessment


@dataclass(frozen=True, slots=True)
class CatalogManifest:
    """Small immutable manifest for validating a generated catalog artifact."""

    catalog_version: str
    record_count: int
    generation_timestamp: str
    checksum: str


@dataclass(frozen=True, slots=True)
class CatalogBuildReport:
    """Human-readable summary of one catalog build."""

    source_url: str
    total_assessments_discovered: int
    skipped_pages: tuple[str, ...]
    duplicate_count: int
    missing_field_count: int
    final_catalog_size: int


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def assessment_to_json(assessment: Assessment) -> dict[str, Any]:
    """Convert a domain assessment into stable JSON-compatible data."""
    return {
        "assessment_id": assessment.assessment_id,
        "name": assessment.name,
        "url": assessment.url,
        "test_types": [test_type.value for test_type in assessment.test_types],
        "description": assessment.description,
        "duration_minutes": assessment.duration_minutes,
        "remote_testing": assessment.remote_testing,
        "adaptive_irt": assessment.adaptive_irt,
        "job_levels": list(assessment.job_levels),
        "languages": list(assessment.languages),
    }


def build_catalog_payload(
    *,
    catalog_version: str,
    generated_at: str,
    source_url: str,
    assessments: tuple[Assessment, ...],
) -> dict[str, Any]:
    """Build the versioned catalog JSON payload."""
    return {
        "catalog_version": catalog_version,
        "generated_at": generated_at,
        "source_url": source_url,
        "record_count": len(assessments),
        "assessments": [assessment_to_json(assessment) for assessment in assessments],
    }


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize JSON deterministically for repeatable checksums."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def checksum_payload(payload: object) -> str:
    """Return a sha256 checksum for a JSON payload."""
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def write_json(path: Path, payload: object) -> None:
    """Write JSON atomically enough for local artifact generation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_catalog_outputs(
    *,
    catalog_path: Path,
    manifest_path: Path,
    report_path: Path,
    catalog_payload: dict[str, Any],
    manifest: CatalogManifest,
    report: CatalogBuildReport,
) -> None:
    """Persist catalog, manifest, and build report artifacts."""
    write_json(catalog_path, catalog_payload)
    write_json(manifest_path, asdict(manifest))
    write_json(report_path, asdict(report))
