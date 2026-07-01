"""Loading and validating the canonical catalog artifact."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType
from shl_agent.scraper.artifacts import checksum_payload


class CatalogArtifactError(ValueError):
    """Raised when catalog artifacts cannot be used for indexing."""


class CatalogArtifactRepository:
    """Load canonical assessments and catalog provenance from catalog.json."""

    def __init__(self, catalog_path: Path, manifest_path: Path | None = None) -> None:
        self._catalog_path = catalog_path
        self._manifest_path = manifest_path

    def load(self) -> tuple[tuple[Assessment, ...], str, str]:
        """Return assessments, catalog version, and validated catalog checksum."""
        payload = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise CatalogArtifactError("catalog.json must contain an object")

        catalog_version = self._required_str(payload, "catalog_version")
        records = payload.get("assessments")
        if not isinstance(records, list):
            raise CatalogArtifactError("catalog.json must contain an assessments list")

        assessments = tuple(self._assessment_from_json(record) for record in records)
        expected_count = payload.get("record_count")
        if expected_count != len(assessments):
            raise CatalogArtifactError("catalog record_count does not match assessments length")

        catalog_checksum = checksum_payload(payload)
        self._validate_manifest_checksum(catalog_checksum)
        return assessments, catalog_version, catalog_checksum

    def _validate_manifest_checksum(self, catalog_checksum: str) -> None:
        if self._manifest_path is None or not self._manifest_path.exists():
            return
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise CatalogArtifactError("catalog manifest must contain an object")
        expected = manifest.get("checksum")
        if expected != catalog_checksum:
            raise CatalogArtifactError("catalog checksum does not match catalog manifest")

    @classmethod
    def _assessment_from_json(cls, value: object) -> Assessment:
        if not isinstance(value, Mapping):
            raise CatalogArtifactError("assessment record must contain an object")
        return Assessment(
            assessment_id=cls._required_str(value, "assessment_id"),
            name=cls._required_str(value, "name"),
            url=cls._required_str(value, "url"),
            test_types=cls._test_types(value.get("test_types")),
            description=cls._required_str(value, "description"),
            duration_minutes=cls._optional_int(value.get("duration_minutes")),
            remote_testing=cls._optional_bool(value.get("remote_testing")),
            adaptive_irt=cls._optional_bool(value.get("adaptive_irt")),
            job_levels=cls._string_tuple(value.get("job_levels")),
            languages=cls._string_tuple(value.get("languages")),
        )

    @staticmethod
    def _required_str(payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CatalogArtifactError(f"{key} must be a non-empty string")
        return value

    @staticmethod
    def _test_types(value: object) -> tuple[TestType, ...]:
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise CatalogArtifactError("test_types must be a list")
        return tuple(TestType(str(item)) for item in value)

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int):
            raise CatalogArtifactError("duration_minutes must be an integer or null")
        return value

    @staticmethod
    def _optional_bool(value: object) -> bool | None:
        if value is None:
            return None
        if not isinstance(value, bool):
            raise CatalogArtifactError("boolean catalog fields must be true, false, or null")
        return value

    @staticmethod
    def _string_tuple(value: object) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise CatalogArtifactError("list field must be a list of strings")
        return tuple(str(item).strip() for item in value if str(item).strip())
