"""Runtime catalog, metadata, and FAISS loading for retrieval."""

import json
from pathlib import Path

from shl_agent.models.assessment import Assessment
from shl_agent.retrieval.catalog_artifact import CatalogArtifactRepository


class RetrievalStoreError(ValueError):
    """Raised when retrieval artifacts are inconsistent."""


class CatalogMetadataStore:
    """In-memory catalog and vector metadata store."""

    def __init__(
        self,
        catalog_path: Path,
        catalog_manifest_path: Path,
        metadata_path: Path,
    ) -> None:
        assessments, _, _ = CatalogArtifactRepository(catalog_path, catalog_manifest_path).load()
        self._assessments_by_id = {
            assessment.assessment_id: assessment for assessment in assessments
        }
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = metadata.get("records")
        if not isinstance(records, list):
            raise RetrievalStoreError("metadata records must be a list")
        self._ids_by_vector_index: dict[int, str] = {}
        for expected_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise RetrievalStoreError("metadata record must be an object")
            vector_index = int(record["vector_index"])
            assessment_id = str(record["assessment_id"])
            if vector_index != expected_index:
                raise RetrievalStoreError("metadata vector ordering is not deterministic")
            if assessment_id not in self._assessments_by_id:
                raise RetrievalStoreError("metadata references unknown assessment")
            self._ids_by_vector_index[vector_index] = assessment_id
        if len(self._ids_by_vector_index) != len(self._assessments_by_id):
            raise RetrievalStoreError("metadata count must equal catalog count")

    def assessment(self, assessment_id: str) -> Assessment:
        """Return one assessment by ID."""
        return self._assessments_by_id[assessment_id]

    def all_assessments(self) -> tuple[Assessment, ...]:
        """Return all assessments in deterministic name order."""
        return tuple(
            sorted(self._assessments_by_id.values(), key=lambda item: item.name.casefold())
        )

    def id_for_vector_index(self, vector_index: int) -> str:
        """Return the assessment ID mapped to a FAISS vector position."""
        return self._ids_by_vector_index[vector_index]

    def contains(self, assessment_id: str) -> bool:
        """Return whether the catalog contains this ID."""
        return assessment_id in self._assessments_by_id
