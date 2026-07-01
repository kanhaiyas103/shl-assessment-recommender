"""End-to-end embedding and FAISS index build pipeline."""

import time
from pathlib import Path
from typing import Protocol

from shl_agent.models.assessment import Assessment
from shl_agent.retrieval.embedding_service import EmbeddingService, FloatMatrix
from shl_agent.retrieval.index_artifacts import (
    EmbeddingManifest,
    EmbeddingReport,
    MetadataRecord,
    build_metadata_record,
    file_checksum,
    metadata_payload,
    utc_now_iso,
    write_index_artifacts,
)
from shl_agent.retrieval.text_builder import AssessmentTextBuilder


class IndexBuildError(ValueError):
    """Raised when index artifacts fail consistency checks."""


class CatalogLoader(Protocol):
    """Catalog loading capability required by the index pipeline."""

    def load(self) -> tuple[tuple[Assessment, ...], str, str]:
        """Return assessments, catalog version, and catalog checksum."""
        ...


class IndexWriter(Protocol):
    """Index writing capability required by the index pipeline."""

    def build_and_write(self, vectors: FloatMatrix, output_path: Path) -> None:
        """Build and write the vector index."""
        ...

    def read_vector_count(self, index_path: Path) -> int:
        """Return the persisted vector count."""
        ...


class EmbeddingIndexPipeline:
    """Build deterministic semantic documents, embeddings, FAISS, and metadata."""

    def __init__(
        self,
        *,
        catalog_repository: CatalogLoader,
        text_builder: AssessmentTextBuilder,
        embedding_service: EmbeddingService,
        index_builder: IndexWriter,
        index_path: Path,
        metadata_path: Path,
        manifest_path: Path,
        report_path: Path,
    ) -> None:
        self._catalog_repository = catalog_repository
        self._text_builder = text_builder
        self._embedding_service = embedding_service
        self._index_builder = index_builder
        self._index_path = index_path
        self._metadata_path = metadata_path
        self._manifest_path = manifest_path
        self._report_path = report_path

    async def build(self) -> tuple[EmbeddingManifest, EmbeddingReport]:
        """Run the full index pipeline and write all artifacts."""
        assessments, catalog_version, catalog_checksum = self._catalog_repository.load()
        documents = tuple(self._text_builder.build(assessment) for assessment in assessments)

        started = time.perf_counter()
        embedding_result = await self._embedding_service.embed_documents(documents)
        duration = time.perf_counter() - started

        vectors = embedding_result.vectors
        if vectors.shape[0] != len(assessments):
            raise IndexBuildError("every catalog record must have exactly one embedding")

        self._index_builder.build_and_write(vectors, self._index_path)
        persisted_count = self._index_builder.read_vector_count(self._index_path)
        if persisted_count != len(assessments):
            raise IndexBuildError("persisted FAISS index count does not match catalog count")

        records = tuple(
            build_metadata_record(
                vector_index=index,
                assessment=assessment,
                semantic_text=document.text,
            )
            for index, (assessment, document) in enumerate(zip(assessments, documents, strict=True))
        )
        self._validate_metadata(records, len(assessments))

        metadata = metadata_payload(
            catalog_version=catalog_version,
            catalog_checksum=catalog_checksum,
            embedding_model=self._embedding_service.model_name,
            embedding_dimension=self._embedding_service.dimension,
            records=records,
        )
        generation_timestamp = utc_now_iso()
        manifest = EmbeddingManifest(
            embedding_model=self._embedding_service.model_name,
            embedding_dimension=self._embedding_service.dimension,
            catalog_version=catalog_version,
            catalog_checksum=catalog_checksum,
            vector_count=len(assessments),
            generation_timestamp=generation_timestamp,
            index_checksum=file_checksum(self._index_path),
        )
        report = EmbeddingReport(
            vectors_created=len(assessments),
            failed_embeddings=embedding_result.failed_embeddings,
            skipped_embeddings=embedding_result.skipped_embeddings,
            embedding_duration_seconds=round(duration, 6),
            average_latency_seconds=round(duration / len(assessments), 6),
        )

        write_index_artifacts(
            metadata_path=self._metadata_path,
            manifest_path=self._manifest_path,
            report_path=self._report_path,
            metadata=metadata,
            manifest=manifest,
            report=report,
        )
        return manifest, report

    @staticmethod
    def _validate_metadata(records: tuple[MetadataRecord, ...], expected_count: int) -> None:
        if len(records) != expected_count:
            raise IndexBuildError("metadata count must equal vector count")
        for expected_index, record in enumerate(records):
            if record.vector_index != expected_index:
                raise IndexBuildError("metadata ordering must match FAISS vector ordering")
