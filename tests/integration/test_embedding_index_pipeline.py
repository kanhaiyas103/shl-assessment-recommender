"""Integration tests for the embedding index build pipeline."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import TestType as AssessmentTestType
from shl_agent.retrieval.embedding_service import EmbeddingService, FloatMatrix
from shl_agent.retrieval.index_pipeline import EmbeddingIndexPipeline
from shl_agent.retrieval.text_builder import AssessmentTextBuilder


class FakeCatalogRepository:
    """Deterministic catalog loader for pipeline integration tests."""

    def load(self) -> tuple[tuple[Assessment, ...], str, str]:
        return (
            (
                Assessment(
                    assessment_id="python-new",
                    name="Python New",
                    url="https://www.shl.com/products/product-catalog/view/python-new/",
                    test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
                    description="Python skills.",
                ),
                Assessment(
                    assessment_id="java-new",
                    name="Java New",
                    url="https://www.shl.com/products/product-catalog/view/java-new/",
                    test_types=(AssessmentTestType.KNOWLEDGE_AND_SKILLS,),
                    description="Java skills.",
                ),
            ),
            "v1",
            "sha256:catalog",
        )


class FakeBackend:
    """Embeds text into deterministic 2D vectors."""

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimension(self) -> int:
        return 2

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        return np.asarray([[1.0, float(index)] for index, _ in enumerate(texts)], dtype=np.float32)


class FakeIndexBuilder:
    """Writes a small file and reports the latest vector count."""

    def __init__(self) -> None:
        self.vector_count = 0

    def build_and_write(self, vectors: FloatMatrix, output_path: Path) -> None:
        self.vector_count = int(vectors.shape[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-faiss-index")

    def read_vector_count(self, _index_path: Path) -> int:
        return self.vector_count


@pytest.mark.asyncio
async def test_embedding_index_pipeline_builds_consistent_artifacts(tmp_path: Path) -> None:
    pipeline = EmbeddingIndexPipeline(
        catalog_repository=FakeCatalogRepository(),
        text_builder=AssessmentTextBuilder(),
        embedding_service=EmbeddingService(FakeBackend()),
        index_builder=FakeIndexBuilder(),
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
        manifest_path=tmp_path / "embedding_manifest.json",
        report_path=tmp_path / "embedding_report.json",
    )

    manifest, report = await pipeline.build()
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert manifest.vector_count == 2
    assert report.vectors_created == 2
    assert metadata["vector_count"] == 2
    assert [record["vector_index"] for record in metadata["records"]] == [0, 1]
    assert [record["assessment_id"] for record in metadata["records"]] == ["python-new", "java-new"]
    assert (tmp_path / "index.faiss").read_bytes() == b"fake-faiss-index"


def test_text_builder_determinism_supports_rebuilds() -> None:
    repository = FakeCatalogRepository()
    assessments, _, _ = repository.load()
    builder = AssessmentTextBuilder()

    first = tuple(builder.build(assessment).text for assessment in assessments)
    second = tuple(builder.build(assessment).text for assessment in assessments)

    assert first == second
