"""Tests for embedding index CLI wiring."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shl_agent.retrieval import cli
from shl_agent.retrieval.index_artifacts import EmbeddingManifest, EmbeddingReport


class FakePipeline:
    """Fake index pipeline that avoids model and FAISS dependencies."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    async def build(self) -> tuple[EmbeddingManifest, EmbeddingReport]:
        return (
            EmbeddingManifest(
                "fake-model",
                3,
                "v1",
                "sha256:catalog",
                2,
                "2026-07-01T00:00:00Z",
                "sha256:index",
            ),
            EmbeddingReport(2, 0, 0, 0.5, 0.25),
        )


@pytest.mark.asyncio
async def test_build_index_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: SimpleNamespace(
            log_level="INFO",
            embedding_model="fake-model",
            embedding_batch_size=4,
            catalog_path=tmp_path / "catalog.json",
            catalog_manifest_path=tmp_path / "catalog_manifest.json",
            faiss_index_path=tmp_path / "index.faiss",
            index_metadata_path=tmp_path / "metadata.json",
            embedding_manifest_path=tmp_path / "embedding_manifest.json",
            embedding_report_path=tmp_path / "embedding_report.json",
        ),
    )
    monkeypatch.setattr(cli, "EmbeddingIndexPipeline", FakePipeline)
    monkeypatch.setattr(cli, "configure_logging", lambda _level: None)

    assert await cli.build_index() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["manifest"]["embedding_model"] == "fake-model"
    assert output["report"]["vectors_created"] == 2
