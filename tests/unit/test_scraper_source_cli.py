"""Tests for the official catalog source and CLI composition."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shl_agent.scraper import cli
from shl_agent.scraper.artifacts import CatalogBuildReport, CatalogManifest
from shl_agent.scraper.official_catalog import OfficialCatalogSource


class FakeFetcher:
    """Return configured text without network I/O."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def fetch_text(self, _url: str) -> str:
        return self._text


@pytest.mark.asyncio
async def test_official_catalog_source_persists_raw_and_skips_non_objects(tmp_path: Path) -> None:
    raw = json.dumps([{"entity_id": "1"}, "bad-record"])
    source = OfficialCatalogSource(
        source_url="https://example.com/catalog.json",
        raw_output_path=tmp_path / "raw.json",
        fetcher=FakeFetcher(raw),
    )

    records = await source.load_records()

    assert records == ({"entity_id": "1"},)
    assert (tmp_path / "raw.json").read_text(encoding="utf-8") == raw


@pytest.mark.asyncio
async def test_official_catalog_source_rejects_non_list_json(tmp_path: Path) -> None:
    source = OfficialCatalogSource(
        source_url="https://example.com/catalog.json",
        raw_output_path=tmp_path / "raw.json",
        fetcher=FakeFetcher("{}"),
    )

    with pytest.raises(TypeError, match="JSON list"):
        await source.load_records()


class FakePipeline:
    """Fake pipeline used to verify CLI wiring without network calls."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    async def build(self) -> tuple[tuple[object, ...], CatalogBuildReport, CatalogManifest]:
        return (
            (),
            CatalogBuildReport("https://example.com/catalog.json", 1, (), 0, 0, 1),
            CatalogManifest("v1", 1, "2026-06-30T00:00:00Z", "sha256:abc"),
        )


@pytest.mark.asyncio
async def test_cli_build_catalog_prints_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: SimpleNamespace(
            log_level="INFO",
            scraper_timeout_seconds=1,
            scraper_retry_attempts=1,
            scraper_retry_backoff_seconds=0,
            scraper_rate_limit_seconds=0,
            catalog_source_url="https://example.com/catalog.json",
            raw_catalog_path=tmp_path / "raw.json",
            catalog_path=tmp_path / "catalog.json",
            catalog_manifest_path=tmp_path / "manifest.json",
            catalog_report_path=tmp_path / "report.json",
        ),
    )
    monkeypatch.setattr(cli, "CatalogPipeline", FakePipeline)
    monkeypatch.setattr(cli, "configure_logging", lambda _level: None)

    assert await cli.build_catalog() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["report"]["final_catalog_size"] == 1
    assert output["manifest"]["checksum"] == "sha256:abc"
