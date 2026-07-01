"""Command-line entry point for building the offline SHL catalog artifact."""

import asyncio
import json
import logging
import sys
from dataclasses import asdict

from shl_agent.scraper.fetcher import FetchConfig, HttpCatalogFetcher
from shl_agent.scraper.official_catalog import OfficialCatalogSource
from shl_agent.scraper.parser import OfficialCatalogRecordParser
from shl_agent.scraper.pipeline import CatalogPipeline, CatalogUrlPolicy
from shl_agent.utils.logging import configure_logging
from shl_agent.utils.settings import get_settings

logger = logging.getLogger(__name__)


async def build_catalog() -> int:
    """Build the catalog from configured settings and print the report."""
    settings = get_settings()
    configure_logging(settings.log_level)

    fetcher = HttpCatalogFetcher(
        FetchConfig(
            timeout_seconds=settings.scraper_timeout_seconds,
            retry_attempts=settings.scraper_retry_attempts,
            retry_backoff_seconds=settings.scraper_retry_backoff_seconds,
            rate_limit_seconds=settings.scraper_rate_limit_seconds,
        )
    )
    source = OfficialCatalogSource(
        source_url=settings.catalog_source_url,
        raw_output_path=settings.raw_catalog_path,
        fetcher=fetcher,
    )
    pipeline = CatalogPipeline(
        source=source,
        record_parser=OfficialCatalogRecordParser(),
        url_policy=CatalogUrlPolicy(),
        catalog_path=settings.catalog_path,
        manifest_path=settings.catalog_manifest_path,
        report_path=settings.catalog_report_path,
    )

    _, report, manifest = await pipeline.build()
    sys.stdout.write(
        json.dumps(
            {"report": asdict(report), "manifest": asdict(manifest)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    logger.info("Catalog build command finished")
    return 0


def main() -> int:
    """Run the async catalog build command."""
    return asyncio.run(build_catalog())


if __name__ == "__main__":
    raise SystemExit(main())
