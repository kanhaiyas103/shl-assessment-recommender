"""Offline SHL Individual Test catalog build pipeline."""

import logging
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from shl_agent.models.assessment import Assessment
from shl_agent.scraper.artifacts import (
    CatalogBuildReport,
    CatalogManifest,
    build_catalog_payload,
    checksum_payload,
    utc_now_iso,
    write_catalog_outputs,
)
from shl_agent.scraper.constants import CATALOG_VERSION, SHL_HOST
from shl_agent.scraper.official_catalog import OfficialCatalogSource
from shl_agent.scraper.parser import CatalogParseError, OfficialCatalogRecordParser

logger = logging.getLogger(__name__)


class CatalogUrlPolicy:
    """Validate that recommendations can only reference SHL catalog URLs."""

    def is_individual_test_url(self, url: str) -> bool:
        """Return true only for SHL product-catalog detail URLs."""
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != SHL_HOST:
            return False
        path = parsed.path.rstrip("/") + "/"
        return path.startswith(
            ("/products/product-catalog/view/", "/solutions/products/product-catalog/view/")
        )


class CatalogPipeline:
    """Build a validated, deduplicated, versioned catalog artifact."""

    def __init__(
        self,
        *,
        source: OfficialCatalogSource,
        record_parser: OfficialCatalogRecordParser,
        url_policy: CatalogUrlPolicy,
        catalog_path: Path,
        manifest_path: Path,
        report_path: Path,
        catalog_version: str = CATALOG_VERSION,
    ) -> None:
        self._source = source
        self._record_parser = record_parser
        self._url_policy = url_policy
        self._catalog_path = catalog_path
        self._manifest_path = manifest_path
        self._report_path = report_path
        self._catalog_version = catalog_version

    async def build(self) -> tuple[tuple[Assessment, ...], CatalogBuildReport, CatalogManifest]:
        """Run the full offline catalog pipeline and persist its outputs."""
        raw_records = await self._source.load_records()
        assessments: list[Assessment] = []
        skipped_pages: list[str] = []
        missing_field_count = 0

        for record in raw_records:
            record_id = str(record.get("entity_id") or record.get("link") or "unknown")
            raw_url = str(record.get("link") or "")
            if raw_url and not self._url_policy.is_individual_test_url(raw_url):
                skipped_pages.append(raw_url)
                logger.warning("Skipping non-individual catalog URL", extra={"url": raw_url})
                continue

            try:
                assessment = self._record_parser.parse_record(record)
            except (CatalogParseError, ValueError) as exc:
                missing_field_count += 1
                skipped_pages.append(record_id)
                logger.warning(
                    "Skipping invalid catalog record",
                    extra={"record_id": record_id, "error": str(exc)},
                )
                continue

            if not self._url_policy.is_individual_test_url(assessment.url):
                skipped_pages.append(assessment.url)
                logger.warning("Skipping non-individual catalog URL", extra={"url": assessment.url})
                continue

            assessments.append(assessment)

        deduplicated, duplicate_count = self._deduplicate(assessments)
        ordered = tuple(sorted(deduplicated, key=lambda item: item.name.casefold()))
        generated_at = utc_now_iso()
        catalog_payload = build_catalog_payload(
            catalog_version=self._catalog_version,
            generated_at=generated_at,
            source_url=self._source.source_url,
            assessments=ordered,
        )
        manifest = CatalogManifest(
            catalog_version=self._catalog_version,
            record_count=len(ordered),
            generation_timestamp=generated_at,
            checksum=checksum_payload(catalog_payload),
        )
        report = CatalogBuildReport(
            source_url=self._source.source_url,
            total_assessments_discovered=len(raw_records),
            skipped_pages=tuple(skipped_pages),
            duplicate_count=duplicate_count,
            missing_field_count=missing_field_count,
            final_catalog_size=len(ordered),
        )

        write_catalog_outputs(
            catalog_path=self._catalog_path,
            manifest_path=self._manifest_path,
            report_path=self._report_path,
            catalog_payload=catalog_payload,
            manifest=manifest,
            report=report,
        )
        logger.info(
            "Catalog pipeline complete",
            extra={
                "discovered": len(raw_records),
                "duplicates": duplicate_count,
                "skipped": len(skipped_pages),
                "final_catalog_size": len(ordered),
            },
        )
        return ordered, report, manifest

    @staticmethod
    def _deduplicate(assessments: Iterable[Assessment]) -> tuple[list[Assessment], int]:
        by_url: dict[str, Assessment] = {}
        duplicate_count = 0
        for assessment in assessments:
            if assessment.url in by_url:
                duplicate_count += 1
                logger.info("Dropping duplicate assessment", extra={"url": assessment.url})
                continue
            by_url[assessment.url] = assessment
        return list(by_url.values()), duplicate_count
