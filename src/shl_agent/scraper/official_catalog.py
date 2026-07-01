"""Assignment-provided SHL catalog source support."""

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class TextFetcher(Protocol):
    """Fetcher capability required by the official catalog source."""

    async def fetch_text(self, url: str) -> str:
        """Return text for a URL."""
        ...


class OfficialCatalogSource:
    """Load the PDF-linked SHL product catalog JSON feed."""

    def __init__(self, *, source_url: str, raw_output_path: Path, fetcher: TextFetcher) -> None:
        self.source_url = source_url
        self._raw_output_path = raw_output_path
        self._fetcher = fetcher

    async def load_records(self) -> Sequence[Mapping[str, object]]:
        """Fetch, persist, and parse source records."""
        raw_text = await self._fetcher.fetch_text(self.source_url)
        self._raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_output_path.write_text(raw_text, encoding="utf-8")

        parsed = json.loads(raw_text, strict=False)
        if not isinstance(parsed, list):
            raise TypeError("Official SHL catalog feed must be a JSON list")

        records: list[Mapping[str, object]] = []
        for item in parsed:
            if isinstance(item, dict):
                records.append(item)
            else:
                logger.warning("Skipping non-object catalog record", extra={"record": repr(item)})
        logger.info("Loaded official catalog records", extra={"record_count": len(records)})
        return tuple(records)
