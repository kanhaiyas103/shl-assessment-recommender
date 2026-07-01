"""Protocols for offline SHL catalog discovery and parsing."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from shl_agent.models.assessment import Assessment


class CatalogPageDiscoverer(Protocol):
    """Discover Individual Test Solution detail-page URLs."""

    async def discover(self) -> AsyncIterator[str]:
        """Yield canonical catalog URLs without pre-packaged solutions."""
        ...


class AssessmentPageParser(Protocol):
    """Parse one fetched catalog detail page into a domain record."""

    def parse(self, *, url: str, html: str) -> Assessment:
        """Return a validated assessment from source HTML."""
        ...


class CatalogScraper(Protocol):
    """Coordinate offline catalog acquisition."""

    async def scrape(self) -> Sequence[Assessment]:
        """Return the complete deduplicated Individual Test catalog."""
        ...
