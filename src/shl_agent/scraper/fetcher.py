"""HTTP fetching primitives for polite offline catalog acquisition."""

import asyncio
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FetchConfig:
    """Network safety controls for catalog fetching."""

    timeout_seconds: float = 20.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 0.75
    rate_limit_seconds: float = 0.25
    user_agent: str = "shl-assessment-agent/0.1 catalog-pipeline"


class FetchError(RuntimeError):
    """Raised when a catalog resource cannot be fetched after retries."""


class HttpCatalogFetcher:
    """Small, injectable HTTP client with retries and polite rate limiting."""

    def __init__(self, config: FetchConfig) -> None:
        self._config = config

    async def fetch_text(self, url: str) -> str:
        """Fetch text content with bounded retries and timeout handling."""
        headers = {"User-Agent": self._config.user_agent, "Accept": "*/*"}
        last_error: Exception | None = None

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=self._config.timeout_seconds,
            headers=headers,
        ) as client:
            for attempt in range(1, self._config.retry_attempts + 1):
                if attempt > 1:
                    await asyncio.sleep(self._config.retry_backoff_seconds * (attempt - 1))

                try:
                    response = await client.get(url)
                    response.raise_for_status()
                except (httpx.HTTPError, httpx.TimeoutException) as exc:
                    last_error = exc
                    logger.warning(
                        "Catalog fetch attempt failed",
                        extra={"url": url, "attempt": attempt, "error": str(exc)},
                    )
                else:
                    await asyncio.sleep(self._config.rate_limit_seconds)
                    logger.debug("Fetched catalog resource", extra={"url": url, "attempt": attempt})
                    return response.text

        msg = (
            f"Failed to fetch catalog resource after {self._config.retry_attempts} attempts: {url}"
        )
        raise FetchError(msg) from last_error
