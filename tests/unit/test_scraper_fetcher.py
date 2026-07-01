"""Tests for bounded HTTP catalog fetching."""

from typing import ClassVar, Self

import httpx
import pytest

from shl_agent.scraper.fetcher import FetchConfig, FetchError, HttpCatalogFetcher


class FakeResponse:
    """Minimal response object used by HttpCatalogFetcher tests."""

    def __init__(self, text: str, *, should_raise: bool = False) -> None:
        self.text = text
        self._should_raise = should_raise

    def raise_for_status(self) -> None:
        if self._should_raise:
            request = httpx.Request("GET", "https://example.com/catalog.json")
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError("server error", request=request, response=response)


class FakeAsyncClient:
    """Async context manager that returns queued responses or errors."""

    calls = 0
    queue: ClassVar[list[FakeResponse | Exception]] = []

    def __init__(self, **_kwargs: object) -> None:
        self.closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.closed = True

    async def get(self, _url: str) -> FakeResponse:
        FakeAsyncClient.calls += 1
        item = FakeAsyncClient.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.mark.asyncio
async def test_fetcher_retries_then_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncClient.calls = 0
    FakeAsyncClient.queue = [httpx.TimeoutException("timeout"), FakeResponse("ok")]
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    fetcher = HttpCatalogFetcher(
        FetchConfig(retry_attempts=2, retry_backoff_seconds=0, rate_limit_seconds=0)
    )

    assert await fetcher.fetch_text("https://example.com/catalog.json") == "ok"
    assert FakeAsyncClient.calls == 2


@pytest.mark.asyncio
async def test_fetcher_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncClient.calls = 0
    FakeAsyncClient.queue = [FakeResponse("bad", should_raise=True)]
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    fetcher = HttpCatalogFetcher(
        FetchConfig(retry_attempts=1, retry_backoff_seconds=0, rate_limit_seconds=0)
    )

    with pytest.raises(FetchError, match="Failed to fetch"):
        await fetcher.fetch_text("https://example.com/catalog.json")
