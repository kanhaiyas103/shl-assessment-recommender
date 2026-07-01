"""Protocols for catalog persistence and vector infrastructure."""

from collections.abc import Sequence
from typing import Protocol

from shl_agent.models.assessment import Assessment


class CatalogRepository(Protocol):
    """Read canonical assessments from immutable catalog artifacts."""

    async def get_by_id(self, assessment_id: str) -> Assessment | None:
        """Return one assessment by stable identifier."""
        ...

    async def get_by_name(self, name: str) -> Sequence[Assessment]:
        """Return canonical name matches for comparison resolution."""
        ...

    async def list_all(self) -> Sequence[Assessment]:
        """Return every Individual Test Solution."""
        ...


class EmbeddingProvider(Protocol):
    """Generate normalized vectors for catalog and query text."""

    @property
    def dimension(self) -> int:
        """Return the fixed output vector dimension."""
        ...

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed text in stable input order."""
        ...


class VectorIndex(Protocol):
    """Search a versioned vector index by normalized query vector."""

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
    ) -> Sequence[tuple[str, float]]:
        """Return stable assessment identifiers with similarity scores."""
        ...
