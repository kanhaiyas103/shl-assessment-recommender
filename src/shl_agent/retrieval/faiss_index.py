"""FAISS index construction and persistence."""

from pathlib import Path

import numpy as np

from shl_agent.retrieval.embedding_service import FloatMatrix


class FaissIndexError(ValueError):
    """Raised when FAISS index construction fails validation."""


class FaissIndexBuilder:
    """Build and persist a FAISS inner-product index over normalized vectors."""

    def build_and_write(self, vectors: FloatMatrix, output_path: Path) -> None:
        """Build an index whose vector IDs match metadata list positions."""
        if vectors.ndim != 2:
            raise FaissIndexError("vectors must be a 2D matrix")
        if vectors.shape[0] == 0:
            raise FaissIndexError("cannot build a FAISS index with zero vectors")

        contiguous = np.ascontiguousarray(vectors, dtype=np.float32)
        dimension = int(contiguous.shape[1])

        import faiss  # noqa: PLC0415

        index = faiss.IndexFlatIP(dimension)
        index.add(contiguous)
        if int(index.ntotal) != int(contiguous.shape[0]):
            raise FaissIndexError("FAISS vector count does not match input vectors")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path))

    def read_vector_count(self, index_path: Path) -> int:
        """Read a persisted FAISS index and return its vector count."""
        import faiss  # noqa: PLC0415

        index = faiss.read_index(str(index_path))
        return int(index.ntotal)
