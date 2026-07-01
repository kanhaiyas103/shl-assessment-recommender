"""Tests for FAISS index adapter behavior using a fake faiss module."""

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from shl_agent.retrieval.faiss_index import FaissIndexBuilder, FaissIndexError


class FakeIndex:
    """Minimal FAISS-like index for adapter tests."""

    def __init__(self, dimension: int, *, mismatch: bool = False) -> None:
        self.dimension = dimension
        self._mismatch = mismatch
        self.ntotal = 0

    def add(self, vectors: np.ndarray) -> None:
        self.ntotal = int(vectors.shape[0]) + int(self._mismatch)


def install_fake_faiss(monkeypatch: pytest.MonkeyPatch, *, mismatch: bool = False) -> None:
    module = ModuleType("faiss")
    state: dict[str, FakeIndex] = {}

    def index_flat_ip(dimension: int) -> FakeIndex:
        return FakeIndex(dimension, mismatch=mismatch)

    def write_index(index: FakeIndex, path: str) -> None:
        state[path] = index
        Path(path).write_bytes(b"fake-index")

    def read_index(path: str) -> FakeIndex:
        return state[path]

    module.IndexFlatIP = index_flat_ip  # type: ignore[attr-defined]
    module.write_index = write_index  # type: ignore[attr-defined]
    module.read_index = read_index  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faiss", module)


def test_faiss_index_builder_writes_and_reads_vector_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_faiss(monkeypatch)
    builder = FaissIndexBuilder()
    index_path = tmp_path / "index.faiss"

    builder.build_and_write(np.ones((2, 3), dtype=np.float32), index_path)

    assert index_path.read_bytes() == b"fake-index"
    assert builder.read_vector_count(index_path) == 2


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((3,), dtype=np.float32),
        np.ones((0, 3), dtype=np.float32),
    ],
)
def test_faiss_index_builder_rejects_invalid_matrices(vectors: np.ndarray) -> None:
    with pytest.raises(FaissIndexError):
        FaissIndexBuilder().build_and_write(vectors, Path("unused.faiss"))


def test_faiss_index_builder_rejects_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_faiss(monkeypatch, mismatch=True)

    with pytest.raises(FaissIndexError, match="vector count"):
        FaissIndexBuilder().build_and_write(np.ones((2, 3), dtype=np.float32), tmp_path / "x")
