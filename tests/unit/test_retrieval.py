"""Tests for retrieve_top_k — uses DummyEmbedder and tmp_path only."""
import pytest

from rageval.core.chunker import Chunk
from rageval.core.embedder import DummyEmbedder
from rageval.core.retrieval import retrieve_top_k
from rageval.storage.chroma_dao import RetrievedChunk, get_or_create_collection, upsert_chunks

DIM = 16
_EMB = DummyEmbedder(dim=DIM)

_TEXTS = (
    "Python is a popular programming language",
    "Machine learning trains models on data",
    "DuckDB is an in-process analytical database",
)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d1", ordinal=0, text=text, num_chars=len(text))


@pytest.fixture
def chroma_path(tmp_path):
    """A tmp Chroma directory pre-populated with three chunks."""
    path = tmp_path / "chroma"
    col = get_or_create_collection(path, "docs")
    chunks = [_chunk(f"c{i}", t) for i, t in enumerate(_TEXTS)]
    upsert_chunks(col, chunks, _EMB.embed([c.text for c in chunks]))
    return path


class TestRetrieveTopK:
    def test_returns_list(self, chroma_path):
        results = retrieve_top_k("Python programming", chroma_path, "docs", _EMB, top_k=3)
        assert isinstance(results, list)

    def test_returns_retrieved_chunk_objects(self, chroma_path):
        results = retrieve_top_k("Python programming", chroma_path, "docs", _EMB, top_k=1)
        assert isinstance(results[0], RetrievedChunk)

    def test_top_k_limits_results(self, chroma_path):
        results = retrieve_top_k("query", chroma_path, "docs", _EMB, top_k=2)
        assert len(results) == 2

    def test_top_k_exceeds_count_returns_all(self, chroma_path):
        results = retrieve_top_k("query", chroma_path, "docs", _EMB, top_k=100)
        assert len(results) == 3

    def test_empty_collection_returns_empty(self, tmp_path):
        results = retrieve_top_k("query", tmp_path / "empty", "empty_col", _EMB, top_k=5)
        assert results == []

    def test_result_has_chunk_id(self, chroma_path):
        r = retrieve_top_k("query", chroma_path, "docs", _EMB, top_k=1)[0]
        assert r.chunk_id

    def test_result_has_doc_id(self, chroma_path):
        r = retrieve_top_k("query", chroma_path, "docs", _EMB, top_k=1)[0]
        assert r.doc_id

    def test_result_has_text(self, chroma_path):
        r = retrieve_top_k("query", chroma_path, "docs", _EMB, top_k=1)[0]
        assert r.text

    def test_result_has_score(self, chroma_path):
        r = retrieve_top_k("query", chroma_path, "docs", _EMB, top_k=1)[0]
        assert isinstance(r.score, float)

    def test_result_rank_starts_at_zero(self, chroma_path):
        r = retrieve_top_k("query", chroma_path, "docs", _EMB, top_k=1)[0]
        assert r.rank == 0

    def test_ranks_are_sequential(self, chroma_path):
        results = retrieve_top_k("query", chroma_path, "docs", _EMB, top_k=3)
        assert [r.rank for r in results] == [0, 1, 2]

    def test_exact_text_query_ranks_first(self, chroma_path):
        # query with the exact text of the first chunk; it must be ranked #1
        results = retrieve_top_k(_TEXTS[0], chroma_path, "docs", _EMB, top_k=3)
        assert results[0].text == _TEXTS[0]
