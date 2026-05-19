"""Tests for chroma_dao — uses DummyEmbedder and tmp_path only."""
import pytest

from rageval.core.chunker import Chunk
from rageval.core.embedder import DummyEmbedder
from rageval.storage.chroma_dao import (
    RetrievedChunk,
    count,
    get_or_create_collection,
    query,
    upsert_chunks,
)

DIM = 16
_EMB = DummyEmbedder(dim=DIM)


def _chunk(chunk_id: str = "c1", doc_id: str = "d1", ordinal: int = 0, text: str = "sample") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, ordinal=ordinal, text=text, num_chars=len(text))


@pytest.fixture
def col(tmp_path):
    return get_or_create_collection(tmp_path / "chroma", "test_col")


# ---------------------------------------------------------------------------
# get_or_create_collection
# ---------------------------------------------------------------------------

class TestGetOrCreateCollection:
    def test_returns_a_collection(self, col):
        assert col is not None

    def test_collection_name(self, col):
        assert col.name == "test_col"

    def test_initially_empty(self, col):
        assert count(col) == 0


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------

class TestUpsertChunks:
    def test_single_chunk_count_one(self, col):
        c = _chunk()
        upsert_chunks(col, [c], _EMB.embed([c.text]))
        assert count(col) == 1

    def test_multiple_chunks(self, col):
        chunks = [_chunk("c1", text="foo"), _chunk("c2", text="bar"), _chunk("c3", text="baz")]
        upsert_chunks(col, chunks, _EMB.embed([c.text for c in chunks]))
        assert count(col) == 3

    def test_idempotent_same_chunks(self, col):
        c = _chunk()
        embs = _EMB.embed([c.text])
        upsert_chunks(col, [c], embs)
        upsert_chunks(col, [c], embs)
        assert count(col) == 1

    def test_empty_upsert_is_noop(self, col):
        upsert_chunks(col, [], [])
        assert count(col) == 0

    def test_extra_metadata_stored(self, col):
        c = _chunk()
        embs = _EMB.embed([c.text])
        upsert_chunks(col, [c], embs, extra_metadata=[{"custom_key": "custom_val"}])
        results = query(col, embs[0], top_k=1)
        assert results[0].metadata.get("custom_key") == "custom_val"


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

QUERY_TEXTS = ("alpha retrieval", "beta machine learning", "gamma database systems")


@pytest.fixture
def populated(col):
    chunks = [_chunk(f"c{i}", text=t) for i, t in enumerate(QUERY_TEXTS)]
    upsert_chunks(col, chunks, _EMB.embed([c.text for c in chunks]))
    return col


class TestQuery:
    def test_returns_list(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        assert isinstance(query(populated, q, top_k=3), list)

    def test_returns_retrieved_chunk_objects(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        results = query(populated, q, top_k=1)
        assert isinstance(results[0], RetrievedChunk)

    def test_top_k_limits_results(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        assert len(query(populated, q, top_k=2)) == 2

    def test_top_k_exceeds_count_returns_all(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        assert len(query(populated, q, top_k=100)) == 3

    def test_empty_collection_returns_empty(self, col):
        q = _EMB.embed(["anything"])[0]
        assert query(col, q, top_k=5) == []

    def test_result_chunk_id_present(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        r = query(populated, q, top_k=1)[0]
        assert r.chunk_id

    def test_result_doc_id_present(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        r = query(populated, q, top_k=1)[0]
        assert r.doc_id == "d1"

    def test_result_text_present(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        r = query(populated, q, top_k=1)[0]
        assert r.text

    def test_result_score_is_float(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        r = query(populated, q, top_k=1)[0]
        assert isinstance(r.score, float)

    def test_result_rank_starts_at_zero(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        assert query(populated, q, top_k=3)[0].rank == 0

    def test_result_ranks_are_sequential(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        results = query(populated, q, top_k=3)
        assert [r.rank for r in results] == [0, 1, 2]

    def test_exact_query_is_top_result(self, populated):
        # Querying with the stored embedding of "alpha retrieval" must rank it first
        exact_emb = _EMB.embed([QUERY_TEXTS[0]])[0]
        results = query(populated, exact_emb, top_k=3)
        assert results[0].text == QUERY_TEXTS[0]

    def test_metadata_contains_doc_id(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        r = query(populated, q, top_k=1)[0]
        assert "doc_id" in r.metadata

    def test_metadata_contains_ordinal(self, populated):
        q = _EMB.embed([QUERY_TEXTS[0]])[0]
        r = query(populated, q, top_k=1)[0]
        assert "ordinal" in r.metadata


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_empty(self, col):
        assert count(col) == 0

    def test_count_after_insert(self, col):
        chunks = [_chunk("c1", text="x"), _chunk("c2", text="y")]
        upsert_chunks(col, chunks, _EMB.embed([c.text for c in chunks]))
        assert count(col) == 2

    def test_count_idempotent_upsert(self, col):
        c = _chunk()
        embs = _EMB.embed([c.text])
        upsert_chunks(col, [c], embs)
        upsert_chunks(col, [c], embs)
        assert count(col) == 1
