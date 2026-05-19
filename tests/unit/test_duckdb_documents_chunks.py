"""Unit tests for document/chunk DAO functions in duckdb_dao.py."""
from pathlib import Path

import pytest

from rageval.core.chunker import Chunk
from rageval.core.loader import Document
from rageval.storage.duckdb_dao import (
    chunk_exists,
    document_exists,
    get_chunk_by_id,
    get_chunk_count,
    get_chunks_for_doc,
    get_connection,
    get_document_count,
    init_db,
    upsert_chunk,
    upsert_chunks,
    upsert_document,
    upsert_documents,
)


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    con = get_connection(db_path)
    yield con
    con.close()


def _doc(
    doc_id: str = "d1",
    text: str = "hello",
    source_path: str = "/fake/file.md",
) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=Path(source_path),
        text=text,
        num_chars=len(text),
    )


def _chunk(
    chunk_id: str = "c1",
    doc_id: str = "d1",
    ordinal: int = 0,
    text: str = "chunk text",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        ordinal=ordinal,
        text=text,
        num_chars=len(text),
    )


# ---------------------------------------------------------------------------
# document_exists / chunk_exists
# ---------------------------------------------------------------------------

class TestDocumentExists:
    def test_false_when_absent(self, db):
        assert document_exists(db, "nonexistent") is False

    def test_true_after_insert(self, db):
        upsert_document(db, _doc("d1"))
        assert document_exists(db, "d1") is True

    def test_does_not_match_different_id(self, db):
        upsert_document(db, _doc("d1"))
        assert document_exists(db, "d2") is False


class TestChunkExists:
    def test_false_when_absent(self, db):
        assert chunk_exists(db, "nonexistent") is False

    def test_true_after_insert(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk())
        assert chunk_exists(db, "c1") is True

    def test_does_not_match_different_id(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk("c1"))
        assert chunk_exists(db, "c2") is False


# ---------------------------------------------------------------------------
# upsert_document
# ---------------------------------------------------------------------------

class TestUpsertDocument:
    def test_returns_true_on_first_insert(self, db):
        assert upsert_document(db, _doc()) is True

    def test_creates_one_row(self, db):
        upsert_document(db, _doc())
        assert get_document_count(db) == 1

    def test_returns_false_on_second_insert(self, db):
        upsert_document(db, _doc())
        assert upsert_document(db, _doc()) is False

    def test_idempotent_row_count(self, db):
        upsert_document(db, _doc())
        upsert_document(db, _doc())
        assert get_document_count(db) == 1

    def test_different_ids_both_inserted(self, db):
        upsert_document(db, _doc("d1"))
        upsert_document(db, _doc("d2"))
        assert get_document_count(db) == 2

    def test_none_source_path_stored_as_empty_string(self, db):
        doc = Document(doc_id="dnull", source_path=None, text="x", num_chars=1)
        assert upsert_document(db, doc) is True
        assert get_document_count(db) == 1


# ---------------------------------------------------------------------------
# upsert_chunk
# ---------------------------------------------------------------------------

class TestUpsertChunk:
    def test_returns_true_on_first_insert(self, db):
        upsert_document(db, _doc())
        assert upsert_chunk(db, _chunk()) is True

    def test_creates_one_row(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk())
        assert get_chunk_count(db) == 1

    def test_returns_false_on_second_insert(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk())
        assert upsert_chunk(db, _chunk()) is False

    def test_idempotent_row_count(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk())
        upsert_chunk(db, _chunk())
        assert get_chunk_count(db) == 1

    def test_chunking_config_hash_stored(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk("c1"), chunking_config_hash="abc123")
        row = get_chunk_by_id(db, "c1")
        assert row["chunking_config_hash"] == "abc123"

    def test_none_config_hash_is_null(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk())
        row = get_chunk_by_id(db, "c1")
        assert row["chunking_config_hash"] is None

    def test_num_tokens_is_null(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk())
        row = get_chunk_by_id(db, "c1")
        assert row["num_tokens"] is None


# ---------------------------------------------------------------------------
# upsert_documents / upsert_chunks (batch)
# ---------------------------------------------------------------------------

class TestBatchUpsert:
    def test_upsert_documents_returns_inserted_count(self, db):
        docs = [_doc("d1"), _doc("d2"), _doc("d3")]
        assert upsert_documents(db, docs) == 3

    def test_upsert_documents_idempotent_returns_zero(self, db):
        docs = [_doc("d1"), _doc("d2")]
        upsert_documents(db, docs)
        assert upsert_documents(db, docs) == 0

    def test_upsert_documents_partial_new(self, db):
        upsert_document(db, _doc("d1"))
        assert upsert_documents(db, [_doc("d1"), _doc("d2")]) == 1

    def test_upsert_chunks_returns_inserted_count(self, db):
        upsert_document(db, _doc())
        chunks = [_chunk("c1", ordinal=0), _chunk("c2", ordinal=1), _chunk("c3", ordinal=2)]
        assert upsert_chunks(db, chunks) == 3

    def test_upsert_chunks_idempotent_returns_zero(self, db):
        upsert_document(db, _doc())
        chunks = [_chunk("c1", ordinal=0), _chunk("c2", ordinal=1)]
        upsert_chunks(db, chunks)
        assert upsert_chunks(db, chunks) == 0

    def test_upsert_chunks_partial_new(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk("c1"))
        chunks = [_chunk("c1"), _chunk("c2", ordinal=1)]
        assert upsert_chunks(db, chunks) == 1


# ---------------------------------------------------------------------------
# get_chunks_for_doc
# ---------------------------------------------------------------------------

class TestGetChunksForDoc:
    def test_returns_ordered_by_ordinal(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk("c3", ordinal=2, text="third"))
        upsert_chunk(db, _chunk("c1", ordinal=0, text="first"))
        upsert_chunk(db, _chunk("c2", ordinal=1, text="second"))
        rows = get_chunks_for_doc(db, "d1")
        assert [r["ordinal"] for r in rows] == [0, 1, 2]
        assert [r["text"] for r in rows] == ["first", "second", "third"]

    def test_empty_for_unknown_doc(self, db):
        assert get_chunks_for_doc(db, "unknown") == []

    def test_returns_only_target_doc_chunks(self, db):
        upsert_document(db, _doc("d1"))
        upsert_document(db, _doc("d2"))
        upsert_chunk(db, _chunk("c1", doc_id="d1"))
        upsert_chunk(db, _chunk("c2", doc_id="d2"))
        rows = get_chunks_for_doc(db, "d1")
        assert len(rows) == 1
        assert rows[0]["doc_id"] == "d1"

    def test_result_contains_expected_keys(self, db):
        upsert_document(db, _doc())
        upsert_chunk(db, _chunk())
        row = get_chunks_for_doc(db, "d1")[0]
        assert set(row.keys()) == {
            "chunk_id", "doc_id", "ordinal", "text", "num_tokens", "chunking_config_hash"
        }


# ---------------------------------------------------------------------------
# get_chunk_by_id
# ---------------------------------------------------------------------------

class TestGetChunkById:
    def test_returns_none_for_missing(self, db):
        assert get_chunk_by_id(db, "nonexistent") is None

    def test_returns_correct_fields(self, db):
        upsert_document(db, _doc("d1"))
        upsert_chunk(db, _chunk("c1", doc_id="d1", ordinal=7, text="specific"), "h123")
        row = get_chunk_by_id(db, "c1")
        assert row["chunk_id"] == "c1"
        assert row["doc_id"] == "d1"
        assert row["ordinal"] == 7
        assert row["text"] == "specific"
        assert row["chunking_config_hash"] == "h123"


# ---------------------------------------------------------------------------
# get_document_count / get_chunk_count
# ---------------------------------------------------------------------------

class TestCounts:
    def test_document_count_starts_zero(self, db):
        assert get_document_count(db) == 0

    def test_chunk_count_starts_zero(self, db):
        assert get_chunk_count(db) == 0

    def test_document_count_after_inserts(self, db):
        for i in range(5):
            upsert_document(db, _doc(f"d{i}"))
        assert get_document_count(db) == 5

    def test_chunk_count_after_inserts(self, db):
        upsert_document(db, _doc())
        for i in range(4):
            upsert_chunk(db, _chunk(f"c{i}", ordinal=i))
        assert get_chunk_count(db) == 4
