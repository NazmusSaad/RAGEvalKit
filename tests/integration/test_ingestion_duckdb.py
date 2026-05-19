"""Integration tests for ingest_documents_to_duckdb using the tiny-corpus fixture."""
from pathlib import Path

import pytest

from rageval.core.ingestion import IngestionResult, ingest_documents_to_duckdb
from rageval.storage.duckdb_dao import (
    get_chunk_count,
    get_chunks_for_doc,
    get_connection,
    get_document_count,
)

TINY_CORPUS = Path(__file__).parents[2] / "examples" / "tiny-corpus"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "runs.db"


# ---------------------------------------------------------------------------
# Basic ingestion
# ---------------------------------------------------------------------------

def test_ingest_returns_ingestion_result(db_path):
    result = ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    assert isinstance(result, IngestionResult)


def test_ingest_tiny_corpus_loads_three_docs(db_path):
    result = ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    assert result.documents_loaded == 3


def test_ingest_tiny_corpus_inserts_three_docs(db_path):
    result = ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    assert result.documents_inserted == 3


def test_ingest_tiny_corpus_creates_three_chunks(db_path):
    result = ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    # each tiny-corpus file is < 512 chars → exactly 1 chunk each
    assert result.chunks_created == 3


def test_ingest_tiny_corpus_inserts_three_chunks(db_path):
    result = ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    assert result.chunks_inserted == 3


def test_ingest_creates_db_file(db_path):
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    assert db_path.exists()


# ---------------------------------------------------------------------------
# DuckDB row counts
# ---------------------------------------------------------------------------

def test_duckdb_document_count_after_ingest(db_path):
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    con = get_connection(db_path)
    try:
        assert get_document_count(con) == 3
    finally:
        con.close()


def test_duckdb_chunk_count_after_ingest(db_path):
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    con = get_connection(db_path)
    try:
        assert get_chunk_count(con) == 3
    finally:
        con.close()


def test_chunks_have_chunking_config_hash(db_path):
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    con = get_connection(db_path)
    try:
        rows = con.execute("SELECT DISTINCT chunking_config_hash FROM chunks").fetchall()
    finally:
        con.close()
    hashes = [r[0] for r in rows]
    assert len(hashes) == 1
    assert hashes[0] is not None


def test_chunks_ordinals_start_at_zero(db_path):
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    con = get_connection(db_path)
    try:
        docs = con.execute("SELECT doc_id FROM documents").fetchall()
        for (doc_id,) in docs:
            chunks = get_chunks_for_doc(con, doc_id)
            assert chunks[0]["ordinal"] == 0
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_second_ingest_inserts_zero_docs(db_path):
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    result2 = ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    assert result2.documents_inserted == 0


def test_second_ingest_inserts_zero_chunks(db_path):
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    result2 = ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    assert result2.chunks_inserted == 0


def test_second_ingest_doc_count_unchanged(db_path):
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    con = get_connection(db_path)
    try:
        assert get_document_count(con) == 3
    finally:
        con.close()


def test_second_ingest_chunk_count_unchanged(db_path):
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    ingest_documents_to_duckdb(TINY_CORPUS, db_path)
    con = get_connection(db_path)
    try:
        assert get_chunk_count(con) == 3
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Custom chunk size
# ---------------------------------------------------------------------------

def test_smaller_chunk_size_produces_more_chunks(db_path):
    result = ingest_documents_to_duckdb(TINY_CORPUS, db_path, chunk_size=50, chunk_overlap=10)
    assert result.chunks_created > 3


def test_custom_chunk_size_rows_in_db(db_path):
    result = ingest_documents_to_duckdb(TINY_CORPUS, db_path, chunk_size=50, chunk_overlap=10)
    con = get_connection(db_path)
    try:
        assert get_chunk_count(con) == result.chunks_created
    finally:
        con.close()
