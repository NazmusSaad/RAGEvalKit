from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rageval.core.chunker import SimpleChunker
from rageval.core.ids import sha256_text
from rageval.core.loader import load_documents
from rageval.storage.duckdb_dao import (
    get_connection,
    init_db,
    upsert_chunks,
    upsert_documents,
)


@dataclass
class IngestionResult:
    documents_loaded: int
    documents_inserted: int
    chunks_created: int
    chunks_inserted: int


def ingest_documents_to_duckdb(
    path: Path,
    db_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> IngestionResult:
    """Load .md/.txt files, chunk them, and persist metadata to DuckDB.

    Idempotent: re-running on already-ingested files inserts 0 new rows.
    """
    documents = load_documents(Path(path))
    chunker = SimpleChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = [chunk for doc in documents for chunk in chunker.chunk_document(doc)]
    chunking_config_hash = sha256_text(f"recursive|{chunk_size}|{chunk_overlap}")

    init_db(db_path)
    con = get_connection(db_path)
    try:
        docs_inserted = upsert_documents(con, documents)
        chunks_inserted = upsert_chunks(con, all_chunks, chunking_config_hash)
    finally:
        con.close()

    return IngestionResult(
        documents_loaded=len(documents),
        documents_inserted=docs_inserted,
        chunks_created=len(all_chunks),
        chunks_inserted=chunks_inserted,
    )
