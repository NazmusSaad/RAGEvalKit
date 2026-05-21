from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rageval.core.chunker import SimpleChunker
from rageval.core.config import PipelineConfig
from rageval.core.embedder import Embedder, create_embedder
from rageval.core.ids import sha256_text
from rageval.core.loader import load_documents
from rageval.storage.chroma_dao import get_or_create_collection
from rageval.storage.chroma_dao import upsert_chunks as _chroma_upsert
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


@dataclass
class FullIngestionResult:
    documents_loaded: int
    documents_inserted: int
    chunks_created: int
    chunks_inserted: int
    vectors_upserted: int
    chroma_collection: str
    elapsed_seconds: float


def ingest_corpus(
    path: Path,
    config: PipelineConfig,
    project_dir: Path,
    embedder: Embedder | None = None,
) -> FullIngestionResult:
    """Full ingestion pipeline: load → chunk → embed → DuckDB + Chroma.

    ``project_dir`` is the project root used to resolve ``.rageval/runs.db``
    and the relative ``vector_store.path`` from config.  The CLI passes
    ``Path.cwd()``; tests pass ``tmp_path``.

    Idempotent: DuckDB skips existing rows; Chroma upserts overwrite in place.
    """
    t0 = time.perf_counter()

    documents = load_documents(Path(path))
    chunker = SimpleChunker(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )
    all_chunks = [chunk for doc in documents for chunk in chunker.chunk_document(doc)]
    chunking_config_hash = sha256_text(
        f"{config.chunking.strategy}|{config.chunking.chunk_size}|{config.chunking.chunk_overlap}"
    )

    if embedder is None:
        embedder = create_embedder(config.embedding)

    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed(texts) if texts else []

    # --- DuckDB ---
    db_path = project_dir / ".rageval" / "runs.db"
    init_db(db_path)
    con = get_connection(db_path)
    try:
        docs_inserted = upsert_documents(con, documents)
        chunks_inserted = upsert_chunks(con, all_chunks, chunking_config_hash)
    finally:
        con.close()

    # --- Chroma ---
    chroma_path = project_dir / config.vector_store.path
    collection = get_or_create_collection(
        chroma_path,
        config.vector_store.collection,
        config.vector_store.distance,
    )
    _chroma_upsert(collection, all_chunks, embeddings)

    elapsed = time.perf_counter() - t0
    return FullIngestionResult(
        documents_loaded=len(documents),
        documents_inserted=docs_inserted,
        chunks_created=len(all_chunks),
        chunks_inserted=chunks_inserted,
        vectors_upserted=len(all_chunks),
        chroma_collection=config.vector_store.collection,
        elapsed_seconds=elapsed,
    )
