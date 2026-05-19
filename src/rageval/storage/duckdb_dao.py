from __future__ import annotations

import re
from pathlib import Path

import duckdb


def _split_statements(sql: str) -> list[str]:
    sql = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in sql.split(";") if s.strip()]


def init_db(path: Path) -> None:
    """Create the database file and initialize all tables from schema.sql."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = (Path(__file__).parent / "schema.sql").read_text()
    con = duckdb.connect(str(path))
    try:
        for stmt in _split_statements(schema_sql):
            con.execute(stmt)
    finally:
        con.close()


def get_connection(path: Path) -> duckdb.DuckDBPyConnection:
    """Return a read-write connection to the runs database."""
    return duckdb.connect(str(path))


# ---------------------------------------------------------------------------
# Document DAO
# ---------------------------------------------------------------------------

from rageval.core.loader import Document  # noqa: E402 — after duckdb import to avoid cycles
from rageval.core.chunker import Chunk  # noqa: E402


def document_exists(con: duckdb.DuckDBPyConnection, doc_id: str) -> bool:
    return con.execute("SELECT 1 FROM documents WHERE doc_id = ?", [doc_id]).fetchone() is not None


def upsert_document(con: duckdb.DuckDBPyConnection, document: Document) -> bool:
    """Insert document if not present. Returns True if inserted, False if already existed."""
    if document_exists(con, document.doc_id):
        return False
    source_path_str = str(document.source_path) if document.source_path is not None else ""
    title = document.source_path.stem if document.source_path is not None else ""
    con.execute(
        "INSERT INTO documents (doc_id, source_path, title, num_chars) VALUES (?, ?, ?, ?)",
        [document.doc_id, source_path_str, title, document.num_chars],
    )
    return True


def upsert_documents(con: duckdb.DuckDBPyConnection, documents: list[Document]) -> int:
    """Insert all new documents. Returns count of newly inserted rows."""
    return sum(1 for doc in documents if upsert_document(con, doc))


def get_document_count(con: duckdb.DuckDBPyConnection) -> int:
    return con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


# ---------------------------------------------------------------------------
# Chunk DAO
# ---------------------------------------------------------------------------

def chunk_exists(con: duckdb.DuckDBPyConnection, chunk_id: str) -> bool:
    return con.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", [chunk_id]).fetchone() is not None


def upsert_chunk(
    con: duckdb.DuckDBPyConnection,
    chunk: Chunk,
    chunking_config_hash: str | None = None,
) -> bool:
    """Insert chunk if not present. Returns True if inserted, False if already existed."""
    if chunk_exists(con, chunk.chunk_id):
        return False
    con.execute(
        "INSERT INTO chunks (chunk_id, doc_id, ordinal, text, num_tokens, chunking_config_hash)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [chunk.chunk_id, chunk.doc_id, chunk.ordinal, chunk.text, None, chunking_config_hash],
    )
    return True


def upsert_chunks(
    con: duckdb.DuckDBPyConnection,
    chunks: list[Chunk],
    chunking_config_hash: str | None = None,
) -> int:
    """Insert all new chunks. Returns count of newly inserted rows."""
    return sum(1 for chunk in chunks if upsert_chunk(con, chunk, chunking_config_hash))


def get_chunk_count(con: duckdb.DuckDBPyConnection) -> int:
    return con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def get_chunks_for_doc(con: duckdb.DuckDBPyConnection, doc_id: str) -> list[dict]:
    """Return all chunks for a document, ordered by ordinal."""
    rows = con.execute(
        "SELECT chunk_id, doc_id, ordinal, text, num_tokens, chunking_config_hash"
        " FROM chunks WHERE doc_id = ? ORDER BY ordinal",
        [doc_id],
    ).fetchall()
    keys = ["chunk_id", "doc_id", "ordinal", "text", "num_tokens", "chunking_config_hash"]
    return [dict(zip(keys, row)) for row in rows]


def get_chunk_by_id(con: duckdb.DuckDBPyConnection, chunk_id: str) -> dict | None:
    row = con.execute(
        "SELECT chunk_id, doc_id, ordinal, text, num_tokens, chunking_config_hash"
        " FROM chunks WHERE chunk_id = ?",
        [chunk_id],
    ).fetchone()
    if row is None:
        return None
    keys = ["chunk_id", "doc_id", "ordinal", "text", "num_tokens", "chunking_config_hash"]
    return dict(zip(keys, row))
