from __future__ import annotations

import json
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


def get_sample_chunks(con: duckdb.DuckDBPyConnection, n: int = 100) -> list[dict]:
    """Return up to *n* chunks ordered by doc_id and ordinal."""
    rows = con.execute(
        "SELECT chunk_id, doc_id, ordinal, text FROM chunks ORDER BY doc_id, ordinal LIMIT ?",
        [n],
    ).fetchall()
    keys = ["chunk_id", "doc_id", "ordinal", "text"]
    return [dict(zip(keys, row)) for row in rows]


# ---------------------------------------------------------------------------
# EvalSet DAO
# ---------------------------------------------------------------------------

from rageval.evalset.synthesize import EvalQuestion  # noqa: E402


def create_eval_set(
    con: duckdb.DuckDBPyConnection,
    evalset_id: str,
    name: str,
    generated_by: str,
    config_json: str,
) -> None:
    """Insert a new eval set row."""
    con.execute(
        "INSERT INTO eval_sets (evalset_id, name, generated_by, config_json)"
        " VALUES (?, ?, ?, ?)",
        [evalset_id, name, generated_by, config_json],
    )


def insert_eval_question(con: duckdb.DuckDBPyConnection, question: EvalQuestion) -> None:
    """Insert a single eval question row."""
    con.execute(
        "INSERT INTO eval_questions"
        " (question_id, evalset_id, question, reference_answer, source_chunk_ids, difficulty, qtype)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            question.question_id,
            question.evalset_id,
            question.question,
            question.reference_answer,
            json.dumps(question.source_chunk_ids),
            question.difficulty,
            question.qtype,
        ],
    )


def insert_eval_questions(
    con: duckdb.DuckDBPyConnection, questions: list[EvalQuestion]
) -> int:
    """Insert multiple eval questions. Returns count inserted."""
    for q in questions:
        insert_eval_question(con, q)
    return len(questions)


def get_eval_question_count(
    con: duckdb.DuckDBPyConnection,
    evalset_id: str | None = None,
) -> int:
    """Count eval questions, optionally filtered by *evalset_id*."""
    if evalset_id is not None:
        return con.execute(
            "SELECT COUNT(*) FROM eval_questions WHERE evalset_id = ?",
            [evalset_id],
        ).fetchone()[0]
    return con.execute("SELECT COUNT(*) FROM eval_questions").fetchone()[0]


# ---------------------------------------------------------------------------
# Run DAO
# ---------------------------------------------------------------------------

def create_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    name: str | None,
    tag: str,
    config_hash: str,
    config_json: str,
    evalset_id: str | None = None,
) -> None:
    """Insert a new run row with status='running' and started_at=now()."""
    con.execute(
        "INSERT INTO runs (run_id, name, tag, config_hash, config_json, evalset_id, started_at, status)"
        " VALUES (?, ?, ?, ?, ?, ?, now(), ?)",
        [run_id, name, tag, config_hash, config_json, evalset_id, "running"],
    )


def finish_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    status: str = "completed",
) -> None:
    """Set finished_at and status on an existing run row."""
    con.execute(
        "UPDATE runs SET finished_at = now(), status = ? WHERE run_id = ?",
        [status, run_id],
    )


def insert_run_item(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    run_id: str,
    question_id: str,
    generated_answer: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_cost_usd: float | None,
    latency_ms: int,
    model: str,
    error: str | None,
) -> None:
    """Insert one run_item row."""
    con.execute(
        "INSERT INTO run_items"
        " (item_id, run_id, question_id, generated_answer,"
        "  prompt_tokens, completion_tokens, total_cost_usd, latency_ms, model, error)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [item_id, run_id, question_id, generated_answer,
         prompt_tokens, completion_tokens, total_cost_usd, latency_ms, model, error],
    )


def insert_retrieved_contexts(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    contexts: list,  # list[RetrievedChunk] — duck-typed to avoid cross-module import
) -> None:
    """Insert one retrieved_contexts row per context (snapshot of chunk text + score)."""
    for ctx in contexts:
        con.execute(
            "INSERT INTO retrieved_contexts (item_id, rank, chunk_id, chunk_text, score)"
            " VALUES (?, ?, ?, ?, ?)",
            [item_id, ctx.rank, ctx.chunk_id, ctx.text, ctx.score],
        )


def get_run_by_id(con: duckdb.DuckDBPyConnection, run_id: str) -> dict | None:
    """Fetch one run row as a dict, or None if not found."""
    row = con.execute(
        "SELECT run_id, name, tag, config_hash, config_json, evalset_id,"
        "       git_sha, started_at, finished_at, status"
        " FROM runs WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if row is None:
        return None
    keys = ["run_id", "name", "tag", "config_hash", "config_json", "evalset_id",
            "git_sha", "started_at", "finished_at", "status"]
    return dict(zip(keys, row))


def get_run_item_count(con: duckdb.DuckDBPyConnection, run_id: str) -> int:
    """Count run_items belonging to *run_id*."""
    return con.execute(
        "SELECT COUNT(*) FROM run_items WHERE run_id = ?", [run_id]
    ).fetchone()[0]


def get_retrieved_context_count(con: duckdb.DuckDBPyConnection, run_id: str) -> int:
    """Count retrieved_contexts rows for all items in *run_id*."""
    return con.execute(
        "SELECT COUNT(*) FROM retrieved_contexts rc"
        " JOIN run_items ri ON rc.item_id = ri.item_id"
        " WHERE ri.run_id = ?",
        [run_id],
    ).fetchone()[0]
