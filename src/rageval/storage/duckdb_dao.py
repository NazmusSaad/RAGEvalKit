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


# ---------------------------------------------------------------------------
# Metric score DAO
# ---------------------------------------------------------------------------

def get_run_items_with_questions(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
) -> list[dict]:
    """Return run items joined with eval_questions, including source_chunk_ids."""
    rows = con.execute(
        "SELECT ri.item_id, ri.question_id, eq.source_chunk_ids"
        " FROM run_items ri"
        " LEFT JOIN eval_questions eq ON ri.question_id = eq.question_id"
        " WHERE ri.run_id = ?",
        [run_id],
    ).fetchall()
    result = []
    for item_id, question_id, source_ids_json in rows:
        source_ids: list[str] = json.loads(source_ids_json) if source_ids_json else []
        result.append({
            "item_id": item_id,
            "question_id": question_id,
            "source_chunk_ids": source_ids,
        })
    return result


def get_retrieved_chunk_ids_for_item(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
) -> list[str]:
    """Return chunk_ids from retrieved_contexts for *item_id*, ordered by rank."""
    rows = con.execute(
        "SELECT chunk_id FROM retrieved_contexts WHERE item_id = ? ORDER BY rank",
        [item_id],
    ).fetchall()
    return [row[0] for row in rows]


def insert_metric_score(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    metric: str,
    score: float,
    label: str,
    reason: str,
    judge_model: str | None = None,
    raw_json: str | None = None,
) -> None:
    """Upsert a metric score row (delete-then-insert for idempotency)."""
    con.execute(
        "DELETE FROM metric_scores WHERE item_id = ? AND metric = ?",
        [item_id, metric],
    )
    con.execute(
        "INSERT INTO metric_scores (item_id, metric, score, label, reason, judge_model, raw_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [item_id, metric, score, label, reason, judge_model, raw_json],
    )


def get_metric_scores_for_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
) -> list[dict]:
    """Return all metric_scores rows for every item in *run_id*."""
    rows = con.execute(
        "SELECT ms.item_id, ms.metric, ms.score, ms.label, ms.reason"
        " FROM metric_scores ms"
        " JOIN run_items ri ON ms.item_id = ri.item_id"
        " WHERE ri.run_id = ?",
        [run_id],
    ).fetchall()
    keys = ["item_id", "metric", "score", "label", "reason"]
    return [dict(zip(keys, row)) for row in rows]


def get_run_items_for_evaluation(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
) -> list[dict]:
    """Return item_id, question text, and generated_answer for every item in *run_id*.

    Used by LLM-as-judge evaluators that need the full question and answer.
    """
    rows = con.execute(
        "SELECT ri.item_id, ri.question_id, eq.question, ri.generated_answer"
        " FROM run_items ri"
        " LEFT JOIN eval_questions eq ON ri.question_id = eq.question_id"
        " WHERE ri.run_id = ?",
        [run_id],
    ).fetchall()
    return [
        {
            "item_id": item_id,
            "question_id": question_id,
            "question": question or "",
            "generated_answer": generated_answer or "",
        }
        for item_id, question_id, question, generated_answer in rows
    ]


# ---------------------------------------------------------------------------
# Claim evaluations DAO
# ---------------------------------------------------------------------------

def delete_claims_for_item(con: duckdb.DuckDBPyConnection, item_id: str) -> None:
    """Delete all claim_evaluations rows for *item_id* (used before re-extraction)."""
    con.execute("DELETE FROM claim_evaluations WHERE item_id = ?", [item_id])


def insert_extracted_claims(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    claims: list,  # list[ExtractedClaim] — duck-typed to avoid cross-package import
) -> None:
    """Insert extracted claims as 'unjudged' rows; verdict/rationale filled later."""
    for claim in claims:
        con.execute(
            "INSERT INTO claim_evaluations"
            " (item_id, claim_idx, claim_text, verdict, supporting_chunk_ids, rationale)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [item_id, claim.claim_idx, claim.claim_text, "unjudged", "[]", None],
        )


def get_claims_for_item(con: duckdb.DuckDBPyConnection, item_id: str) -> list[dict]:
    """Return claim_evaluations rows for *item_id*, ordered by claim_idx."""
    rows = con.execute(
        "SELECT item_id, claim_idx, claim_text, verdict, supporting_chunk_ids, rationale"
        " FROM claim_evaluations WHERE item_id = ? ORDER BY claim_idx",
        [item_id],
    ).fetchall()
    keys = ["item_id", "claim_idx", "claim_text", "verdict", "supporting_chunk_ids", "rationale"]
    return [dict(zip(keys, row)) for row in rows]


def get_claim_count_for_run(con: duckdb.DuckDBPyConnection, run_id: str) -> int:
    """Count all claim_evaluations rows for every item in *run_id*."""
    return con.execute(
        "SELECT COUNT(*) FROM claim_evaluations ce"
        " JOIN run_items ri ON ce.item_id = ri.item_id"
        " WHERE ri.run_id = ?",
        [run_id],
    ).fetchone()[0]


def get_claims_for_run(con: duckdb.DuckDBPyConnection, run_id: str) -> list[dict]:
    """Return all claim_evaluations rows for every item in *run_id*, ordered by item then claim."""
    rows = con.execute(
        "SELECT ce.item_id, ce.claim_idx, ce.claim_text, ce.verdict,"
        "       ce.supporting_chunk_ids, ce.rationale"
        " FROM claim_evaluations ce"
        " JOIN run_items ri ON ce.item_id = ri.item_id"
        " WHERE ri.run_id = ?"
        " ORDER BY ce.item_id, ce.claim_idx",
        [run_id],
    ).fetchall()
    keys = ["item_id", "claim_idx", "claim_text", "verdict", "supporting_chunk_ids", "rationale"]
    return [dict(zip(keys, row)) for row in rows]


def update_claim_evaluation(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    claim_idx: int,
    verdict: str,
    supporting_chunk_ids: str,  # JSON string
    rationale: str | None,
) -> None:
    """Update verdict, supporting_chunk_ids, and rationale for one claim row."""
    con.execute(
        "UPDATE claim_evaluations"
        " SET verdict = ?, supporting_chunk_ids = ?, rationale = ?"
        " WHERE item_id = ? AND claim_idx = ?",
        [verdict, supporting_chunk_ids, rationale, item_id, claim_idx],
    )


def get_retrieved_contexts_for_item(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
) -> list[dict]:
    """Return full retrieved_contexts rows for *item_id*, ordered by rank."""
    rows = con.execute(
        "SELECT rank, chunk_id, chunk_text, score"
        " FROM retrieved_contexts WHERE item_id = ? ORDER BY rank",
        [item_id],
    ).fetchall()
    keys = ["rank", "chunk_id", "chunk_text", "score"]
    return [dict(zip(keys, row)) for row in rows]


# ---------------------------------------------------------------------------
# Root-cause DAO
# ---------------------------------------------------------------------------

def get_run_items_basic(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
) -> list[dict]:
    """Return item_id and question_id for every run item (lightweight query)."""
    rows = con.execute(
        "SELECT item_id, question_id FROM run_items WHERE run_id = ?",
        [run_id],
    ).fetchall()
    return [{"item_id": row[0], "question_id": row[1]} for row in rows]


def upsert_root_cause(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    primary_cause: str,
    secondary_causes: list,
    suggested_fix: str,
) -> None:
    """Upsert one root_causes row (delete-then-insert for idempotency)."""
    con.execute("DELETE FROM root_causes WHERE item_id = ?", [item_id])
    con.execute(
        "INSERT INTO root_causes (item_id, primary_cause, secondary_causes, suggested_fix)"
        " VALUES (?, ?, ?, ?)",
        [item_id, primary_cause, json.dumps(secondary_causes), suggested_fix],
    )


def get_root_causes_for_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
) -> list[dict]:
    """Return all root_causes rows for every item in *run_id*."""
    rows = con.execute(
        "SELECT rc.item_id, rc.primary_cause, rc.secondary_causes, rc.suggested_fix"
        " FROM root_causes rc"
        " JOIN run_items ri ON rc.item_id = ri.item_id"
        " WHERE ri.run_id = ?",
        [run_id],
    ).fetchall()
    return [
        {
            "item_id": row[0],
            "primary_cause": row[1],
            "secondary_causes": json.loads(row[2]) if row[2] else [],
            "suggested_fix": row[3],
        }
        for row in rows
    ]


def get_run_items_for_report(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
) -> list[dict]:
    """Return items with question, generated answer, and reference answer for *run_id*."""
    rows = con.execute(
        "SELECT ri.item_id, ri.question_id, eq.question, ri.generated_answer, eq.reference_answer"
        " FROM run_items ri"
        " LEFT JOIN eval_questions eq ON ri.question_id = eq.question_id"
        " WHERE ri.run_id = ?",
        [run_id],
    ).fetchall()
    return [
        {
            "item_id": row[0],
            "question_id": row[1],
            "question": row[2] or "",
            "generated_answer": row[3] or "",
            "reference_answer": row[4] or "",
        }
        for row in rows
    ]


def get_run_metric_means(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
) -> dict[str, float]:
    """Mean score per metric for *run_id*, excluding unknown-labelled rows."""
    rows = con.execute(
        "SELECT ms.metric, AVG(ms.score) AS mean_score"
        " FROM metric_scores ms"
        " JOIN run_items ri ON ms.item_id = ri.item_id"
        " WHERE ri.run_id = ? AND ms.label != 'unknown'"
        " GROUP BY ms.metric",
        [run_id],
    ).fetchall()
    return {row[0]: row[1] for row in rows if row[1] is not None}


def get_root_cause_distribution(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
) -> dict[str, int]:
    """Count of each primary_cause across all items in *run_id*."""
    rows = con.execute(
        "SELECT rc.primary_cause, COUNT(*) AS cnt"
        " FROM root_causes rc"
        " JOIN run_items ri ON rc.item_id = ri.item_id"
        " WHERE ri.run_id = ?"
        " GROUP BY rc.primary_cause",
        [run_id],
    ).fetchall()
    return {row[0]: row[1] for row in rows}
