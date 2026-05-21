"""Integration tests for `rageval run`.

Uses DummyEmbedder (embedding.provider: dummy) and MockLLMClient
(generation.provider: mock).  No real API calls.  All storage under tmp_path.
"""
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.storage.duckdb_dao import (
    get_connection,
    get_retrieved_context_count,
    get_run_by_id,
    get_run_item_count,
)

runner = CliRunner()
TINY_CORPUS = Path(__file__).parents[2] / "examples" / "tiny-corpus"

_CONFIG = textwrap.dedent("""\
    version: 1
    name: test_config
    corpus:
      path: ./docs
    embedding:
      provider: dummy
      model: dummy
      batch_size: 16
    vector_store:
      path: .rageval/chroma
      collection: test_col
    generation:
      provider: mock
      model: mock-model
      temperature: 0.0
      max_tokens: 100
    judge:
      provider: mock
      model: mock
    evalset:
      path: evalsets/test.jsonl
""")

# A small JSONL evalset for use across tests.
_EVAL_QUESTIONS = [
    {
        "question_id": f"q{i}",
        "question": f"Question {i}?",
        "reference_answer": f"Answer {i}.",
        "source_chunk_ids": [],
        "difficulty": "easy",
        "qtype": "factoid",
    }
    for i in range(3)
]


def _write_evalset(path: Path, rows: list[dict] | None = None) -> None:
    rows = rows or _EVAL_QUESTIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def ingested_project(tmp_path, monkeypatch):
    """Project dir: init + ingest already complete."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
    runner.invoke(
        app,
        ["ingest", str(TINY_CORPUS), "--config", "configs/test.yaml"],
        catch_exceptions=False,
    )
    return tmp_path


@pytest.fixture
def run_project(ingested_project):
    """Extends ingested_project with a ready-to-use evalset JSONL."""
    _write_evalset(ingested_project / "evalsets" / "test.jsonl")
    return ingested_project


def _run(evalset: str = "evalsets/test.jsonl", tag: str = "test", limit: int | None = None) -> object:
    args = [
        "run",
        "--config", "configs/test.yaml",
        "--evalset", evalset,
        "--tag", tag,
    ]
    if limit is not None:
        args += ["--limit", str(limit)]
    return runner.invoke(app, args, catch_exceptions=False)


def _last_run_id(db_path: Path) -> str:
    con = get_connection(db_path)
    try:
        row = con.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    finally:
        con.close()
    return row[0] if row else ""


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestRunHappyPath:
    def test_exit_code_zero(self, run_project):
        assert _run().exit_code == 0

    def test_output_mentions_run_id(self, run_project):
        result = _run()
        assert "Run ID" in result.output or "run" in result.output.lower()

    def test_output_mentions_tag(self, run_project):
        result = _run(tag="baseline")
        assert "baseline" in result.output

    def test_creates_run_row(self, run_project):
        _run()
        db = run_project / ".rageval" / "runs.db"
        con = get_connection(db)
        try:
            row = con.execute("SELECT COUNT(*) FROM runs").fetchone()
        finally:
            con.close()
        assert row[0] >= 1

    def test_run_status_completed(self, run_project):
        _run(tag="check_status")
        db = run_project / ".rageval" / "runs.db"
        run_id = _last_run_id(db)
        con = get_connection(db)
        try:
            run = get_run_by_id(con, run_id)
        finally:
            con.close()
        assert run["status"] == "completed"

    def test_run_tag_stored(self, run_project):
        _run(tag="my-tag")
        db = run_project / ".rageval" / "runs.db"
        run_id = _last_run_id(db)
        con = get_connection(db)
        try:
            run = get_run_by_id(con, run_id)
        finally:
            con.close()
        assert run["tag"] == "my-tag"

    def test_creates_run_items(self, run_project):
        _run()
        db = run_project / ".rageval" / "runs.db"
        run_id = _last_run_id(db)
        con = get_connection(db)
        try:
            count = get_run_item_count(con, run_id)
        finally:
            con.close()
        assert count == 3  # 3 questions in the default evalset

    def test_creates_retrieved_contexts(self, run_project):
        _run()
        db = run_project / ".rageval" / "runs.db"
        run_id = _last_run_id(db)
        con = get_connection(db)
        try:
            count = get_retrieved_context_count(con, run_id)
        finally:
            con.close()
        assert count > 0  # at least 1 context per question

    def test_run_items_have_generated_answer(self, run_project):
        _run()
        db = run_project / ".rageval" / "runs.db"
        con = get_connection(db)
        try:
            rows = con.execute("SELECT generated_answer FROM run_items").fetchall()
        finally:
            con.close()
        assert all(row[0] for row in rows)  # non-empty answer for every item

    def test_run_items_model_stored(self, run_project):
        _run()
        db = run_project / ".rageval" / "runs.db"
        con = get_connection(db)
        try:
            models = {row[0] for row in con.execute("SELECT DISTINCT model FROM run_items").fetchall()}
        finally:
            con.close()
        assert "mock-model" in models


# ---------------------------------------------------------------------------
# --limit option
# ---------------------------------------------------------------------------

class TestRunLimit:
    def test_limit_reduces_run_items(self, run_project):
        _run(limit=2)
        db = run_project / ".rageval" / "runs.db"
        run_id = _last_run_id(db)
        con = get_connection(db)
        try:
            count = get_run_item_count(con, run_id)
        finally:
            con.close()
        assert count == 2

    def test_limit_one(self, run_project):
        _run(limit=1)
        db = run_project / ".rageval" / "runs.db"
        run_id = _last_run_id(db)
        con = get_connection(db)
        try:
            count = get_run_item_count(con, run_id)
        finally:
            con.close()
        assert count == 1

    def test_limit_larger_than_evalset(self, run_project):
        _run(limit=100)
        db = run_project / ".rageval" / "runs.db"
        run_id = _last_run_id(db)
        con = get_connection(db)
        try:
            count = get_run_item_count(con, run_id)
        finally:
            con.close()
        assert count == 3  # only 3 questions exist


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestRunErrors:
    def test_fails_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
        _write_evalset(tmp_path / "evalsets" / "test.jsonl")
        result = runner.invoke(
            app,
            ["run", "--config", "configs/test.yaml",
             "--evalset", "evalsets/test.jsonl"],
        )
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    def test_fails_with_missing_evalset(self, ingested_project):
        result = runner.invoke(
            app,
            ["run", "--config", "configs/test.yaml",
             "--evalset", "evalsets/missing.jsonl"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "evalset" in result.output.lower()

    def test_fails_before_ingest(self, tmp_path, monkeypatch):
        """No vectors in Chroma → clear error message."""
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"], catch_exceptions=False)
        (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
        _write_evalset(tmp_path / "evalsets" / "test.jsonl")
        result = runner.invoke(
            app,
            ["run", "--config", "configs/test.yaml",
             "--evalset", "evalsets/test.jsonl"],
        )
        assert result.exit_code != 0
        assert "ingest" in result.output.lower()

    def test_fails_with_missing_config(self, run_project):
        result = runner.invoke(
            app,
            ["run", "--config", "configs/missing.yaml",
             "--evalset", "evalsets/test.jsonl"],
        )
        assert result.exit_code != 0
