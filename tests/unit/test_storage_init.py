import duckdb
import pytest
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.storage.duckdb_dao import init_db

runner = CliRunner()

EXPECTED_TABLES = {
    "documents",
    "chunks",
    "eval_sets",
    "eval_questions",
    "runs",
    "run_items",
    "retrieved_contexts",
    "metric_scores",
    "claim_evaluations",
    "root_causes",
}


def _table_names(db_path) -> set[str]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()


def test_init_db_directly_creates_all_tables(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    assert db_path.exists()
    assert EXPECTED_TABLES.issubset(_table_names(db_path))


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    init_db(db_path)  # second call must not raise
    assert EXPECTED_TABLES.issubset(_table_names(db_path))


def test_rageval_init_creates_db_with_all_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"], catch_exceptions=False)
    assert result.exit_code == 0
    db_path = tmp_path / ".rageval" / "runs.db"
    assert db_path.exists()
    assert EXPECTED_TABLES.issubset(_table_names(db_path))


def test_db_table_count_matches_schema(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    tables = _table_names(db_path)
    assert len(tables) == len(EXPECTED_TABLES)
