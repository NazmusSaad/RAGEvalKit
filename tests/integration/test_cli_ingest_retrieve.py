"""Integration tests for `rageval ingest` and `rageval retrieve` CLI commands.

All storage paths are rooted at tmp_path.  No sentence-transformers model is
downloaded — the test config specifies embedding.provider: dummy.
"""
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.storage.chroma_dao import count as chroma_count
from rageval.storage.chroma_dao import get_or_create_collection
from rageval.storage.duckdb_dao import get_chunk_count, get_connection, get_document_count

runner = CliRunner()

TINY_CORPUS = Path(__file__).parents[2] / "examples" / "tiny-corpus"

# Config using the dummy embedder — no model downloads, no network calls.
_DUMMY_CONFIG = textwrap.dedent("""\
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
    evalset:
      path: evalsets/test.jsonl
""")


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Initialized project directory with a dummy-embedding config."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    (tmp_path / "configs" / "test.yaml").write_text(_DUMMY_CONFIG)
    return tmp_path


def _ingest(config: str = "configs/test.yaml") -> object:
    return runner.invoke(
        app,
        ["ingest", str(TINY_CORPUS), "--config", config],
        catch_exceptions=False,
    )


def _retrieve(query: str, top_k: int = 3, config: str = "configs/test.yaml") -> object:
    return runner.invoke(
        app,
        ["retrieve", query, "--top-k", str(top_k), "--config", config],
        catch_exceptions=False,
    )


# ---------------------------------------------------------------------------
# rageval ingest — basic
# ---------------------------------------------------------------------------

def test_ingest_exit_code_zero(project):
    assert _ingest().exit_code == 0


def test_ingest_output_mentions_documents(project):
    result = _ingest()
    assert "Documents" in result.output or "Ingest" in result.output


def test_ingest_creates_duckdb_documents(project):
    _ingest()
    con = get_connection(project / ".rageval" / "runs.db")
    try:
        assert get_document_count(con) == 3
    finally:
        con.close()


def test_ingest_creates_duckdb_chunks(project):
    _ingest()
    con = get_connection(project / ".rageval" / "runs.db")
    try:
        # default chunk_size=512; each tiny-corpus file is <512 chars → 1 chunk each
        assert get_chunk_count(con) == 3
    finally:
        con.close()


def test_ingest_populates_chroma(project):
    _ingest()
    col = get_or_create_collection(project / ".rageval" / "chroma", "test_col")
    assert chroma_count(col) == 3


# ---------------------------------------------------------------------------
# rageval ingest — idempotency
# ---------------------------------------------------------------------------

def test_ingest_twice_duckdb_count_unchanged(project):
    _ingest()
    _ingest()
    con = get_connection(project / ".rageval" / "runs.db")
    try:
        assert get_document_count(con) == 3
        assert get_chunk_count(con) == 3
    finally:
        con.close()


def test_ingest_twice_chroma_count_unchanged(project):
    _ingest()
    _ingest()
    col = get_or_create_collection(project / ".rageval" / "chroma", "test_col")
    assert chroma_count(col) == 3


# ---------------------------------------------------------------------------
# rageval ingest — error handling
# ---------------------------------------------------------------------------

def test_ingest_fails_without_init(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "test.yaml").write_text(_DUMMY_CONFIG)
    result = runner.invoke(app, ["ingest", str(TINY_CORPUS), "--config", "configs/test.yaml"])
    assert result.exit_code != 0
    assert "init" in result.output.lower()


def test_ingest_fails_with_missing_config(project):
    result = runner.invoke(app, ["ingest", str(TINY_CORPUS), "--config", "configs/missing.yaml"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "error" in result.output.lower()


def test_ingest_empty_dir_graceful(project, tmp_path):
    empty_dir = tmp_path / "empty_corpus"
    empty_dir.mkdir()
    result = runner.invoke(
        app,
        ["ingest", str(empty_dir), "--config", "configs/test.yaml"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "no documents" in result.output.lower()


# ---------------------------------------------------------------------------
# rageval retrieve — basic
# ---------------------------------------------------------------------------

def test_retrieve_after_ingest_exit_code(project):
    _ingest()
    result = _retrieve("RAG evaluation")
    assert result.exit_code == 0


def test_retrieve_after_ingest_has_results(project):
    _ingest()
    result = _retrieve("RAG evaluation")
    # output should mention results or contain chunk data
    assert "No results" not in result.output


def test_retrieve_top_k_limits_output(project):
    _ingest()
    result = _retrieve("RAG", top_k=2)
    assert result.exit_code == 0
    # "Top 2 results" or similar should appear
    assert "2" in result.output


# ---------------------------------------------------------------------------
# rageval retrieve — empty/missing collection
# ---------------------------------------------------------------------------

def test_retrieve_without_ingest_prints_no_results(project):
    result = _retrieve("anything")
    assert result.exit_code == 0
    assert "no results" in result.output.lower() or "ingest" in result.output.lower()


# ---------------------------------------------------------------------------
# rageval retrieve — error handling
# ---------------------------------------------------------------------------

def test_retrieve_fails_without_init(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "test.yaml").write_text(_DUMMY_CONFIG)
    result = runner.invoke(app, ["retrieve", "query", "--config", "configs/test.yaml"])
    assert result.exit_code != 0
    assert "init" in result.output.lower()


def test_retrieve_fails_with_missing_config(project):
    result = runner.invoke(app, ["retrieve", "query", "--config", "configs/missing.yaml"])
    assert result.exit_code != 0
