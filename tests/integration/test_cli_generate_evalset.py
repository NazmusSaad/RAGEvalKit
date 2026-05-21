"""Integration tests for `rageval generate-evalset`.

Uses MockLLMClient (judge.provider: mock) and DummyEmbedder.
No real API calls.  All storage under tmp_path.
"""
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.storage.duckdb_dao import get_connection, get_eval_question_count

runner = CliRunner()

TINY_CORPUS = Path(__file__).parents[2] / "examples" / "tiny-corpus"

# Config with both dummy embedder (ingest) and mock LLM (generate-evalset).
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
    judge:
      provider: mock
      model: mock
    evalset:
      path: evalsets/test.jsonl
""")


@pytest.fixture
def ingested_project(tmp_path, monkeypatch):
    """Project dir: init + ingest already run."""
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
def init_only_project(tmp_path, monkeypatch):
    """Project dir: init but no ingest."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
    return tmp_path


def _run_generate(num_questions: int = 3, output: str = "evalsets/test.jsonl") -> object:
    return runner.invoke(
        app,
        [
            "generate-evalset", ".",
            "--num-questions", str(num_questions),
            "--output", output,
            "--config", "configs/test.yaml",
        ],
        catch_exceptions=False,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestGenerateEvalsetHappyPath:
    def test_exit_code_zero(self, ingested_project):
        assert _run_generate().exit_code == 0

    def test_creates_jsonl_file(self, ingested_project):
        _run_generate()
        assert (ingested_project / "evalsets" / "test.jsonl").exists()

    def test_jsonl_has_at_least_one_line(self, ingested_project):
        _run_generate(num_questions=3)
        lines = [
            l for l in
            (ingested_project / "evalsets" / "test.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert len(lines) >= 1

    def test_jsonl_lines_are_valid_json(self, ingested_project):
        _run_generate(num_questions=3)
        for line in (ingested_project / "evalsets" / "test.jsonl").read_text().splitlines():
            if line.strip():
                json.loads(line)  # must not raise

    def test_jsonl_contains_required_fields(self, ingested_project):
        _run_generate(num_questions=3)
        first_line = next(
            l for l in
            (ingested_project / "evalsets" / "test.jsonl").read_text().splitlines()
            if l.strip()
        )
        data = json.loads(first_line)
        for field in ("question_id", "question", "reference_answer", "source_chunk_ids"):
            assert field in data, f"Missing field: {field}"

    def test_inserts_into_duckdb(self, ingested_project):
        _run_generate(num_questions=3)
        con = get_connection(ingested_project / ".rageval" / "runs.db")
        try:
            assert get_eval_question_count(con) >= 1
        finally:
            con.close()

    def test_output_mentions_model(self, ingested_project):
        result = _run_generate()
        assert "mock" in result.output.lower()

    def test_output_mentions_questions_generated(self, ingested_project):
        result = _run_generate()
        assert "Questions" in result.output or "questions" in result.output.lower()

    def test_custom_output_path(self, ingested_project):
        _run_generate(output="evalsets/custom.jsonl")
        assert (ingested_project / "evalsets" / "custom.jsonl").exists()

    def test_second_run_appends_more_questions(self, ingested_project):
        _run_generate(num_questions=3)
        _run_generate(num_questions=3)
        con = get_connection(ingested_project / ".rageval" / "runs.db")
        try:
            # Two separate evalsets, each with questions
            count = get_eval_question_count(con)
        finally:
            con.close()
        assert count >= 2  # at least one per run


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestGenerateEvalsetErrors:
    def test_fails_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
        result = runner.invoke(
            app, ["generate-evalset", ".", "--config", "configs/test.yaml"]
        )
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    def test_fails_before_ingest(self, init_only_project):
        result = runner.invoke(
            app, ["generate-evalset", ".", "--config", "configs/test.yaml"]
        )
        assert result.exit_code != 0
        assert "ingest" in result.output.lower()

    def test_fails_with_missing_config(self, ingested_project):
        result = runner.invoke(
            app,
            ["generate-evalset", ".", "--config", "configs/missing.yaml"],
        )
        assert result.exit_code != 0

    def test_error_message_mentions_ingest(self, init_only_project):
        result = runner.invoke(
            app, ["generate-evalset", ".", "--config", "configs/test.yaml"]
        )
        assert "ingest" in result.output.lower()
