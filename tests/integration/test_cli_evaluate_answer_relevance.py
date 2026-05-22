"""Integration tests for `rageval evaluate-answer-relevance`.

Happy-path tests use monkeypatch to swap create_llm_client for a MockLLMClient
configured with valid answer-relevance JSON, so scores and labels are meaningful.
Parse-failure tests use a MockLLMClient returning invalid JSON → label="unknown".
No real API calls are made.
"""
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.core.llm import MockLLMClient
from rageval.storage.duckdb_dao import get_connection, get_metric_scores_for_run

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
      model: mock-judge
    evalset:
      path: evalsets/test.jsonl
""")

_QUESTIONS = [
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

# Valid answer-relevance JSON that scores 3/4 = 0.75 → "pass"
_VALID_AR_RESPONSE = '{"score": 3, "reason": "The answer addresses the question."}'


def _write_evalset(path: Path, questions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(q) for q in questions) + "\n")


def _last_run_id(db_path: Path) -> str:
    con = get_connection(db_path)
    try:
        row = con.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else ""


def _patch_llm(monkeypatch, response_text: str):
    """Replace _build_judge_client in the CLI module with a fixed MockLLMClient."""
    import rageval.cli.evaluate_answer_relevance as ea_module

    mock = MockLLMClient(response_text=response_text)
    monkeypatch.setattr(ea_module, "_build_judge_client", lambda _config: mock)
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def run_project(tmp_path, monkeypatch):
    """Project with init + ingest + run (3 questions, empty source_chunk_ids)."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
    runner.invoke(
        app,
        ["ingest", str(TINY_CORPUS), "--config", "configs/test.yaml"],
        catch_exceptions=False,
    )
    _write_evalset(tmp_path / "evalsets" / "test.jsonl", _QUESTIONS)
    runner.invoke(
        app,
        ["run", "--config", "configs/test.yaml", "--evalset", "evalsets/test.jsonl"],
        catch_exceptions=False,
    )
    run_id = _last_run_id(tmp_path / ".rageval" / "runs.db")
    return tmp_path, run_id


def _invoke_eval(run_id: str) -> object:
    return runner.invoke(
        app,
        ["evaluate-answer-relevance", "--run", run_id, "--config", "configs/test.yaml"],
        catch_exceptions=False,
    )


# ---------------------------------------------------------------------------
# Happy path — valid answer-relevance responses (score=3 → 0.75 → pass)
# ---------------------------------------------------------------------------

class TestEvaluateAnswerRelevanceHappyPath:
    def test_exit_code_zero(self, run_project, monkeypatch):
        _, run_id = run_project
        _patch_llm(monkeypatch, _VALID_AR_RESPONSE)
        assert _invoke_eval(run_id).exit_code == 0

    def test_creates_one_metric_row_per_item(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_llm(monkeypatch, _VALID_AR_RESPONSE)
        _invoke_eval(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        ar = [s for s in scores if s["metric"] == "answer_relevance"]
        assert len(ar) == 3

    def test_scores_are_0_75(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_llm(monkeypatch, _VALID_AR_RESPONSE)
        _invoke_eval(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        ar = [s for s in scores if s["metric"] == "answer_relevance"]
        assert all(abs(s["score"] - 0.75) < 1e-6 for s in ar)

    def test_labels_are_pass(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_llm(monkeypatch, _VALID_AR_RESPONSE)
        _invoke_eval(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        ar = [s for s in scores if s["metric"] == "answer_relevance"]
        assert all(s["label"] == "pass" for s in ar)

    def test_reason_is_stored(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_llm(monkeypatch, _VALID_AR_RESPONSE)
        _invoke_eval(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        ar = [s for s in scores if s["metric"] == "answer_relevance"]
        assert all(s["reason"] == "The answer addresses the question." for s in ar)

    def test_output_shows_pass_count(self, run_project, monkeypatch):
        _, run_id = run_project
        _patch_llm(monkeypatch, _VALID_AR_RESPONSE)
        result = _invoke_eval(run_id)
        assert "3" in result.output  # 3 passes

    def test_output_shows_mean(self, run_project, monkeypatch):
        _, run_id = run_project
        _patch_llm(monkeypatch, _VALID_AR_RESPONSE)
        result = _invoke_eval(run_id)
        assert "0.750" in result.output

    def test_output_shows_judge_model(self, run_project, monkeypatch):
        _, run_id = run_project
        _patch_llm(monkeypatch, _VALID_AR_RESPONSE)
        result = _invoke_eval(run_id)
        assert "mock-judge" in result.output


# ---------------------------------------------------------------------------
# Parse failure path — invalid JSON → label="unknown"
# ---------------------------------------------------------------------------

class TestEvaluateAnswerRelevanceUnknown:
    def test_invalid_json_gives_unknown_labels(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_llm(monkeypatch, "this is not valid answer-relevance json")
        _invoke_eval(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        ar = [s for s in scores if s["metric"] == "answer_relevance"]
        assert len(ar) == 3
        assert all(s["label"] == "unknown" for s in ar)

    def test_output_shows_na_when_all_unknown(self, run_project, monkeypatch):
        _, run_id = run_project
        _patch_llm(monkeypatch, "bad json")
        result = _invoke_eval(run_id)
        assert "N/A" in result.output

    def test_fail_label_on_low_score(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_llm(monkeypatch, '{"score": 1, "reason": "Off-topic."}')
        _invoke_eval(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        ar = [s for s in scores if s["metric"] == "answer_relevance"]
        assert all(s["label"] == "fail" for s in ar)
        assert all(abs(s["score"] - 0.25) < 1e-6 for s in ar)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_re_run_same_row_count(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_llm(monkeypatch, _VALID_AR_RESPONSE)

        _invoke_eval(run_id)
        _invoke_eval(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        ar = [s for s in scores if s["metric"] == "answer_relevance"]
        assert len(ar) == 3  # not 6 after two runs


# ---------------------------------------------------------------------------
# Default mock behavior — no monkeypatching
# ---------------------------------------------------------------------------

class TestDefaultMockBehavior:
    """When judge.provider='mock' and _build_judge_client is not patched, the
    built-in _MOCK_JUDGE_RESPONSE should produce score=0.75 / label=pass — not unknown."""

    def test_default_mock_gives_pass_scores(self, run_project):
        project, run_id = run_project
        # deliberately no monkeypatching → _build_judge_client uses _MOCK_JUDGE_RESPONSE
        _invoke_eval(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()

        ar = [s for s in scores if s["metric"] == "answer_relevance"]
        assert len(ar) == 3
        assert all(abs(s["score"] - 0.75) < 1e-6 for s in ar)
        assert all(s["label"] == "pass" for s in ar)

    def test_default_mock_output_shows_mean_not_na(self, run_project):
        _, run_id = run_project
        result = _invoke_eval(run_id)
        assert "0.750" in result.output
        assert "N/A" not in result.output


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestEvaluateAnswerRelevanceErrors:
    def test_fails_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app, ["evaluate-answer-relevance", "--run", "fake-id"]
        )
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    def test_fails_with_unknown_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"], catch_exceptions=False)
        (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
        result = runner.invoke(
            app,
            ["evaluate-answer-relevance", "--run", "no-such-run",
             "--config", "configs/test.yaml"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_fails_with_missing_config(self, run_project, monkeypatch):
        _, run_id = run_project
        result = runner.invoke(
            app,
            ["evaluate-answer-relevance", "--run", run_id,
             "--config", "configs/missing.yaml"],
        )
        assert result.exit_code != 0
