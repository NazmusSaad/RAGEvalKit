"""Integration tests for `rageval extract-claims`.

Default-mock tests run without monkeypatching and verify the built-in
_MOCK_JUDGE_RESPONSE produces 2 claims per item.  Specific-response tests
patch _build_judge_client for precise assertions.  No real API calls.
"""
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.core.llm import MockLLMClient
from rageval.storage.duckdb_dao import (
    get_claim_count_for_run,
    get_claims_for_item,
    get_connection,
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

_TWO_CLAIM_RESPONSE = json.dumps({
    "claims": ["First claim.", "Second claim."]
})


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


def _patch_judge(monkeypatch, response_text: str):
    import rageval.cli.extract_claims as ec_module
    mock = MockLLMClient(response_text=response_text)
    monkeypatch.setattr(ec_module, "_build_judge_client", lambda _config: mock)
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def run_project(tmp_path, monkeypatch):
    """Project with init + ingest + run (3 questions)."""
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


def _invoke_extract(run_id: str) -> object:
    return runner.invoke(
        app,
        ["extract-claims", "--run", run_id, "--config", "configs/test.yaml"],
        catch_exceptions=False,
    )


# ---------------------------------------------------------------------------
# Default mock behavior (no monkeypatch) — verifies dev mode
# ---------------------------------------------------------------------------

class TestDefaultMockBehavior:
    """_build_judge_client for provider='mock' returns 2-claim JSON."""

    def test_exit_code_zero(self, run_project):
        _, run_id = run_project
        assert _invoke_extract(run_id).exit_code == 0

    def test_default_mock_produces_two_claims_per_item(self, run_project):
        project, run_id = run_project
        _invoke_extract(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            count = get_claim_count_for_run(con, run_id)
        finally:
            con.close()
        assert count == 6  # 2 claims × 3 items

    def test_output_shows_total_claims(self, run_project):
        _, run_id = run_project
        result = _invoke_extract(run_id)
        assert "6" in result.output

    def test_output_shows_zero_unknowns(self, run_project):
        _, run_id = run_project
        result = _invoke_extract(run_id)
        assert "0" in result.output  # 0 unknown


# ---------------------------------------------------------------------------
# Specific response tests (monkeypatched)
# ---------------------------------------------------------------------------

class TestClaimExtractionCLI:
    def test_creates_correct_claim_count(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_judge(monkeypatch, _TWO_CLAIM_RESPONSE)
        _invoke_extract(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            count = get_claim_count_for_run(con, run_id)
        finally:
            con.close()
        assert count == 6  # 2 × 3

    def test_claim_texts_stored(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_judge(monkeypatch, _TWO_CLAIM_RESPONSE)
        _invoke_extract(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            item_id = con.execute(
                "SELECT item_id FROM run_items LIMIT 1"
            ).fetchone()[0]
            claims = get_claims_for_item(con, item_id)
        finally:
            con.close()
        assert len(claims) == 2
        assert claims[0]["claim_text"] == "First claim."
        assert claims[1]["claim_text"] == "Second claim."

    def test_claims_have_unjudged_verdict(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_judge(monkeypatch, _TWO_CLAIM_RESPONSE)
        _invoke_extract(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            rows = con.execute("SELECT verdict FROM claim_evaluations").fetchall()
        finally:
            con.close()
        assert all(row[0] == "unjudged" for row in rows)

    def test_claims_have_empty_supporting_chunk_ids(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_judge(monkeypatch, _TWO_CLAIM_RESPONSE)
        _invoke_extract(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            rows = con.execute("SELECT supporting_chunk_ids FROM claim_evaluations").fetchall()
        finally:
            con.close()
        for (chunk_ids,) in rows:
            parsed = json.loads(chunk_ids)
            assert parsed == []

    def test_output_mentions_judge_model(self, run_project, monkeypatch):
        _, run_id = run_project
        _patch_judge(monkeypatch, _TWO_CLAIM_RESPONSE)
        result = _invoke_extract(run_id)
        assert "mock-judge" in result.output


# ---------------------------------------------------------------------------
# Invalid JSON — unknown count increases, zero claims stored
# ---------------------------------------------------------------------------

class TestInvalidJudgeResponse:
    def test_invalid_json_stores_zero_claims(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_judge(monkeypatch, "definitely not json")
        _invoke_extract(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            count = get_claim_count_for_run(con, run_id)
        finally:
            con.close()
        assert count == 0

    def test_invalid_json_increments_unknown_count(self, run_project, monkeypatch):
        _, run_id = run_project
        _patch_judge(monkeypatch, "bad json")
        result = _invoke_extract(run_id)
        assert "3" in result.output  # 3 unknowns


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_re_run_same_claim_count(self, run_project, monkeypatch):
        project, run_id = run_project
        _patch_judge(monkeypatch, _TWO_CLAIM_RESPONSE)
        _invoke_extract(run_id)
        _invoke_extract(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            count = get_claim_count_for_run(con, run_id)
        finally:
            con.close()
        assert count == 6  # not 12 after two runs


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestExtractClaimsErrors:
    def test_fails_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["extract-claims", "--run", "fake-id"])
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    def test_fails_with_unknown_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"], catch_exceptions=False)
        (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
        result = runner.invoke(
            app,
            ["extract-claims", "--run", "no-such-run", "--config", "configs/test.yaml"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_fails_with_missing_config(self, run_project, monkeypatch):
        _, run_id = run_project
        result = runner.invoke(
            app,
            ["extract-claims", "--run", run_id, "--config", "configs/missing.yaml"],
        )
        assert result.exit_code != 0
