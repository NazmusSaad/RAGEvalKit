"""Integration tests for `rageval evaluate-groundedness`.

Default-mock tests run without monkeypatching and verify the built-in
_MOCK_JUDGE_RESPONSE ("supported") produces faithfulness=1.0.
Specific-response tests patch _build_judge_client for precise assertions.
No real API calls.
"""
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.core.llm import MockLLMClient
from rageval.storage.duckdb_dao import (
    get_claims_for_item,
    get_connection,
    get_metric_scores_for_run,
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

_SUPPORTED_RESPONSE = json.dumps({
    "verdict": "supported",
    "supporting_indices": [0],
    "rationale": "Top context supports.",
})
_CONTRADICTED_RESPONSE = json.dumps({
    "verdict": "contradicted",
    "supporting_indices": [],
    "rationale": "Context contradicts.",
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
    import rageval.cli.evaluate_groundedness as eg_module
    mock = MockLLMClient(response_text=response_text)
    monkeypatch.setattr(eg_module, "_build_judge_client", lambda _config: mock)
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def run_project(tmp_path, monkeypatch):
    """Project with init + ingest + run (3 items, 2 retrieved contexts each)."""
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


@pytest.fixture
def project_with_claims(run_project):
    """Extends run_project: extract-claims runs (2 claims per item via default mock)."""
    project, run_id = run_project
    runner.invoke(
        app,
        ["extract-claims", "--run", run_id, "--config", "configs/test.yaml"],
        catch_exceptions=False,
    )
    return project, run_id


def _invoke_groundedness(run_id: str) -> object:
    return runner.invoke(
        app,
        ["evaluate-groundedness", "--run", run_id, "--config", "configs/test.yaml"],
        catch_exceptions=False,
    )


# ---------------------------------------------------------------------------
# Default mock behavior — no monkeypatching
# ---------------------------------------------------------------------------

class TestDefaultMockBehavior:
    """evaluate_groundedness._build_judge_client for 'mock' → supported → faithfulness=1.0."""

    def test_exit_code_zero(self, project_with_claims):
        _, run_id = project_with_claims
        assert _invoke_groundedness(run_id).exit_code == 0

    def test_faithfulness_metric_stored(self, project_with_claims):
        project, run_id = project_with_claims
        _invoke_groundedness(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        faithfulness = [s for s in scores if s["metric"] == "faithfulness"]
        assert len(faithfulness) == 3  # one per item

    def test_default_mock_faithfulness_is_1_0(self, project_with_claims):
        project, run_id = project_with_claims
        _invoke_groundedness(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        faithfulness = [s for s in scores if s["metric"] == "faithfulness"]
        assert all(abs(s["score"] - 1.0) < 1e-6 for s in faithfulness)

    def test_default_mock_labels_are_pass(self, project_with_claims):
        project, run_id = project_with_claims
        _invoke_groundedness(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        faithfulness = [s for s in scores if s["metric"] == "faithfulness"]
        assert all(s["label"] == "pass" for s in faithfulness)

    def test_output_shows_mean_1_000(self, project_with_claims):
        _, run_id = project_with_claims
        result = _invoke_groundedness(run_id)
        assert "1.000" in result.output

    def test_output_shows_judge_model(self, project_with_claims):
        _, run_id = project_with_claims
        result = _invoke_groundedness(run_id)
        assert "mock-judge" in result.output


# ---------------------------------------------------------------------------
# Verdict propagation — monkeypatched
# ---------------------------------------------------------------------------

class TestVerdictPropagation:
    def test_claim_verdicts_updated_from_unjudged(self, project_with_claims, monkeypatch):
        project, run_id = project_with_claims
        _patch_judge(monkeypatch, _SUPPORTED_RESPONSE)
        _invoke_groundedness(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            item_id = con.execute(
                "SELECT item_id FROM run_items LIMIT 1"
            ).fetchone()[0]
            claims = get_claims_for_item(con, item_id)
        finally:
            con.close()

        assert all(c["verdict"] == "supported" for c in claims)

    def test_supporting_chunk_ids_stored_as_json_list(self, project_with_claims, monkeypatch):
        project, run_id = project_with_claims
        _patch_judge(monkeypatch, _SUPPORTED_RESPONSE)
        _invoke_groundedness(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            rows = con.execute("SELECT supporting_chunk_ids FROM claim_evaluations").fetchall()
        finally:
            con.close()

        for (chunk_ids_json,) in rows:
            parsed = json.loads(chunk_ids_json)
            assert isinstance(parsed, list)

    def test_rationale_stored(self, project_with_claims, monkeypatch):
        project, run_id = project_with_claims
        _patch_judge(monkeypatch, _SUPPORTED_RESPONSE)
        _invoke_groundedness(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            rows = con.execute("SELECT rationale FROM claim_evaluations").fetchall()
        finally:
            con.close()

        assert all(row[0] is not None and len(row[0]) > 0 for row in rows)

    def test_contradicted_gives_faithfulness_0(self, project_with_claims, monkeypatch):
        project, run_id = project_with_claims
        _patch_judge(monkeypatch, _CONTRADICTED_RESPONSE)
        _invoke_groundedness(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()

        faithfulness = [s for s in scores if s["metric"] == "faithfulness"]
        assert all(abs(s["score"] - 0.0) < 1e-6 for s in faithfulness)
        assert all(s["label"] == "fail" for s in faithfulness)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_re_run_same_faithfulness_count(self, project_with_claims, monkeypatch):
        project, run_id = project_with_claims
        _patch_judge(monkeypatch, _SUPPORTED_RESPONSE)

        _invoke_groundedness(run_id)
        _invoke_groundedness(run_id)

        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()

        faithfulness = [s for s in scores if s["metric"] == "faithfulness"]
        assert len(faithfulness) == 3  # not 6


# ---------------------------------------------------------------------------
# Error: no claims extracted yet
# ---------------------------------------------------------------------------

class TestNoClaimsError:
    def test_fails_when_no_claims_extracted(self, run_project, monkeypatch):
        """evaluate-groundedness should reject runs with zero claims."""
        _, run_id = run_project
        # No extract-claims step → zero claims
        result = runner.invoke(
            app,
            ["evaluate-groundedness", "--run", run_id, "--config", "configs/test.yaml"],
        )
        assert result.exit_code != 0
        assert "extract-claims" in result.output


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestEvaluateGroundednessErrors:
    def test_fails_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate-groundedness", "--run", "fake-id"])
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    def test_fails_with_unknown_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"], catch_exceptions=False)
        (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
        result = runner.invoke(
            app,
            ["evaluate-groundedness", "--run", "no-such-run",
             "--config", "configs/test.yaml"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_fails_with_missing_config(self, project_with_claims, monkeypatch):
        _, run_id = project_with_claims
        result = runner.invoke(
            app,
            ["evaluate-groundedness", "--run", run_id,
             "--config", "configs/missing.yaml"],
        )
        assert result.exit_code != 0
