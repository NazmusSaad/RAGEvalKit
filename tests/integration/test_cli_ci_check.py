"""Integration tests for `rageval ci-check`.

Strategy:
- PASS cases: thresholds set so low / None that mock data always satisfies them.
- FAIL (absolute): set faithfulness_min=1.01 (impossible to reach).
- FAIL (missing metric): run evaluators on baseline but NOT on candidate
  → candidate has no metric_scores → absolute threshold violation for missing metric.
- Error cases: missing file, unknown run ID, no .rageval/.
"""
import json
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.storage.duckdb_dao import get_connection

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
      path: evalsets/auto.jsonl
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


def _write_evalset(path: Path, questions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(q) for q in questions) + "\n")


def _all_run_ids(db_path: Path) -> list[str]:
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT run_id FROM runs ORDER BY started_at ASC"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def _invoke(*args) -> object:
    return runner.invoke(app, list(args), catch_exceptions=False)


def _invoke_unchecked(*args) -> object:
    return runner.invoke(app, list(args))


def _write_thresholds(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


# ---------------------------------------------------------------------------
# Shared fixture: project with two evaluated runs
# ---------------------------------------------------------------------------

@pytest.fixture
def two_evaluated_runs(tmp_path, monkeypatch):
    """Returns (tmp_path, baseline_id, candidate_id) — both fully evaluated."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)

    _invoke("init")
    _invoke("ingest", str(TINY_CORPUS), "--config", "configs/test.yaml")

    evalset_path = tmp_path / "evalsets" / "auto.jsonl"
    _write_evalset(evalset_path, _QUESTIONS)
    config_path = "configs/test.yaml"

    def _full_run() -> str:
        _invoke("run", "--config", config_path, "--evalset", str(evalset_path))
        run_id = _all_run_ids(tmp_path / ".rageval" / "runs.db")[-1]
        _invoke("evaluate-retrieval", "--run", run_id)
        _invoke("evaluate-answer-relevance", "--run", run_id, "--config", config_path)
        _invoke("extract-claims", "--run", run_id, "--config", config_path)
        _invoke("evaluate-groundedness", "--run", run_id, "--config", config_path)
        return run_id

    baseline_id = _full_run()
    candidate_id = _full_run()
    return tmp_path, baseline_id, candidate_id


# ---------------------------------------------------------------------------
# PASS: permissive thresholds
# ---------------------------------------------------------------------------

class TestCICheckPass:
    def test_exit_code_zero_no_thresholds(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        # Thresholds file with no checks configured → always passes
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1})
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", c)
        assert result.exit_code == 0

    def test_exit_code_zero_permissive_absolute(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {
            "version": 1,
            "absolute": {"faithfulness_min": 0.0, "answer_relevance_min": 0.0},
        })
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", c)
        assert result.exit_code == 0

    def test_output_contains_pass(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1})
        result = _invoke("ci-check", "--baseline", b, "--candidate", c)
        assert "PASS" in result.output

    def test_short_flags_work(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1})
        result = _invoke_unchecked("ci-check", "-b", b, "-c", c, "-t", str(tf))
        assert result.exit_code == 0

    def test_permissive_relative_passes(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {
            "version": 1,
            "relative": {"faithfulness_drop_max": 1.0, "answer_relevance_drop_max": 1.0},
        })
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", c)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# FAIL: impossible absolute threshold
# ---------------------------------------------------------------------------

class TestCICheckFail:
    def test_exit_code_one_on_absolute_violation(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        # faithfulness_min=1.01 is impossible (max score is 1.0)
        _write_thresholds(tf, {
            "version": 1,
            "absolute": {"faithfulness_min": 1.01},
        })
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", c)
        assert result.exit_code == 1

    def test_output_contains_fail(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1, "absolute": {"faithfulness_min": 1.01}})
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", c)
        assert "FAIL" in result.output

    def test_output_contains_violations_table(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1, "absolute": {"faithfulness_min": 1.01}})
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", c)
        assert "Violations" in result.output or "violation" in result.output.lower()


# ---------------------------------------------------------------------------
# FAIL: missing metric on candidate (no evaluators run on candidate)
# ---------------------------------------------------------------------------

class TestCICheckMissingMetric:
    @pytest.fixture
    def baseline_evaluated_candidate_bare(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)

        _invoke("init")
        _invoke("ingest", str(TINY_CORPUS), "--config", "configs/test.yaml")
        evalset_path = tmp_path / "evalsets" / "auto.jsonl"
        _write_evalset(evalset_path, _QUESTIONS)
        config_path = "configs/test.yaml"

        # Baseline: fully evaluated
        _invoke("run", "--config", config_path, "--evalset", str(evalset_path))
        baseline_id = _all_run_ids(tmp_path / ".rageval" / "runs.db")[-1]
        _invoke("evaluate-retrieval", "--run", baseline_id)
        _invoke("evaluate-answer-relevance", "--run", baseline_id, "--config", config_path)
        _invoke("extract-claims", "--run", baseline_id, "--config", config_path)
        _invoke("evaluate-groundedness", "--run", baseline_id, "--config", config_path)

        # Candidate: run only, no evaluators → no metric_scores
        _invoke("run", "--config", config_path, "--evalset", str(evalset_path))
        candidate_id = _all_run_ids(tmp_path / ".rageval" / "runs.db")[-1]

        return tmp_path, baseline_id, candidate_id

    def test_missing_metric_triggers_violation(self, baseline_evaluated_candidate_bare):
        tmp_path, b, c = baseline_evaluated_candidate_bare
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {
            "version": 1,
            "absolute": {"faithfulness_min": 0.50},
        })
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", c)
        assert result.exit_code == 1

    def test_missing_metric_violation_shows_na(self, baseline_evaluated_candidate_bare):
        tmp_path, b, c = baseline_evaluated_candidate_bare
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1, "absolute": {"faithfulness_min": 0.50}})
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", c)
        assert "N/A" in result.output or "missing" in result.output.lower()


# ---------------------------------------------------------------------------
# --json flag
# ---------------------------------------------------------------------------

class TestCICheckJSON:
    def test_json_flag_produces_valid_json(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1})
        result = _invoke("ci-check", "--baseline", b, "--candidate", c, "--json")
        # Isolate the JSON portion (last {...} block in output)
        lines = result.output.strip().splitlines()
        json_start = next(i for i, l in enumerate(lines) if l.strip() == "{")
        json_text = "\n".join(lines[json_start:])
        payload = json.loads(json_text)
        assert "passed" in payload
        assert "violations" in payload
        assert "baseline_run_id" in payload
        assert "candidate_run_id" in payload

    def test_json_passed_true_on_pass(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1})
        result = _invoke("ci-check", "--baseline", b, "--candidate", c, "--json")
        lines = result.output.strip().splitlines()
        json_start = next(i for i, l in enumerate(lines) if l.strip() == "{")
        payload = json.loads("\n".join(lines[json_start:]))
        assert payload["passed"] is True

    def test_json_passed_false_on_fail(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1, "absolute": {"faithfulness_min": 1.01}})
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", c, "--json")
        lines = result.output.strip().splitlines()
        json_start = next(i for i, l in enumerate(lines) if l.strip() == "{")
        payload = json.loads("\n".join(lines[json_start:]))
        assert payload["passed"] is False
        assert len(payload["violations"]) > 0

    def test_json_violation_has_required_fields(self, two_evaluated_runs):
        tmp_path, b, c = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1, "absolute": {"faithfulness_min": 1.01}})
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", c, "--json")
        lines = result.output.strip().splitlines()
        json_start = next(i for i, l in enumerate(lines) if l.strip() == "{")
        payload = json.loads("\n".join(lines[json_start:]))
        v = payload["violations"][0]
        assert "metric" in v
        assert "check_type" in v
        assert "threshold" in v
        assert "actual" in v
        assert "message" in v


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestCICheckErrors:
    def test_fails_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _invoke_unchecked("ci-check", "--baseline", "a", "--candidate", "b")
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    def test_fails_with_missing_thresholds_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _invoke("init")
        result = _invoke_unchecked(
            "ci-check", "--baseline", "a", "--candidate", "b",
            "--thresholds", "no-such-file.yaml",
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_fails_with_unknown_baseline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _invoke("init")
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1})
        result = _invoke_unchecked("ci-check", "--baseline", "no-such-run", "--candidate", "also-fake")
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "baseline" in result.output.lower()

    def test_fails_with_unknown_candidate(self, two_evaluated_runs):
        tmp_path, b, _ = two_evaluated_runs
        tf = tmp_path / "rageval.yaml"
        _write_thresholds(tf, {"version": 1})
        result = _invoke_unchecked("ci-check", "--baseline", b, "--candidate", "no-such-run")
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "candidate" in result.output.lower()

    def test_fails_with_malformed_thresholds_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _invoke("init")
        tf = tmp_path / "rageval.yaml"
        # extra="forbid" on ThresholdsConfig will reject unknown top-level fields
        tf.write_text("version: 1\nunknown_top_level_key: true\n")
        result = _invoke_unchecked("ci-check", "--baseline", "a", "--candidate", "b")
        assert result.exit_code != 0
