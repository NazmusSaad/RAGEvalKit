import pytest
from typer.testing import CliRunner

from rageval.cli.main import app

runner = CliRunner()

EXPECTED_FILES = [
    ".rageval/runs.db",
    "configs/baseline.yaml",
    "configs/experiment.yaml",
    "rageval.yaml",
    "prompts/system.txt",
    "prompts/rag.j2",
    ".gitignore",
]


def test_init_creates_all_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    for rel in EXPECTED_FILES:
        assert (tmp_path / rel).exists(), f"Missing: {rel}"


def test_init_creates_rageval_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    assert (tmp_path / ".rageval").is_dir()


def test_init_gitignore_contains_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    content = (tmp_path / ".gitignore").read_text()
    assert ".rageval/" in content


def test_init_appends_to_existing_gitignore(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")
    runner.invoke(app, ["init"], catch_exceptions=False)
    content = (tmp_path / ".gitignore").read_text()
    assert "*.pyc" in content
    assert ".rageval/" in content


def test_init_does_not_duplicate_gitignore_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    runner.invoke(app, ["init", "--force"], catch_exceptions=False)
    content = (tmp_path / ".gitignore").read_text()
    assert content.count(".rageval/") == 1


def test_init_fails_when_already_initialized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    result = runner.invoke(app, ["init"])
    assert result.exit_code != 0


def test_init_force_succeeds_when_already_initialized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    result = runner.invoke(app, ["init", "--force"], catch_exceptions=False)
    assert result.exit_code == 0


def test_baseline_yaml_has_correct_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    import yaml
    config = yaml.safe_load((tmp_path / "configs" / "baseline.yaml").read_text())
    assert config["name"] == "baseline"
    assert config["retrieval"]["top_k"] == 5


def test_experiment_yaml_has_smaller_top_k(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    import yaml
    config = yaml.safe_load((tmp_path / "configs" / "experiment.yaml").read_text())
    assert config["name"] == "experiment"
    assert config["retrieval"]["top_k"] == 3
