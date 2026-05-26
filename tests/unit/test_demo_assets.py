"""Tests that verify the demo corpus, config, scripts, and docs are present and valid."""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent


class TestDemoCorpus:
    def test_corpus_directory_exists(self):
        assert (ROOT / "examples" / "demo-corpus").is_dir()

    def test_all_three_corpus_files_exist(self):
        corpus = ROOT / "examples" / "demo-corpus"
        assert (corpus / "01_rag_evaluation.md").is_file()
        assert (corpus / "02_rag_failure_modes.md").is_file()
        assert (corpus / "03_ci_regression_testing.md").is_file()

    def test_corpus_files_are_nonempty(self):
        corpus = ROOT / "examples" / "demo-corpus"
        for f in corpus.glob("*.md"):
            assert f.stat().st_size > 200, f"{f.name} is suspiciously small"

    def test_corpus_files_have_h1_headings(self):
        corpus = ROOT / "examples" / "demo-corpus"
        for f in corpus.glob("*.md"):
            text = f.read_text(encoding="utf-8")
            assert text.startswith("# "), f"{f.name} missing H1 heading"

    def test_rag_evaluation_covers_key_metrics(self):
        text = (ROOT / "examples" / "demo-corpus" / "01_rag_evaluation.md").read_text(encoding="utf-8")
        for term in ("recall@k", "MRR", "faithfulness", "answer relevance"):
            assert term.lower() in text.lower(), f"Missing term: {term}"

    def test_failure_modes_covers_three_failure_types(self):
        text = (ROOT / "examples" / "demo-corpus" / "02_rag_failure_modes.md").read_text(encoding="utf-8")
        for term in ("retrieval failure", "grounding failure", "answer relevance failure"):
            assert term.lower() in text.lower(), f"Missing failure mode: {term}"

    def test_ci_testing_covers_thresholds_and_exit_codes(self):
        text = (ROOT / "examples" / "demo-corpus" / "03_ci_regression_testing.md").read_text(encoding="utf-8")
        for term in ("absolute", "relative", "exit code"):
            assert term.lower() in text.lower(), f"Missing term: {term}"


class TestDemoConfig:
    def _load_yaml(self):
        path = ROOT / "examples" / "configs" / "demo_openai.yaml"
        assert path.is_file(), "demo_openai.yaml not found"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_config_file_exists(self):
        assert (ROOT / "examples" / "configs" / "demo_openai.yaml").is_file()

    def test_config_parses_as_yaml(self):
        cfg = self._load_yaml()
        assert isinstance(cfg, dict)

    def test_config_has_required_top_level_keys(self):
        cfg = self._load_yaml()
        for key in ("version", "name", "embedding", "vector_store", "generation", "judge"):
            assert key in cfg, f"Missing key: {key}"

    def test_config_version_is_1(self):
        cfg = self._load_yaml()
        assert cfg["version"] == 1

    def test_config_uses_openai_provider(self):
        cfg = self._load_yaml()
        assert cfg["generation"]["provider"] == "openai"
        assert cfg["judge"]["provider"] == "openai"

    def test_config_uses_isolated_chroma_collection(self):
        cfg = self._load_yaml()
        collection = cfg["vector_store"]["collection"]
        assert collection == "demo_openai", f"Unexpected collection: {collection}"

    def test_config_uses_sentence_transformers_embedding(self):
        cfg = self._load_yaml()
        assert cfg["embedding"]["provider"] == "sentence_transformers"

    def test_config_no_api_key_hardcoded(self):
        raw = (ROOT / "examples" / "configs" / "demo_openai.yaml").read_text(encoding="utf-8")
        assert "sk-" not in raw, "API key appears to be hardcoded in demo config"

    def test_config_parses_with_pipelineconfig(self):
        from rageval.core.config import PipelineConfig

        cfg = self._load_yaml()
        parsed = PipelineConfig(**cfg)
        assert parsed.name == "demo_openai"
        assert parsed.generation.provider == "openai"
        assert parsed.judge.provider == "openai"


class TestDemoScripts:
    def test_bash_script_exists(self):
        assert (ROOT / "examples" / "demo_live_openai.sh").is_file()

    def test_powershell_script_exists(self):
        assert (ROOT / "examples" / "demo_live_openai.ps1").is_file()

    def test_bash_script_checks_api_key(self):
        text = (ROOT / "examples" / "demo_live_openai.sh").read_text(encoding="utf-8")
        assert "OPENAI_API_KEY" in text
        assert "ERROR" in text or "error" in text.lower()

    def test_powershell_script_checks_api_key(self):
        text = (ROOT / "examples" / "demo_live_openai.ps1").read_text(encoding="utf-8")
        assert "OPENAI_API_KEY" in text
        assert "ERROR" in text or "error" in text.lower()

    def test_bash_script_has_shebang(self):
        text = (ROOT / "examples" / "demo_live_openai.sh").read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")

    def test_bash_script_uses_set_euo_pipefail(self):
        text = (ROOT / "examples" / "demo_live_openai.sh").read_text(encoding="utf-8")
        assert "set -euo pipefail" in text

    def test_powershell_script_sets_error_action_stop(self):
        text = (ROOT / "examples" / "demo_live_openai.ps1").read_text(encoding="utf-8")
        assert "ErrorActionPreference" in text
        assert "Stop" in text

    def test_bash_script_captures_run_id(self):
        text = (ROOT / "examples" / "demo_live_openai.sh").read_text(encoding="utf-8")
        assert "RUN_ID" in text
        assert "runs.db" in text

    def test_powershell_script_captures_run_id(self):
        text = (ROOT / "examples" / "demo_live_openai.ps1").read_text(encoding="utf-8")
        assert "RUN_ID" in text
        assert "runs.db" in text

    def test_bash_script_runs_all_10_steps(self):
        text = (ROOT / "examples" / "demo_live_openai.sh").read_text(encoding="utf-8")
        for cmd in (
            "rageval init",
            "rageval ingest",
            "rageval generate-evalset",
            "rageval run",
            "rageval evaluate-retrieval",
            "rageval evaluate-answer-relevance",
            "rageval extract-claims",
            "rageval evaluate-groundedness",
            "rageval summarize-run",
            "rageval report",
        ):
            assert cmd in text, f"Missing command: {cmd}"

    def test_powershell_script_runs_all_10_steps(self):
        text = (ROOT / "examples" / "demo_live_openai.ps1").read_text(encoding="utf-8")
        for cmd in (
            "rageval init",
            "rageval ingest",
            "rageval generate-evalset",
            "rageval run",
            "rageval evaluate-retrieval",
            "rageval evaluate-answer-relevance",
            "rageval extract-claims",
            "rageval evaluate-groundedness",
            "rageval summarize-run",
            "rageval report",
        ):
            assert cmd in text, f"Missing command: {cmd}"

    def test_bash_script_no_hardcoded_api_key(self):
        text = (ROOT / "examples" / "demo_live_openai.sh").read_text(encoding="utf-8")
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", text), "API key appears hardcoded in bash script"

    def test_powershell_script_no_hardcoded_api_key(self):
        text = (ROOT / "examples" / "demo_live_openai.ps1").read_text(encoding="utf-8")
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", text), "API key appears hardcoded in PowerShell script"


class TestMockDemoScript:
    def test_mock_demo_script_exists(self):
        assert (ROOT / "examples" / "demo_commands.ps1").is_file()

    def test_mock_demo_script_has_no_api_key_requirement(self):
        text = (ROOT / "examples" / "demo_commands.ps1").read_text(encoding="utf-8")
        assert "OPENAI_API_KEY" not in text

    def test_mock_demo_script_runs_core_commands(self):
        text = (ROOT / "examples" / "demo_commands.ps1").read_text(encoding="utf-8")
        for cmd in (
            "rageval init",
            "rageval ingest",
            "rageval run",
            "rageval evaluate-retrieval",
            "rageval summarize-run",
            "rageval report",
        ):
            assert cmd in text, f"Missing command: {cmd}"

    def test_mock_demo_script_captures_run_id(self):
        text = (ROOT / "examples" / "demo_commands.ps1").read_text(encoding="utf-8")
        assert "RUN_ID" in text
        assert "runs.db" in text

    def test_mock_demo_script_sets_error_action_stop(self):
        text = (ROOT / "examples" / "demo_commands.ps1").read_text(encoding="utf-8")
        assert "ErrorActionPreference" in text
        assert "Stop" in text

    def test_mock_demo_script_no_hardcoded_api_key(self):
        text = (ROOT / "examples" / "demo_commands.ps1").read_text(encoding="utf-8")
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", text)


class TestDocPages:
    def test_architecture_doc_exists(self):
        assert (ROOT / "docs" / "architecture.md").is_file()

    def test_cli_doc_exists(self):
        assert (ROOT / "docs" / "cli.md").is_file()

    def test_metrics_doc_exists(self):
        assert (ROOT / "docs" / "metrics.md").is_file()

    def test_demo_script_doc_exists(self):
        assert (ROOT / "docs" / "demo_script.md").is_file()

    def test_architecture_doc_mentions_duckdb_and_chroma(self):
        text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
        assert "DuckDB" in text
        assert "Chroma" in text

    def test_cli_doc_covers_all_commands(self):
        text = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
        for cmd in ("rageval init", "rageval ingest", "rageval run", "rageval ci-check", "rageval report"):
            assert cmd in text, f"cli.md missing: {cmd}"

    def test_metrics_doc_covers_all_four_metrics(self):
        text = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
        for term in ("recall@k", "MRR", "answer_relevance", "faithfulness"):
            assert term in text, f"metrics.md missing: {term}"

    def test_readme_exists(self):
        assert (ROOT / "README.md").is_file()

    def test_readme_references_screenshots(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "report_summary_live_demo.png" in text
        assert "report_retrieval_failure_case.png" in text
        assert "report_success_case.png" in text

    def test_readme_screenshot_paths_exist(self):
        for name in (
            "report_summary_live_demo.png",
            "report_retrieval_failure_case.png",
            "report_success_case.png",
        ):
            assert (ROOT / "docs" / "assets" / name).is_file(), f"Missing: {name}"

    def test_readme_contains_mermaid_diagram(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "```mermaid" in text

    def test_readme_contains_project_summary(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "714 tests" in text or "tests" in text.lower()


class TestCIWorkflows:
    def test_tests_workflow_exists(self):
        assert (ROOT / ".github" / "workflows" / "tests.yml").is_file()

    def test_live_demo_workflow_exists(self):
        assert (ROOT / ".github" / "workflows" / "live-demo.yml").is_file()

    def test_tests_workflow_triggers_on_push_and_pr(self):
        text = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        assert "push" in text
        assert "pull_request" in text

    def test_live_demo_workflow_triggers_only_on_dispatch(self):
        import yaml as _yaml
        data = _yaml.safe_load(
            (ROOT / ".github" / "workflows" / "live-demo.yml").read_text(encoding="utf-8")
        )
        triggers = data.get("on", data.get(True, {}))
        assert "workflow_dispatch" in triggers, "live-demo.yml must use workflow_dispatch"
        assert "push" not in triggers, "live-demo.yml must not trigger on push"
        assert "pull_request" not in triggers, "live-demo.yml must not trigger on pull_request"

    def test_tests_workflow_does_not_set_openai_key(self):
        # The secret must not be injected as an env var in the default tests workflow.
        text = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        assert "secrets.OPENAI_API_KEY" not in text

    def test_live_demo_workflow_uses_secret_for_api_key(self):
        text = (ROOT / ".github" / "workflows" / "live-demo.yml").read_text(encoding="utf-8")
        assert "secrets.OPENAI_API_KEY" in text
        # must not hardcode a real key
        assert not __import__("re").search(r"sk-[A-Za-z0-9]{20,}", text)

    def test_tests_workflow_uses_ubuntu_and_python311(self):
        text = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        assert "ubuntu-latest" in text
        assert "3.11" in text

    def test_live_demo_workflow_uploads_artifact(self):
        text = (ROOT / ".github" / "workflows" / "live-demo.yml").read_text(encoding="utf-8")
        assert "upload-artifact" in text
        assert "live_demo_report.html" in text

    def test_tests_workflow_runs_pytest(self):
        text = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        assert "pytest" in text

    def test_tests_workflow_checks_cli_help(self):
        text = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        assert "rageval --help" in text


class TestLiveDemoDoc:
    def test_live_demo_doc_exists(self):
        assert (ROOT / "docs" / "live_demo.md").is_file()

    def test_live_demo_doc_covers_api_key_setup(self):
        text = (ROOT / "docs" / "live_demo.md").read_text(encoding="utf-8")
        assert "OPENAI_API_KEY" in text
        assert "export" in text or "$env:" in text

    def test_live_demo_doc_mentions_both_scripts(self):
        text = (ROOT / "docs" / "live_demo.md").read_text(encoding="utf-8")
        assert "demo_live_openai.sh" in text
        assert "demo_live_openai.ps1" in text

    def test_live_demo_doc_describes_cost(self):
        text = (ROOT / "docs" / "live_demo.md").read_text(encoding="utf-8")
        assert "cost" in text.lower() or "Cost" in text

    def test_live_demo_doc_mentions_report_output(self):
        text = (ROOT / "docs" / "live_demo.md").read_text(encoding="utf-8")
        assert "report" in text.lower()
        assert ".html" in text
