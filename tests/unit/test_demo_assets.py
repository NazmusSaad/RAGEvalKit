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
