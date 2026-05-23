import subprocess
import sys

from typer.testing import CliRunner

from rageval.cli.main import app

runner = CliRunner()

EXPECTED_COMMANDS = [
    "init",
    "ingest",
    "generate-evalset",
    "run",
    "compare",
    "report",
    "ci-check",
    "inspect",
    "retrieve",
    "evaluate-retrieval",
    "evaluate-answer-relevance",
    "extract-claims",
    "evaluate-groundedness",
]


def test_help_exit_code_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    for cmd in EXPECTED_COMMANDS:
        assert cmd in result.output, f"'{cmd}' missing from --help output"


def test_each_command_has_help():
    for cmd in EXPECTED_COMMANDS:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"'{cmd} --help' failed with exit code {result.exit_code}"


def test_importing_cli_main_does_not_load_chromadb():
    """chromadb must not be imported at module load time.

    Spawns a fresh interpreter so no prior test has already loaded chromadb,
    then verifies it is absent from sys.modules after importing the CLI app.
    """
    code = (
        "import sys; "
        "from rageval.cli.main import app; "
        "assert 'chromadb' not in sys.modules, "
        "'chromadb was imported eagerly: ' + str(sys.modules.get('chromadb'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"chromadb was loaded eagerly:\n{result.stderr}"
