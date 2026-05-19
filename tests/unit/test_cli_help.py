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
