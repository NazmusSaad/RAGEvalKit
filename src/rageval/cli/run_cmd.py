from __future__ import annotations

from typing import Annotated

import typer

from rageval.cli._console import console


def run(
    config: Annotated[str, typer.Option("--config", "-c", help="Pipeline config YAML")],
    evalset: Annotated[str | None, typer.Option("--evalset", help="Eval set JSONL path (overrides config)")] = None,
    name: Annotated[str | None, typer.Option("--name", help="Human-readable run label")] = None,
    tag: Annotated[str, typer.Option("--tag", help="Run tag (baseline | candidate | ...)")] = "candidate",
) -> None:
    """Run the RAG pipeline against an eval set and store traces in DuckDB."""
    console.print("[yellow]not yet implemented[/yellow]")
