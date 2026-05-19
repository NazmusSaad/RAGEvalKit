from __future__ import annotations

from typing import Annotated

import typer

from rageval.cli._console import console


def inspect_run(
    run_id: Annotated[str, typer.Argument(help="Run ID or 'latest'")],
    top: Annotated[int, typer.Option("--top", "-n", help="Number of worst-failing queries to show")] = 5,
    by: Annotated[str, typer.Option("--by", help="Metric to sort failures by")] = "faithfulness",
) -> None:
    """Print the worst-failing queries with root-cause and unsupported claims."""
    console.print("[yellow]not yet implemented[/yellow]")
