from __future__ import annotations

from typing import Annotated

import typer

from rageval.cli._console import console


def compare(
    baseline_id: Annotated[str, typer.Argument(help="Baseline run ID or tag")],
    candidate_id: Annotated[str, typer.Argument(help="Candidate run ID or tag")],
    thresholds: Annotated[str, typer.Option("--thresholds", "-t", help="Thresholds YAML")] = "rageval.yaml",
) -> None:
    """Print a side-by-side metric delta between two runs."""
    console.print("[yellow]not yet implemented[/yellow]")
