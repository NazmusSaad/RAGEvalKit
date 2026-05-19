from __future__ import annotations

from typing import Annotated

import typer

from rageval.cli._console import console


def ci_check(
    baseline: Annotated[str, typer.Option("--baseline", "-b", help="Baseline run ID or tag")],
    candidate: Annotated[str, typer.Option("--candidate", "-c", help="Candidate run ID or tag")],
    thresholds: Annotated[str, typer.Option("--thresholds", "-t", help="Thresholds YAML")] = "rageval.yaml",
) -> None:
    """CI gate: exit 1 if any metric threshold is breached."""
    console.print("[yellow]not yet implemented[/yellow]")
