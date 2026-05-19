from __future__ import annotations

from typing import Annotated

import typer

from rageval.cli._console import console


def report(
    run: Annotated[str | None, typer.Option("--run", "-r", help="Run ID to report on")] = None,
    compare: Annotated[str | None, typer.Option("--compare", help="Two run IDs separated by a space")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output HTML file")] = None,
    open_browser: Annotated[bool, typer.Option("--open", help="Open report in browser after generating")] = False,
) -> None:
    """Generate a self-contained HTML evaluation report."""
    console.print("[yellow]not yet implemented[/yellow]")
