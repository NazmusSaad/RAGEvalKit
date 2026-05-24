from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rageval.cli._console import console
from rageval.report.render import build_run_report_data, render_run_report
from rageval.storage.duckdb_dao import get_connection, get_run_by_id


def report(
    run: Annotated[
        str | None, typer.Option("--run", "-r", help="Run ID to report on")
    ] = None,
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output HTML file path")
    ] = "report.html",
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open report in the default browser after generating")
    ] = False,
) -> None:
    """Generate a self-contained HTML evaluation report."""
    if run is None:
        console.print("[red]Error:[/red] --run <run_id> is required.")
        raise typer.Exit(1)

    project_dir = Path.cwd()

    if not (project_dir / ".rageval").exists():
        console.print(
            "[red]Error:[/red] .rageval/ not found.\n"
            "  Run [bold]rageval init[/bold] first."
        )
        raise typer.Exit(1)

    db_path = project_dir / ".rageval" / "runs.db"
    con = get_connection(db_path)

    try:
        run_row = get_run_by_id(con, run)
        if run_row is None:
            console.print(f"[red]Error:[/red] Run not found: {run!r}")
            raise typer.Exit(1)

        report_data = build_run_report_data(con, run)
    finally:
        con.close()

    html = render_run_report(report_data)

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    lc = report_data.label_counts
    console.print(f"[green]Report written to:[/green] {output_path}")
    console.print(
        f"  Items: {len(report_data.items)} | "
        f"Pass: {lc.get('pass', 0)} | "
        f"Fail: {lc.get('fail', 0)} | "
        f"Unknown: {lc.get('unknown', 0)}"
    )

    if open_browser:
        typer.launch(str(output_path))
