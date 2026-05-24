from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from rageval.ci.check import run_ci_check
from rageval.cli._console import console
from rageval.core.config import ThresholdsConfig
from rageval.storage.duckdb_dao import (
    get_connection,
    get_run_by_id,
    get_run_metric_means,
)


def ci_check(
    baseline: Annotated[str, typer.Option("--baseline", "-b", help="Baseline run ID")],
    candidate: Annotated[str, typer.Option("--candidate", "-c", help="Candidate run ID")],
    thresholds: Annotated[
        str, typer.Option("--thresholds", "-t", help="Thresholds YAML file")
    ] = "rageval.yaml",
    json_output: Annotated[
        bool, typer.Option("--json/--no-json", help="Also print JSON result to stdout")
    ] = False,
) -> None:
    """CI gate: exit 1 if any metric threshold is breached."""
    project_dir = Path.cwd()

    if not (project_dir / ".rageval").exists():
        console.print(
            "[red]Error:[/red] .rageval/ not found.\n"
            "  Run [bold]rageval init[/bold] first."
        )
        raise typer.Exit(1)

    thresholds_path = Path(thresholds)
    if not thresholds_path.is_absolute():
        thresholds_path = project_dir / thresholds_path

    if not thresholds_path.exists():
        console.print(f"[red]Error:[/red] Thresholds file not found: {thresholds!r}")
        raise typer.Exit(1)

    try:
        thresholds_cfg = ThresholdsConfig.from_yaml(thresholds_path)
    except Exception as exc:
        console.print(f"[red]Error:[/red] Could not parse thresholds file: {exc}")
        raise typer.Exit(1)

    db_path = project_dir / ".rageval" / "runs.db"
    con = get_connection(db_path)

    try:
        baseline_row = get_run_by_id(con, baseline)
        if baseline_row is None:
            console.print(f"[red]Error:[/red] Baseline run not found: {baseline!r}")
            raise typer.Exit(1)

        candidate_row = get_run_by_id(con, candidate)
        if candidate_row is None:
            console.print(f"[red]Error:[/red] Candidate run not found: {candidate!r}")
            raise typer.Exit(1)

        baseline_means = get_run_metric_means(con, baseline)
        candidate_means = get_run_metric_means(con, candidate)
    finally:
        con.close()

    result = run_ci_check(
        baseline_metric_means=baseline_means,
        candidate_metric_means=candidate_means,
        thresholds=thresholds_cfg,
        baseline_run_id=baseline,
        candidate_run_id=candidate,
        thresholds_path=str(thresholds_path),
    )

    # --- run info header ---
    console.print(
        f"\n[bold]Baseline:[/bold]  {baseline[:16]}...  tag=[bold]{baseline_row['tag']}[/bold]"
    )
    console.print(
        f"[bold]Candidate:[/bold] {candidate[:16]}...  tag=[bold]{candidate_row['tag']}[/bold]\n"
    )

    # --- violations table ---
    if result.violations:
        tbl = Table(title="Threshold Violations", show_header=True)
        tbl.add_column("Metric", style="bold")
        tbl.add_column("Check")
        tbl.add_column("Threshold", justify="right")
        tbl.add_column("Actual", justify="right")
        tbl.add_column("Message")
        for v in result.violations:
            tbl.add_row(
                v.metric,
                v.check_type,
                f"{v.threshold:.3f}",
                f"{v.actual:.3f}" if v.actual is not None else "N/A",
                v.message,
            )
        console.print(tbl)
    else:
        console.print("[green]No threshold violations.[/green]\n")

    # --- final verdict ---
    if result.passed:
        console.print("[bold green]CI RESULT: PASS[/bold green]")
    else:
        console.print("[bold red]CI RESULT: FAIL[/bold red]")

    # --- optional JSON ---
    if json_output:
        payload = {
            "passed": result.passed,
            "baseline_run_id": result.baseline_run_id,
            "candidate_run_id": result.candidate_run_id,
            "thresholds_path": result.thresholds_path,
            "violations": [
                {
                    "metric": v.metric,
                    "check_type": v.check_type,
                    "threshold": v.threshold,
                    "actual": v.actual,
                    "message": v.message,
                }
                for v in result.violations
            ],
        }
        typer.echo(json.dumps(payload, indent=2))

    if not result.passed:
        raise typer.Exit(1)
