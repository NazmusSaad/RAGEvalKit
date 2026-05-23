from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from rageval.cli._console import console
from rageval.evaluators.summary import OPTIONAL_METRICS, REQUIRED_METRICS, build_run_summary, mean_metric
from rageval.storage.duckdb_dao import (
    get_connection,
    get_metric_scores_for_run,
    get_root_causes_for_run,
    get_run_by_id,
    get_run_items_basic,
    upsert_root_cause,
)

_ALL_CAUSES = [
    "none",
    "retrieval_failure",
    "grounding_failure",
    "answer_relevance_failure",
    "judge_uncertain",
    "missing_metric",
]


def summarize_run(
    run: Annotated[str, typer.Option("--run", "-r", help="Run ID to summarise")],
) -> None:
    """Aggregate metrics, classify root causes, and print a run summary."""
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

        items_data = get_run_items_basic(con, run)
        if not items_data:
            console.print("[yellow]No run items found for this run.[/yellow]")
            return

        scores_data = get_metric_scores_for_run(con, run)
        run_summary = build_run_summary(run, items_data, scores_data)

        for item_summary in run_summary.items:
            upsert_root_cause(
                con,
                item_id=item_summary.item_id,
                primary_cause=item_summary.primary_cause,
                secondary_causes=item_summary.secondary_causes,
                suggested_fix=item_summary.suggested_fix,
            )

    finally:
        con.close()

    # --- Run summary table ---
    n = len(run_summary.items)
    summary_table = Table(title="Run Summary", show_header=True)
    summary_table.add_column("Field", style="bold")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Run ID", run[:16] + "...")
    summary_table.add_row("Items summarized", str(n))
    summary_table.add_row("Pass", str(run_summary.pass_count))
    summary_table.add_row("Fail", str(run_summary.fail_count))
    summary_table.add_row("Unknown", str(run_summary.unknown_count))

    for metric_key, display_name in [
        ("recall_at_k", "Mean recall@k"),
        ("mrr", "Mean MRR"),
        ("answer_relevance", "Mean answer relevance"),
        ("faithfulness", "Mean faithfulness"),
    ]:
        summary_table.add_row(display_name, mean_metric(run_summary, metric_key))

    console.print(summary_table)

    # --- Root-cause distribution table ---
    cause_counts: dict[str, int] = {c: 0 for c in _ALL_CAUSES}
    for item in run_summary.items:
        cause = item.primary_cause
        cause_counts[cause] = cause_counts.get(cause, 0) + 1

    rc_table = Table(title="Root-Cause Distribution", show_header=True)
    rc_table.add_column("Primary cause", style="bold")
    rc_table.add_column("Count", justify="right")
    for cause in _ALL_CAUSES:
        rc_table.add_row(cause, str(cause_counts.get(cause, 0)))

    console.print(rc_table)
