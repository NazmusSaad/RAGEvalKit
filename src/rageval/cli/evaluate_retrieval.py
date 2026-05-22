from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from rageval.cli._console import console
from rageval.evaluators.retrieval_metrics import evaluate_retrieval_for_item
from rageval.storage.duckdb_dao import (
    get_connection,
    get_metric_scores_for_run,
    get_retrieved_chunk_ids_for_item,
    get_run_by_id,
    get_run_items_with_questions,
    insert_metric_score,
)


def evaluate_retrieval(
    run: Annotated[str, typer.Option("--run", "-r", help="Run ID to evaluate")],
    k: Annotated[int, typer.Option("--k", help="Top-k window for recall@k")] = 5,
) -> None:
    """Compute recall@k and MRR for a completed run and store results in DuckDB."""
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

        items = get_run_items_with_questions(con, run)
        if not items:
            console.print("[yellow]No run items found for this run.[/yellow]")
            return

        unknown_count = 0
        scored_recall: list[float] = []
        scored_mrr: list[float] = []

        for item in items:
            retrieved_ids = get_retrieved_chunk_ids_for_item(con, item["item_id"])
            results = evaluate_retrieval_for_item(
                source_chunk_ids=item["source_chunk_ids"],
                retrieved_chunk_ids=retrieved_ids,
                k=k,
            )
            for result in results:
                insert_metric_score(
                    con,
                    item_id=item["item_id"],
                    metric=result.metric,
                    score=result.score,
                    label=result.label,
                    reason=result.reason,
                )

            is_unknown = any(r.label == "unknown" for r in results)
            if is_unknown:
                unknown_count += 1
            else:
                for r in results:
                    if r.metric == "recall_at_k":
                        scored_recall.append(r.score)
                    elif r.metric == "mrr":
                        scored_mrr.append(r.score)

    finally:
        con.close()

    n_total = len(items)
    n_scored = len(scored_recall)

    if n_scored > 0:
        mean_recall_str = f"{sum(scored_recall) / n_scored:.3f}"
        mean_mrr_str = f"{sum(scored_mrr) / n_scored:.3f}"
    else:
        mean_recall_str = "N/A (all unknown)"
        mean_mrr_str = "N/A (all unknown)"

    table = Table(title="Retrieval Evaluation Summary", show_header=True)
    table.add_column("Field", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Run ID", run[:16] + "...")
    table.add_row("k", str(k))
    table.add_row("Items evaluated", str(n_total))
    table.add_row("Items with ground truth", str(n_scored))
    table.add_row("Unknown (no ground truth)", str(unknown_count))
    table.add_row(f"Mean recall@{k}", mean_recall_str)
    table.add_row("Mean MRR", mean_mrr_str)
    console.print(table)
