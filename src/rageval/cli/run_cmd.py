from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table

from rageval.cli._console import console
from rageval.core.config import PipelineConfig
from rageval.core.embedder import create_embedder
from rageval.core.ids import sha256_text
from rageval.core.llm import create_llm_client
from rageval.core.pipeline import RAGPipeline
from rageval.evalset.loader import load_evalset_from_jsonl
from rageval.storage.duckdb_dao import (
    create_run,
    finish_run,
    get_connection,
    insert_retrieved_contexts,
    insert_run_item,
)

if TYPE_CHECKING:
    import duckdb
    from rageval.evalset.synthesize import EvalQuestion


def _ensure_questions_registered(
    con: "duckdb.DuckDBPyConnection",
    questions: list["EvalQuestion"],
) -> str | None:
    """Ensure *questions* exist in DuckDB so the run_items FK is satisfied.

    If the first question is already in ``eval_questions`` we return its
    ``evalset_id`` unchanged.  Otherwise we create a new ``eval_set`` row and
    insert all questions.  This covers both questions from ``generate-evalset``
    (already in DuckDB) and externally-supplied JSONL files (not yet in DuckDB).
    """
    if not questions:
        return None

    row = con.execute(
        "SELECT evalset_id FROM eval_questions WHERE question_id = ?",
        [questions[0].question_id],
    ).fetchone()
    if row:
        return row[0] or None

    # Questions not yet registered — create an eval_set and insert them.
    evalset_id = uuid.uuid4().hex
    con.execute(
        "INSERT INTO eval_sets (evalset_id, name, generated_by) VALUES (?, ?, ?)",
        [evalset_id, "run-import", "jsonl_import"],
    )
    for q in questions:
        con.execute(
            "INSERT INTO eval_questions"
            " (question_id, evalset_id, question, reference_answer,"
            "  source_chunk_ids, difficulty, qtype)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                q.question_id, evalset_id, q.question, q.reference_answer,
                json.dumps(q.source_chunk_ids), q.difficulty, q.qtype,
            ],
        )
    return evalset_id


def run(
    config: Annotated[str, typer.Option("--config", "-c", help="Pipeline config YAML")],
    evalset: Annotated[str | None, typer.Option("--evalset", help="Eval set JSONL path (overrides config)")] = None,
    name: Annotated[str | None, typer.Option("--name", help="Human-readable run label")] = None,
    tag: Annotated[str, typer.Option("--tag", help="Run tag (baseline | candidate | ...)")] = "candidate",
    limit: Annotated[int | None, typer.Option("--limit", "-l", help="Process only the first N questions")] = None,
) -> None:
    """Run the RAG pipeline against an eval set and store full traces in DuckDB."""
    project_dir = Path.cwd()

    # --- guard: init ---
    if not (project_dir / ".rageval").exists():
        console.print(
            "[red]Error:[/red] .rageval/ not found.\n"
            "  Run [bold]rageval init[/bold] first."
        )
        raise typer.Exit(1)

    # --- load config ---
    config_path = project_dir / config
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_path}")
        raise typer.Exit(1)

    try:
        pipeline_config = PipelineConfig.from_yaml(config_path)
    except Exception as exc:
        console.print(f"[red]Error loading config:[/red] {exc}")
        raise typer.Exit(1)

    # --- resolve evalset path ---
    if evalset:
        evalset_path = Path(evalset)
        if not evalset_path.is_absolute():
            evalset_path = project_dir / evalset_path
    else:
        evalset_path = project_dir / pipeline_config.evalset.path

    try:
        questions = load_evalset_from_jsonl(evalset_path)
    except FileNotFoundError:
        console.print(
            f"[red]Error:[/red] Evalset not found: {evalset_path}\n"
            "  Run [bold]rageval generate-evalset[/bold] first."
        )
        raise typer.Exit(1)
    except ValueError as exc:
        console.print(f"[red]Error loading evalset:[/red] {exc}")
        raise typer.Exit(1)

    # --- apply --limit BEFORE any expensive operations ---
    if limit is not None:
        questions = questions[:limit]

    if not questions:
        console.print("[yellow]No questions to process.[/yellow]")
        return

    # --- guard: Chroma must have vectors ---
    chroma_path = project_dir / pipeline_config.vector_store.path
    from rageval.storage.chroma_dao import count as _chroma_count
    from rageval.storage.chroma_dao import get_or_create_collection as _get_col
    _col = _get_col(chroma_path, pipeline_config.vector_store.collection)
    if _chroma_count(_col) == 0:
        console.print(
            "[yellow]No vectors found in ChromaDB.[/yellow]\n"
            "  Run [bold]rageval ingest <path>[/bold] first."
        )
        raise typer.Exit(1)

    # --- build pipeline ---
    embedder = create_embedder(pipeline_config.embedding)
    llm_client = create_llm_client(pipeline_config.generation)
    pipeline = RAGPipeline(pipeline_config, embedder, llm_client, project_dir)

    # --- open DB, register questions, create run ---
    run_id = uuid.uuid4().hex
    config_hash = sha256_text(
        json.dumps(pipeline_config.model_dump(), sort_keys=True, default=str)
    )
    db_path = project_dir / ".rageval" / "runs.db"
    con = get_connection(db_path)

    evalset_id = _ensure_questions_registered(con, questions)
    create_run(
        con,
        run_id=run_id,
        name=name or f"{pipeline_config.name}-run",
        tag=tag,
        config_hash=config_hash,
        config_json=pipeline_config.model_dump_json(),
        evalset_id=evalset_id,
    )

    console.print(
        f"Running [bold]{len(questions)}[/bold] question(s) "
        f"([bold]{tag}[/bold], run=[bold]{run_id[:12]}...[/bold]) ..."
    )

    # --- process questions ---
    total_latency_ms = 0
    total_cost = 0.0
    errors = 0
    total_contexts = 0
    run_exception: Exception | None = None
    run_status = "completed"

    try:
        for q in questions:
            trace = pipeline.run_question(
                run_id=run_id,
                question_id=q.question_id,
                question=q.question,
            )
            insert_run_item(
                con,
                item_id=trace.item_id,
                run_id=trace.run_id,
                question_id=trace.question_id,
                generated_answer=trace.generated_answer,
                prompt_tokens=trace.prompt_tokens,
                completion_tokens=trace.completion_tokens,
                total_cost_usd=trace.total_cost_usd,
                latency_ms=trace.latency_ms,
                model=trace.model,
                error=trace.error,
            )
            insert_retrieved_contexts(con, trace.item_id, trace.retrieved_contexts)

            total_latency_ms += trace.latency_ms
            total_cost += trace.total_cost_usd or 0.0
            total_contexts += len(trace.retrieved_contexts)
            if trace.error:
                errors += 1

    except Exception as exc:
        run_status = "failed"
        run_exception = exc

    finally:
        try:
            finish_run(con, run_id, status=run_status)
        finally:
            con.close()

    if run_exception is not None:
        console.print(f"[red]Run failed:[/red] {run_exception}")
        raise typer.Exit(1)

    # --- summary ---
    table = Table(title="Run Summary", show_header=True)
    table.add_column("Field", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Run ID", run_id[:16] + "...")
    table.add_row("Tag", tag)
    table.add_row("Questions processed", str(len(questions)))
    table.add_row("Retrieved contexts", str(total_contexts))
    table.add_row("Total latency", f"{total_latency_ms:,} ms")
    table.add_row("Estimated cost", f"${total_cost:.6f}")
    table.add_row("Errors", str(errors))
    table.add_row("Model", pipeline_config.generation.model)
    console.print(table)

    if errors:
        console.print(
            f"[yellow]Warning:[/yellow] {errors} question(s) had errors — "
            "check DuckDB run_items.error for details."
        )
