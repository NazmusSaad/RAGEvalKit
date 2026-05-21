from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from rageval.cli._console import console
from rageval.core.config import PipelineConfig
from rageval.core.llm import create_llm_client
from rageval.evalset.synthesize import generate_evalset_from_chunks, write_evalset_jsonl
from rageval.storage.duckdb_dao import (
    create_eval_set,
    get_connection,
    get_sample_chunks,
    insert_eval_questions,
)


def generate_evalset(
    docs_path: Annotated[str, typer.Argument(help="Document corpus path (for consistency; reads chunks from DuckDB)")],
    num_questions: Annotated[int, typer.Option("--num-questions", "-n", help="Number of questions to generate")] = 20,
    output: Annotated[str, typer.Option("--output", "-o", help="Output JSONL file")] = "evalsets/auto.jsonl",
    config: Annotated[str, typer.Option("--config", "-c", help="Pipeline config YAML")] = "configs/baseline.yaml",
    diversity: Annotated[str, typer.Option("--diversity", help="Diversity mode hint (reserved)")] = "mixed",
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed (reserved)")] = None,
) -> None:
    """Synthesize an eval set from already-ingested chunks using an LLM."""
    project_dir = Path.cwd()

    if not (project_dir / ".rageval").exists():
        console.print(
            "[red]Error:[/red] .rageval/ not found.\n"
            "  Run [bold]rageval init[/bold] first."
        )
        raise typer.Exit(1)

    config_path = project_dir / config
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_path}")
        raise typer.Exit(1)

    try:
        pipeline_config = PipelineConfig.from_yaml(config_path)
    except Exception as exc:
        console.print(f"[red]Error loading config:[/red] {exc}")
        raise typer.Exit(1)

    db_path = project_dir / ".rageval" / "runs.db"
    con = get_connection(db_path)
    try:
        chunks = get_sample_chunks(con, n=min(num_questions * 3, 1000))
        if not chunks:
            console.print(
                "[yellow]No chunks found in DuckDB.[/yellow]\n"
                "  Run [bold]rageval ingest <path>[/bold] first."
            )
            raise typer.Exit(1)

        evalset_id = uuid.uuid4().hex
        llm_client = create_llm_client(pipeline_config.judge)

        console.print(
            f"Generating up to [bold]{num_questions}[/bold] questions "
            f"from [bold]{len(chunks)}[/bold] chunks "
            f"using [bold]{pipeline_config.judge.model}[/bold] ..."
        )

        result = generate_evalset_from_chunks(
            chunks=chunks,
            llm_client=llm_client,
            evalset_id=evalset_id,
            num_questions=num_questions,
            model=pipeline_config.judge.model,
        )

        create_eval_set(
            con,
            evalset_id=evalset_id,
            name=f"{pipeline_config.name}-evalset",
            generated_by="synthetic",
            config_json=json.dumps({
                "model": pipeline_config.judge.model,
                "num_questions": num_questions,
            }),
        )
        insert_eval_questions(con, result.questions)
    finally:
        con.close()

    output_path = project_dir / output
    write_evalset_jsonl(result.questions, output_path)

    table = Table(title="Generate EvalSet Summary", show_header=True)
    table.add_column("Field", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("EvalSet ID", evalset_id[:16] + "...")
    table.add_row("Questions generated", str(len(result.questions)))
    table.add_row("Questions requested", str(num_questions))
    table.add_row("Parse failures", str(result.parse_failures))
    table.add_row("Source chunks used", str(result.source_chunks_used))
    table.add_row("Model", result.model)
    try:
        display_path = str(output_path.relative_to(project_dir))
    except ValueError:
        display_path = str(output_path)
    table.add_row("Output", display_path)
    console.print(table)

    if result.parse_failures > 0:
        console.print(
            f"[yellow]Warning:[/yellow] {result.parse_failures} chunk(s) returned "
            "unparseable JSON and were skipped."
        )
