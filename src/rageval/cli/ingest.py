from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from rageval.cli._console import console
from rageval.core.config import PipelineConfig
from rageval.core.ingestion import ingest_corpus


def ingest(
    path: Annotated[str, typer.Argument(help="Path to file or directory to ingest")],
    config: Annotated[str, typer.Option("--config", "-c", help="Pipeline config YAML")] = "configs/baseline.yaml",
    collection: Annotated[str | None, typer.Option("--collection", help="Override ChromaDB collection name")] = None,
    chunk_size: Annotated[int | None, typer.Option("--chunk-size", help="Override chunk size")] = None,
    chunk_overlap: Annotated[int | None, typer.Option("--chunk-overlap", help="Override chunk overlap")] = None,
) -> None:
    """Load documents, chunk, embed, and persist to DuckDB + ChromaDB."""
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

    ingest_path = Path(path)
    if not ingest_path.exists():
        console.print(f"[red]Error:[/red] Path not found: {ingest_path}")
        raise typer.Exit(1)

    console.print(f"Ingesting [bold]{ingest_path}[/bold] ...")

    try:
        result = ingest_corpus(ingest_path, pipeline_config, project_dir)
    except Exception as exc:
        console.print(f"[red]Ingestion failed:[/red] {exc}")
        raise typer.Exit(1)

    if result.documents_loaded == 0:
        console.print(
            "[yellow]No documents found.[/yellow] "
            "Check path and glob pattern (.md and .txt files only)."
        )
        return

    table = Table(title="Ingest Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Documents loaded", str(result.documents_loaded))
    table.add_row("Documents new", str(result.documents_inserted))
    table.add_row("Chunks created", str(result.chunks_created))
    table.add_row("Chunks new (DuckDB)", str(result.chunks_inserted))
    table.add_row("Vectors stored (Chroma)", str(result.vectors_upserted))
    table.add_row("Collection", result.chroma_collection)
    table.add_row("Elapsed", f"{result.elapsed_seconds:.2f}s")
    console.print(table)
