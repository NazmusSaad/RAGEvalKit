from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from rageval.cli._console import console
from rageval.core.config import PipelineConfig
from rageval.core.embedder import create_embedder
from rageval.core.retrieval import retrieve_top_k


def retrieve(
    query: Annotated[str, typer.Argument(help="Query string to search for")],
    config: Annotated[str, typer.Option("--config", "-c", help="Pipeline config YAML")] = "configs/baseline.yaml",
    top_k: Annotated[int | None, typer.Option("--top-k", "-k", help="Number of results (default: from config)")] = None,
) -> None:
    """Embed a query and print the top-k matching chunks from ChromaDB."""
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

    effective_top_k = top_k if top_k is not None else pipeline_config.retrieval.top_k
    chroma_path = project_dir / pipeline_config.vector_store.path

    try:
        embedder = create_embedder(pipeline_config.embedding)
    except Exception as exc:
        console.print(f"[red]Error creating embedder:[/red] {exc}")
        raise typer.Exit(1)

    results = retrieve_top_k(
        query_text=query,
        chroma_path=chroma_path,
        collection_name=pipeline_config.vector_store.collection,
        embedder=embedder,
        top_k=effective_top_k,
    )

    if not results:
        console.print(
            "[yellow]No results found.[/yellow] "
            "Run [bold]rageval ingest[/bold] first, or check your collection name."
        )
        return

    table = Table(title=f"Top {len(results)} results for: {query!r}", show_header=True)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Chunk ID")
    table.add_column("Doc ID")
    table.add_column("Text Preview")

    for r in results:
        preview = r.text[:80].replace("\n", " ") + ("..." if len(r.text) > 80 else "")
        table.add_row(
            str(r.rank + 1),
            f"{r.score:.4f}",
            r.chunk_id[:12] + "...",
            r.doc_id[:12] + "...",
            preview,
        )

    console.print(table)
