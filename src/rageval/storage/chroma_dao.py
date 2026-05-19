from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

from rageval.core.chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


def get_or_create_collection(
    path: str | Path,
    collection_name: str,
    distance: str = "cosine",
) -> Any:
    """Open (or create) a persistent Chroma collection at *path*."""
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance},
    )


def upsert_chunks(
    collection: Any,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    extra_metadata: list[dict[str, Any]] | None = None,
) -> None:
    """Upsert chunks and their embeddings.  Idempotent by Chroma's upsert semantics."""
    if not chunks:
        return
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        meta: dict[str, Any] = {
            "doc_id": chunk.doc_id,
            "ordinal": chunk.ordinal,
            "num_chars": chunk.num_chars,
        }
        if extra_metadata:
            meta.update(extra_metadata[i])
        metadatas.append(meta)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(
    collection: Any,
    query_embedding: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    """Return top-k :class:`RetrievedChunk` results for *query_embedding*."""
    n = min(top_k, collection.count())
    if n == 0:
        return []
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=meta.get("doc_id", ""),
            text=text,
            score=1.0 - distance,  # cosine distance → similarity
            rank=rank,
            metadata=meta,
        )
        for rank, (chunk_id, text, meta, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        )
    ]


def count(collection: Any) -> int:
    """Return the number of embeddings stored in *collection*."""
    return collection.count()
