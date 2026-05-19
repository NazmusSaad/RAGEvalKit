from __future__ import annotations

from pathlib import Path

from rageval.core.embedder import Embedder
from rageval.storage.chroma_dao import RetrievedChunk, get_or_create_collection
from rageval.storage.chroma_dao import query as _chroma_query


def retrieve_top_k(
    query_text: str,
    chroma_path: Path,
    collection_name: str,
    embedder: Embedder,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Embed *query_text* and return the top-k matching chunks from Chroma."""
    query_embedding = embedder.embed([query_text])[0]
    collection = get_or_create_collection(chroma_path, collection_name)
    return _chroma_query(collection, query_embedding, top_k)
