from __future__ import annotations

from dataclasses import dataclass

from rageval.core.ids import make_chunk_id
from rageval.core.loader import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    num_chars: int


class SimpleChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        text = document.text
        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        ordinal = 0
        start = 0
        while start < len(text):
            chunk_text = text[start : start + self.chunk_size].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(document.doc_id, ordinal, chunk_text),
                        doc_id=document.doc_id,
                        ordinal=ordinal,
                        text=chunk_text,
                        num_chars=len(chunk_text),
                    )
                )
                ordinal += 1
            start += step
        return chunks
