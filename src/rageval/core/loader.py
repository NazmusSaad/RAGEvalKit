from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rageval.core.ids import make_doc_id

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt"})


@dataclass
class Document:
    doc_id: str
    source_path: Path | None
    text: str
    num_chars: int


def load_document(path: Path) -> Document:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    return Document(
        doc_id=make_doc_id(raw),
        source_path=path,
        text=text,
        num_chars=len(text),
    )


def load_documents(path: Path) -> list[Document]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [load_document(path)]
        return []
    docs: list[Document] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(load_document(file_path))
    return docs
