from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_doc_id(content: bytes) -> str:
    return sha256_bytes(content)


def make_chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    return sha256_text(f"{doc_id}:{ordinal}:{text}")
