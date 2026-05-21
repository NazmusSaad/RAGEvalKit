from __future__ import annotations

import json
from pathlib import Path

from rageval.evalset.synthesize import EvalQuestion

_REQUIRED_FIELDS = frozenset({
    "question_id", "question", "reference_answer",
    "source_chunk_ids", "difficulty", "qtype",
})


def load_evalset_from_jsonl(path: Path) -> list[EvalQuestion]:
    """Load eval questions from a JSONL file.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: a line contains invalid JSON or is missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evalset file not found: {path}")

    questions: list[EvalQuestion] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {lineno}: invalid JSON — {exc}") from exc

        missing = _REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise ValueError(
                f"Line {lineno}: missing required fields: {sorted(missing)}"
            )

        questions.append(
            EvalQuestion(
                question_id=data["question_id"],
                evalset_id=data.get("evalset_id", ""),
                question=data["question"],
                reference_answer=data["reference_answer"],
                source_chunk_ids=data["source_chunk_ids"],
                difficulty=data["difficulty"],
                qtype=data["qtype"],
            )
        )
    return questions
