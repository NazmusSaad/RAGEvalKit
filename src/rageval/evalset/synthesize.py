from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Template

from rageval.core.llm import LLMClient

_SYSTEM_PROMPT = (
    "You generate evaluation questions for a RAG system. "
    "Output JSON only. No preamble, no explanation, no markdown."
)

_USER_TEMPLATE = Template(
    """\
Given the following passage, generate {{ n }} question(s) that test whether a RAG
system can correctly answer them using ONLY this passage.

Rules:
- Each question must be answerable from the passage alone.
- Vary difficulty: include factoid, multi-hop, and reasoning questions.
- Provide a concise reference answer drawn verbatim or paraphrased from the passage.
- Do not invent facts.
- Return ONLY a valid JSON object, no preamble.

PASSAGE (chunk_id={{ chunk_id }}):
\"\"\"
{{ chunk_text }}
\"\"\"

Return JSON matching this schema exactly:
{
  "questions": [
    {
      "question": "string",
      "reference_answer": "string",
      "qtype": "factoid",
      "difficulty": "easy"
    }
  ]
}
"""
)


@dataclass
class EvalQuestion:
    question_id: str
    evalset_id: str
    question: str
    reference_answer: str
    source_chunk_ids: list[str]
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    qtype: str = "factoid"      # "factoid" | "multi_hop" | "reasoning"


@dataclass
class EvalSetResult:
    evalset_id: str
    questions: list[EvalQuestion]
    source_chunks_used: int
    model: str
    parse_failures: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    match = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _parse_questions(
    raw: str,
    chunk_id: str,
    evalset_id: str,
) -> tuple[list[EvalQuestion], bool]:
    """Parse LLM output into :class:`EvalQuestion` objects.

    Returns *(questions, parse_ok)*.  On failure, *parse_ok* is ``False``
    and *questions* is empty.  Malformed individual items are silently
    dropped so one bad item doesn't discard the whole response.
    """
    for attempt in (raw.strip(), _strip_fences(raw)):
        try:
            data = json.loads(attempt)
            break
        except (json.JSONDecodeError, ValueError):
            continue
    else:
        return [], False

    items = data.get("questions", [])
    questions: list[EvalQuestion] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q_text = item.get("question", "").strip()
        a_text = item.get("reference_answer", "").strip()
        if not q_text or not a_text:
            continue
        questions.append(
            EvalQuestion(
                question_id=uuid.uuid4().hex,
                evalset_id=evalset_id,
                question=q_text,
                reference_answer=a_text,
                source_chunk_ids=[chunk_id],
                difficulty=item.get("difficulty", "medium"),
                qtype=item.get("qtype", "factoid"),
            )
        )
    return questions, True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_evalset_from_chunks(
    chunks: list[dict],
    llm_client: LLMClient,
    evalset_id: str,
    num_questions: int = 20,
    model: str = "unknown",
) -> EvalSetResult:
    """Generate evaluation questions from a list of chunk dicts.

    Each dict must have at least ``chunk_id`` and ``text`` keys.
    Questions are distributed across chunks.  Parse failures are counted
    in :attr:`EvalSetResult.parse_failures` and never raise an exception.
    """
    if not chunks:
        return EvalSetResult(
            evalset_id=evalset_id,
            questions=[],
            source_chunks_used=0,
            model=model,
            parse_failures=0,
        )

    questions_per_chunk = max(1, math.ceil(num_questions / len(chunks)))
    all_questions: list[EvalQuestion] = []
    parse_failures = 0

    for chunk in chunks:
        if len(all_questions) >= num_questions:
            break
        still_needed = num_questions - len(all_questions)
        n_to_request = min(questions_per_chunk, still_needed)

        user_prompt = _USER_TEMPLATE.render(
            n=n_to_request,
            chunk_id=chunk["chunk_id"],
            chunk_text=chunk["text"],
        )
        result = llm_client.complete(system=_SYSTEM_PROMPT, user=user_prompt)
        parsed, ok = _parse_questions(result.text, chunk["chunk_id"], evalset_id)
        if not ok:
            parse_failures += 1
        all_questions.extend(parsed)

    return EvalSetResult(
        evalset_id=evalset_id,
        questions=all_questions[:num_questions],
        source_chunks_used=len(chunks),
        model=model,
        parse_failures=parse_failures,
    )


def write_evalset_jsonl(questions: list[EvalQuestion], path: Path) -> None:
    """Write *questions* to a JSONL file, one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(
                json.dumps({
                    "question_id": q.question_id,
                    "question": q.question,
                    "reference_answer": q.reference_answer,
                    "source_chunk_ids": q.source_chunk_ids,
                    "difficulty": q.difficulty,
                    "qtype": q.qtype,
                })
                + "\n"
            )
