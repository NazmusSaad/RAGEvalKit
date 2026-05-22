"""LLM-as-judge evaluator for answer relevance.

Rubric (raw score 0-4, normalized to 0.0-1.0):
  4 = Directly and fully addresses the question.
  3 = Addresses the question but with minor irrelevant content.
  2 = Partially addresses; significant off-topic content.
  1 = Mostly off-topic but mentions the question subject.
  0 = Off-topic / non-answer / empty / refusal.

Label policy:
  pass    — normalized score >= 0.75 (raw >= 3)
  fail    — normalized score < 0.75  (raw < 3) with a valid parse
  unknown — judge output unparseable or missing required fields
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rageval.core.llm import LLMClient

_PASS_THRESHOLD = 0.75
_MAX_RAW_SCORE = 4

_SYSTEM = (
    "You score answer relevance. Output JSON only. "
    "No preamble, no explanation, no markdown."
)

_USER_TEMPLATE = """\
Question: {question}
Answer:   {answer}

Score 0-4:
4 = Directly and fully addresses the question.
3 = Addresses the question but with minor irrelevant content.
2 = Partially addresses; significant off-topic content.
1 = Mostly off-topic but mentions the question subject.
0 = Off-topic / non-answer / empty answer / refusal.

Return JSON only:
{{ "score": 0|1|2|3|4, "reason": "short explanation (<=25 words)" }}
"""


@dataclass
class AnswerRelevanceResult:
    metric: str = "answer_relevance"
    score: float = 0.0
    label: str = "unknown"
    reason: str = ""
    raw_json: dict[str, Any] = field(default_factory=dict)


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def parse_answer_relevance_json(raw: str) -> AnswerRelevanceResult:
    """Parse the LLM judge response into an :class:`AnswerRelevanceResult`.

    Two-pass parser: tries raw text first, then strips markdown fences.
    Returns ``label="unknown"`` on any parse or validation failure.
    """
    data: Any = None
    for attempt in (raw.strip(), _strip_fences(raw)):
        try:
            data = json.loads(attempt)
            break
        except (json.JSONDecodeError, ValueError):
            continue

    if data is None:
        return AnswerRelevanceResult(
            score=0.0,
            label="unknown",
            reason="judge response could not be parsed as JSON",
        )

    if not isinstance(data, dict) or "score" not in data:
        return AnswerRelevanceResult(
            score=0.0,
            label="unknown",
            reason="judge response missing required 'score' field",
            raw_json=data if isinstance(data, dict) else {},
        )

    raw_score = data["score"]
    if not isinstance(raw_score, (int, float)) or not (0 <= raw_score <= _MAX_RAW_SCORE):
        return AnswerRelevanceResult(
            score=0.0,
            label="unknown",
            reason=f"judge returned invalid score: {raw_score!r}",
            raw_json=data,
        )

    score = float(raw_score) / _MAX_RAW_SCORE
    label = "pass" if score >= _PASS_THRESHOLD else "fail"
    reason = str(data.get("reason", ""))

    return AnswerRelevanceResult(
        score=score,
        label=label,
        reason=reason,
        raw_json=data,
    )


def evaluate_answer_relevance_for_item(
    question: str,
    generated_answer: str,
    llm_client: LLMClient,
) -> AnswerRelevanceResult:
    """Score answer relevance by calling the judge LLM.

    Returns ``label="unknown"`` on parse failures without propagating exceptions.
    """
    user_prompt = _USER_TEMPLATE.format(
        question=question or "(no question)",
        answer=generated_answer or "(empty answer)",
    )
    try:
        result = llm_client.complete(system=_SYSTEM, user=user_prompt)
        return parse_answer_relevance_json(result.text)
    except Exception as exc:  # noqa: BLE001
        return AnswerRelevanceResult(
            score=0.0,
            label="unknown",
            reason=f"judge call failed: {exc}",
        )
