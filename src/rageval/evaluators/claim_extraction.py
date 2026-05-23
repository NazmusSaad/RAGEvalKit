"""LLM-based claim extraction: decompose a generated answer into atomic claims.

Label policy
------------
pass    — extraction completed (0 or more claims returned)
unknown — judge response could not be parsed or is structurally invalid

An empty answer returns label="pass" with zero claims ("extraction completed;
nothing to decompose").  An LLM that returns {"claims": []} also yields
label="pass" with zero claims.  "unknown" is reserved for judge failures.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rageval.core.llm import LLMClient

_SYSTEM = (
    "You decompose answers into atomic factual claims. "
    "Output JSON only. No preamble, no explanation, no markdown."
)

_USER_TEMPLATE = """\
Decompose the following answer into a list of atomic claims. Rules:
- One distinct fact per claim.
- Rewrite pronouns and references to be self-contained.
- Exclude meta-commentary, hedges, or non-factual sentences.
- Return an empty list if there are no verifiable claims.

ANSWER:
\"\"\"{answer}\"\"\"

Return JSON only:
{{ "claims": ["string", "string", ...] }}
"""


@dataclass
class ExtractedClaim:
    claim_idx: int
    claim_text: str


@dataclass
class ClaimExtractionResult:
    claims: list[ExtractedClaim]
    label: str   # "pass" | "unknown"
    reason: str
    raw_json: dict[str, Any] = field(default_factory=dict)


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def parse_claim_extraction_json(raw: str) -> ClaimExtractionResult:
    """Parse the LLM claim-extraction response.

    Two-pass: tries raw text first, then strips markdown fences.
    Returns ``label="unknown"`` for parse/structural failures.
    Returns ``label="pass"`` (possibly with zero claims) otherwise.
    """
    data: Any = None
    for attempt in (raw.strip(), _strip_fences(raw)):
        try:
            data = json.loads(attempt)
            break
        except (json.JSONDecodeError, ValueError):
            continue

    if data is None:
        return ClaimExtractionResult(
            claims=[],
            label="unknown",
            reason="judge response could not be parsed as JSON",
        )

    if not isinstance(data, dict) or "claims" not in data:
        return ClaimExtractionResult(
            claims=[],
            label="unknown",
            reason="judge response missing required 'claims' field",
            raw_json=data if isinstance(data, dict) else {},
        )

    raw_claims = data["claims"]
    if not isinstance(raw_claims, list):
        return ClaimExtractionResult(
            claims=[],
            label="unknown",
            reason=f"'claims' field is not a list: {type(raw_claims).__name__}",
            raw_json=data,
        )

    # Strip whitespace, drop empty strings, preserve order
    filtered = [str(c).strip() for c in raw_claims if str(c).strip()]
    claims = [ExtractedClaim(claim_idx=i, claim_text=text) for i, text in enumerate(filtered)]

    reason = f"{len(claims)} claim(s) extracted" if claims else "LLM returned no claims"
    return ClaimExtractionResult(claims=claims, label="pass", reason=reason, raw_json=data)


def extract_claims_for_item(
    generated_answer: str,
    llm_client: LLMClient,
) -> ClaimExtractionResult:
    """Decompose *generated_answer* into atomic claims via the judge LLM.

    Returns ``label="pass"`` with zero claims for empty answers — extraction
    completed successfully; there is simply nothing to decompose.
    Returns ``label="unknown"`` on judge failures without propagating exceptions.
    """
    if not generated_answer or not generated_answer.strip():
        return ClaimExtractionResult(
            claims=[],
            label="pass",
            reason="no answer to decompose",
        )

    user_prompt = _USER_TEMPLATE.format(answer=generated_answer)
    try:
        result = llm_client.complete(system=_SYSTEM, user=user_prompt)
        return parse_claim_extraction_json(result.text)
    except Exception as exc:  # noqa: BLE001
        return ClaimExtractionResult(
            claims=[],
            label="unknown",
            reason=f"judge call failed: {exc}",
        )
