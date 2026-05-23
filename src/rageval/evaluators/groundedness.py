"""LLM-as-judge groundedness evaluator.

Per-claim verdicts: supported | contradicted | not_enough_info | unknown
Item-level faithfulness: supported_claims / total_claims  (unknowns count as 0)

Label policy
------------
pass    — faithfulness >= 0.75
fail    — faithfulness < 0.75  (and at least one non-unknown verdict)
unknown — zero claims, or ALL verdicts unknown (judge failures)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rageval.core.llm import LLMClient

_VALID_VERDICTS = frozenset({"supported", "contradicted", "not_enough_info"})
_PASS_THRESHOLD = 0.75

_SYSTEM = (
    "You verify whether claims are supported by source passages. "
    "Output JSON only. No preamble, no markdown."
)

_USER_TEMPLATE = """\
Claim: {claim}

Sources:
{sources}

Decide:
- "supported"       if at least one source directly supports the claim
- "contradicted"    if at least one source directly contradicts the claim
- "not_enough_info" otherwise (partial overlap, ambiguous, or absent)

Return JSON only:
{{ "verdict": "supported" | "contradicted" | "not_enough_info", "supporting_indices": [int, ...], "rationale": "<=30 words" }}
"""


@dataclass
class GroundednessClaimResult:
    verdict: str                          # "supported" | "contradicted" | "not_enough_info" | "unknown"
    supporting_indices: list[int] = field(default_factory=list)
    rationale: str = ""
    raw_json: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundednessItemResult:
    claim_results: list[GroundednessClaimResult]
    faithfulness: float                   # supported / total_claims
    label: str                            # "pass" | "fail" | "unknown"
    reason: str


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def parse_groundedness_json(raw: str) -> GroundednessClaimResult:
    """Parse the judge response for one claim.

    Negative supporting indices are dropped; positive but out-of-range ones
    are passed through and filtered in the caller when mapping to chunk_ids.
    """
    data: Any = None
    for attempt in (raw.strip(), _strip_fences(raw)):
        try:
            data = json.loads(attempt)
            break
        except (json.JSONDecodeError, ValueError):
            continue

    if data is None:
        return GroundednessClaimResult(
            verdict="unknown",
            rationale="judge response could not be parsed as JSON",
        )

    if not isinstance(data, dict) or "verdict" not in data:
        return GroundednessClaimResult(
            verdict="unknown",
            rationale="judge response missing required 'verdict' field",
            raw_json=data if isinstance(data, dict) else {},
        )

    verdict = data.get("verdict", "")
    if verdict not in _VALID_VERDICTS:
        return GroundednessClaimResult(
            verdict="unknown",
            rationale=f"judge returned invalid verdict: {verdict!r}",
            raw_json=data,
        )

    raw_indices = data.get("supporting_indices", [])
    indices = (
        [i for i in raw_indices if isinstance(i, int) and i >= 0]
        if isinstance(raw_indices, list)
        else []
    )

    return GroundednessClaimResult(
        verdict=verdict,
        supporting_indices=indices,
        rationale=str(data.get("rationale", "")),
        raw_json=data,
    )


def evaluate_claim_groundedness(
    claim_text: str,
    retrieved_contexts: list[dict],
    llm_client: LLMClient,
) -> GroundednessClaimResult:
    """Judge one claim against retrieved contexts.

    *retrieved_contexts* is a list of dicts with ``rank``, ``chunk_id``,
    ``chunk_text`` keys, ordered by rank.
    """
    sources = "\n".join(
        f"[{i}] {ctx.get('chunk_text', '')}"
        for i, ctx in enumerate(retrieved_contexts)
    )
    user_prompt = _USER_TEMPLATE.format(
        claim=claim_text,
        sources=sources if sources else "(no context available)",
    )
    try:
        result = llm_client.complete(system=_SYSTEM, user=user_prompt)
        return parse_groundedness_json(result.text)
    except Exception as exc:  # noqa: BLE001
        return GroundednessClaimResult(
            verdict="unknown",
            rationale=f"judge call failed: {exc}",
        )


def evaluate_groundedness_for_item(
    claims: list[dict],
    retrieved_contexts: list[dict],
    llm_client: LLMClient,
) -> GroundednessItemResult:
    """Judge groundedness for all claims of one run item.

    *claims*: rows from ``get_claims_for_item`` (each has ``claim_idx``, ``claim_text``).
    *retrieved_contexts*: rows from ``get_retrieved_contexts_for_item``, ordered by rank.

    Faithfulness = supported_count / total_claims (unknowns count as zero).
    Returns ``label="unknown"`` when there are no claims or all verdicts are unknown.
    """
    if not claims:
        return GroundednessItemResult(
            claim_results=[],
            faithfulness=0.0,
            label="unknown",
            reason="no claims to judge",
        )

    results: list[GroundednessClaimResult] = []
    for claim in claims:
        r = evaluate_claim_groundedness(claim["claim_text"], retrieved_contexts, llm_client)
        results.append(r)

    total = len(results)
    supported = sum(1 for r in results if r.verdict == "supported")
    unknown_count = sum(1 for r in results if r.verdict == "unknown")

    if unknown_count == total:
        return GroundednessItemResult(
            claim_results=results,
            faithfulness=0.0,
            label="unknown",
            reason="all claim judgments produced unknown verdict",
        )

    faithfulness = supported / total
    label = "pass" if faithfulness >= _PASS_THRESHOLD else "fail"
    reason = f"{supported}/{total} claims supported (faithfulness={faithfulness:.3f})"

    return GroundednessItemResult(
        claim_results=results,
        faithfulness=faithfulness,
        label=label,
        reason=reason,
    )
