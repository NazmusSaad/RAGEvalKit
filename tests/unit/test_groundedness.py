"""Unit tests for groundedness evaluator — no real LLM calls."""
import pytest

from rageval.core.llm import CompletionResult, MockLLMClient
from rageval.evaluators.groundedness import (
    GroundednessClaimResult,
    GroundednessItemResult,
    evaluate_claim_groundedness,
    evaluate_groundedness_for_item,
    parse_groundedness_json,
)

_SUPPORTED = '{"verdict": "supported", "supporting_indices": [0], "rationale": "Source supports."}'
_CONTRADICTED = '{"verdict": "contradicted", "supporting_indices": [], "rationale": "Source contradicts."}'
_NOT_ENOUGH = '{"verdict": "not_enough_info", "supporting_indices": [], "rationale": "Unclear."}'
_FENCED = '```json\n{"verdict": "supported", "supporting_indices": [1], "rationale": "OK."}\n```'
_INVALID_JSON = "not json at all"
_INVALID_VERDICT = '{"verdict": "maybe", "supporting_indices": [], "rationale": "Hmm."}'
_MISSING_VERDICT = '{"supporting_indices": [0], "rationale": "No verdict."}'

_CTX = [
    {"rank": 0, "chunk_id": "c0", "chunk_text": "Context A."},
    {"rank": 1, "chunk_id": "c1", "chunk_text": "Context B."},
]
_CLAIMS_1 = [{"claim_idx": 0, "claim_text": "Claim one."}]
_CLAIMS_2 = [{"claim_idx": 0, "claim_text": "Claim A."}, {"claim_idx": 1, "claim_text": "Claim B."}]


# ---------------------------------------------------------------------------
# parse_groundedness_json
# ---------------------------------------------------------------------------

class TestParseGroundednessJson:
    def test_supported_verdict(self):
        r = parse_groundedness_json(_SUPPORTED)
        assert r.verdict == "supported"
        assert r.supporting_indices == [0]
        assert r.rationale == "Source supports."

    def test_contradicted_verdict(self):
        r = parse_groundedness_json(_CONTRADICTED)
        assert r.verdict == "contradicted"
        assert r.supporting_indices == []

    def test_not_enough_info_verdict(self):
        r = parse_groundedness_json(_NOT_ENOUGH)
        assert r.verdict == "not_enough_info"

    def test_fenced_json_parsed(self):
        r = parse_groundedness_json(_FENCED)
        assert r.verdict == "supported"
        assert r.supporting_indices == [1]

    def test_plain_fences(self):
        fenced = '```\n{"verdict": "supported", "supporting_indices": [], "rationale": "OK."}\n```'
        r = parse_groundedness_json(fenced)
        assert r.verdict == "supported"

    def test_invalid_json_returns_unknown(self):
        r = parse_groundedness_json(_INVALID_JSON)
        assert r.verdict == "unknown"

    def test_empty_string_returns_unknown(self):
        r = parse_groundedness_json("")
        assert r.verdict == "unknown"

    def test_invalid_verdict_returns_unknown(self):
        r = parse_groundedness_json(_INVALID_VERDICT)
        assert r.verdict == "unknown"

    def test_missing_verdict_field_returns_unknown(self):
        r = parse_groundedness_json(_MISSING_VERDICT)
        assert r.verdict == "unknown"

    def test_negative_supporting_index_dropped(self):
        raw = '{"verdict": "supported", "supporting_indices": [-1, 0], "rationale": "OK."}'
        r = parse_groundedness_json(raw)
        assert -1 not in r.supporting_indices
        assert 0 in r.supporting_indices

    def test_non_integer_supporting_index_dropped(self):
        raw = '{"verdict": "supported", "supporting_indices": ["a", 0], "rationale": "OK."}'
        r = parse_groundedness_json(raw)
        assert r.supporting_indices == [0]

    def test_non_list_supporting_indices_gives_empty(self):
        raw = '{"verdict": "supported", "supporting_indices": 0, "rationale": "OK."}'
        r = parse_groundedness_json(raw)
        assert r.supporting_indices == []

    def test_raw_json_stored(self):
        r = parse_groundedness_json(_SUPPORTED)
        assert "verdict" in r.raw_json

    def test_returns_groundedness_claim_result(self):
        r = parse_groundedness_json(_SUPPORTED)
        assert isinstance(r, GroundednessClaimResult)


# ---------------------------------------------------------------------------
# evaluate_claim_groundedness
# ---------------------------------------------------------------------------

class TestEvaluateClaimGroundedness:
    def test_returns_claim_result(self):
        mock = MockLLMClient(response_text=_SUPPORTED)
        r = evaluate_claim_groundedness("A claim.", _CTX, mock)
        assert isinstance(r, GroundednessClaimResult)

    def test_supported_verdict_returned(self):
        mock = MockLLMClient(response_text=_SUPPORTED)
        r = evaluate_claim_groundedness("A claim.", _CTX, mock)
        assert r.verdict == "supported"

    def test_invalid_response_returns_unknown(self):
        mock = MockLLMClient(response_text="bad json")
        r = evaluate_claim_groundedness("A claim.", _CTX, mock)
        assert r.verdict == "unknown"

    def test_claim_in_user_prompt(self):
        mock = MockLLMClient(response_text=_SUPPORTED)
        evaluate_claim_groundedness("My specific claim.", _CTX, mock)
        assert "My specific claim." in mock.calls[0]["user"]

    def test_context_texts_in_prompt(self):
        mock = MockLLMClient(response_text=_SUPPORTED)
        evaluate_claim_groundedness("Claim.", _CTX, mock)
        assert "Context A." in mock.calls[0]["user"]
        assert "Context B." in mock.calls[0]["user"]

    def test_empty_context_handled(self):
        mock = MockLLMClient(response_text=_SUPPORTED)
        r = evaluate_claim_groundedness("A claim.", [], mock)
        assert r.verdict == "supported"  # parsed normally; no context = still valid parse


# ---------------------------------------------------------------------------
# evaluate_groundedness_for_item
# ---------------------------------------------------------------------------

class TestEvaluateGroundednessForItem:
    def test_no_claims_returns_unknown(self):
        mock = MockLLMClient(response_text=_SUPPORTED)
        r = evaluate_groundedness_for_item([], _CTX, mock)
        assert r.label == "unknown"
        assert r.faithfulness == 0.0

    def test_all_supported_faithfulness_1_0(self):
        mock = MockLLMClient(response_text=_SUPPORTED)
        r = evaluate_groundedness_for_item(_CLAIMS_2, _CTX, mock)
        assert r.faithfulness == pytest.approx(1.0)
        assert r.label == "pass"

    def test_all_contradicted_faithfulness_0_0(self):
        mock = MockLLMClient(response_text=_CONTRADICTED)
        r = evaluate_groundedness_for_item(_CLAIMS_2, _CTX, mock)
        assert r.faithfulness == pytest.approx(0.0)
        assert r.label == "fail"

    def test_all_not_enough_info_faithfulness_0_0(self):
        mock = MockLLMClient(response_text=_NOT_ENOUGH)
        r = evaluate_groundedness_for_item(_CLAIMS_1, _CTX, mock)
        assert r.faithfulness == pytest.approx(0.0)
        assert r.label == "fail"

    def test_half_supported_faithfulness_0_5(self):
        responses = [_SUPPORTED, _CONTRADICTED]
        idx = 0

        class _Alt:
            def complete(self, system, user, **_):
                nonlocal idx
                text = responses[idx % 2]
                idx += 1
                return CompletionResult(
                    text=text, prompt_tokens=10, completion_tokens=5,
                    total_tokens=15, cost_usd=None, model="mock",
                )

        r = evaluate_groundedness_for_item(_CLAIMS_2, _CTX, _Alt())
        assert r.faithfulness == pytest.approx(0.5)
        assert r.label == "fail"

    def test_faithfulness_0_75_is_pass(self):
        # 3 supported out of 4 = 0.75 → pass
        three_claims = [{"claim_idx": i, "claim_text": f"C{i}."} for i in range(4)]
        responses = [_SUPPORTED] * 3 + [_CONTRADICTED]
        idx = 0

        class _Alt:
            def complete(self, system, user, **_):
                nonlocal idx
                text = responses[min(idx, len(responses) - 1)]
                idx += 1
                return CompletionResult(
                    text=text, prompt_tokens=10, completion_tokens=5,
                    total_tokens=15, cost_usd=None, model="mock",
                )

        r = evaluate_groundedness_for_item(three_claims, _CTX, _Alt())
        assert r.faithfulness == pytest.approx(0.75)
        assert r.label == "pass"

    def test_all_unknown_verdicts_returns_unknown(self):
        mock = MockLLMClient(response_text="bad json")
        r = evaluate_groundedness_for_item(_CLAIMS_2, _CTX, mock)
        assert r.label == "unknown"
        assert r.faithfulness == 0.0

    def test_claim_results_length_matches_claims(self):
        mock = MockLLMClient(response_text=_SUPPORTED)
        r = evaluate_groundedness_for_item(_CLAIMS_2, _CTX, mock)
        assert len(r.claim_results) == 2

    def test_reason_mentions_counts(self):
        mock = MockLLMClient(response_text=_SUPPORTED)
        r = evaluate_groundedness_for_item(_CLAIMS_1, _CTX, mock)
        assert "1" in r.reason

    def test_returns_item_result(self):
        mock = MockLLMClient(response_text=_SUPPORTED)
        r = evaluate_groundedness_for_item(_CLAIMS_1, _CTX, mock)
        assert isinstance(r, GroundednessItemResult)
