"""Unit tests for claim extraction — no real LLM calls."""
import pytest

from rageval.core.llm import MockLLMClient
from rageval.evaluators.claim_extraction import (
    ClaimExtractionResult,
    ExtractedClaim,
    extract_claims_for_item,
    parse_claim_extraction_json,
)

_VALID_JSON = '{"claims": ["Claim one.", "Claim two."]}'
_FENCED_JSON = '```json\n{"claims": ["Fenced claim."]}\n```'
_EMPTY_CLAIMS_JSON = '{"claims": []}'
_MISSING_FIELD_JSON = '{"other": "value"}'
_INVALID_JSON = "not json at all"


# ---------------------------------------------------------------------------
# parse_claim_extraction_json
# ---------------------------------------------------------------------------

class TestParseClaimExtractionJson:
    def test_valid_json_returns_claims(self):
        result = parse_claim_extraction_json(_VALID_JSON)
        assert len(result.claims) == 2
        assert result.label == "pass"

    def test_claim_texts_correct(self):
        result = parse_claim_extraction_json(_VALID_JSON)
        assert result.claims[0].claim_text == "Claim one."
        assert result.claims[1].claim_text == "Claim two."

    def test_claim_indices_sequential_from_zero(self):
        result = parse_claim_extraction_json(_VALID_JSON)
        assert result.claims[0].claim_idx == 0
        assert result.claims[1].claim_idx == 1

    def test_fenced_json_parsed(self):
        result = parse_claim_extraction_json(_FENCED_JSON)
        assert result.label == "pass"
        assert len(result.claims) == 1
        assert result.claims[0].claim_text == "Fenced claim."

    def test_plain_fences_stripped(self):
        fenced = '```\n{"claims": ["Plain fence."]}\n```'
        result = parse_claim_extraction_json(fenced)
        assert result.label == "pass"
        assert len(result.claims) == 1

    def test_invalid_json_returns_unknown(self):
        result = parse_claim_extraction_json(_INVALID_JSON)
        assert result.label == "unknown"
        assert result.claims == []

    def test_empty_string_returns_unknown(self):
        result = parse_claim_extraction_json("")
        assert result.label == "unknown"

    def test_missing_claims_field_returns_unknown(self):
        result = parse_claim_extraction_json(_MISSING_FIELD_JSON)
        assert result.label == "unknown"

    def test_non_list_claims_returns_unknown(self):
        result = parse_claim_extraction_json('{"claims": "not a list"}')
        assert result.label == "unknown"

    def test_empty_claims_list_returns_pass_zero_claims(self):
        result = parse_claim_extraction_json(_EMPTY_CLAIMS_JSON)
        assert result.label == "pass"
        assert result.claims == []

    def test_strips_whitespace_from_claim_text(self):
        raw = '{"claims": ["  Padded claim.  "]}'
        result = parse_claim_extraction_json(raw)
        assert result.claims[0].claim_text == "Padded claim."

    def test_drops_empty_claim_strings(self):
        raw = '{"claims": ["Valid.", "", "  "]}'
        result = parse_claim_extraction_json(raw)
        assert len(result.claims) == 1
        assert result.claims[0].claim_text == "Valid."

    def test_preserves_claim_order(self):
        raw = '{"claims": ["First.", "Second.", "Third."]}'
        result = parse_claim_extraction_json(raw)
        texts = [c.claim_text for c in result.claims]
        assert texts == ["First.", "Second.", "Third."]

    def test_indices_resequenced_after_dropping_empties(self):
        raw = '{"claims": ["A.", "", "B."]}'
        result = parse_claim_extraction_json(raw)
        assert len(result.claims) == 2
        assert result.claims[0].claim_idx == 0
        assert result.claims[1].claim_idx == 1

    def test_raw_json_stored_on_success(self):
        result = parse_claim_extraction_json(_VALID_JSON)
        assert "claims" in result.raw_json

    def test_returns_claim_extraction_result(self):
        result = parse_claim_extraction_json(_VALID_JSON)
        assert isinstance(result, ClaimExtractionResult)

    def test_claims_are_extracted_claim_objects(self):
        result = parse_claim_extraction_json(_VALID_JSON)
        assert all(isinstance(c, ExtractedClaim) for c in result.claims)

    def test_reason_mentions_count_on_success(self):
        result = parse_claim_extraction_json(_VALID_JSON)
        assert "2" in result.reason


# ---------------------------------------------------------------------------
# extract_claims_for_item
# ---------------------------------------------------------------------------

class TestExtractClaimsForItem:
    def test_valid_response_returns_claims(self):
        mock = MockLLMClient(response_text=_VALID_JSON)
        result = extract_claims_for_item("Some answer.", mock)
        assert len(result.claims) == 2
        assert result.label == "pass"

    def test_empty_answer_returns_pass_zero_claims(self):
        mock = MockLLMClient(response_text=_VALID_JSON)
        result = extract_claims_for_item("", mock)
        assert result.label == "pass"
        assert result.claims == []

    def test_empty_answer_does_not_call_llm(self):
        mock = MockLLMClient(response_text=_VALID_JSON)
        extract_claims_for_item("", mock)
        assert len(mock.calls) == 0

    def test_whitespace_only_answer_returns_pass_zero_claims(self):
        mock = MockLLMClient(response_text=_VALID_JSON)
        result = extract_claims_for_item("   \n  ", mock)
        assert result.label == "pass"
        assert result.claims == []

    def test_whitespace_only_does_not_call_llm(self):
        mock = MockLLMClient(response_text=_VALID_JSON)
        extract_claims_for_item("   ", mock)
        assert len(mock.calls) == 0

    def test_invalid_json_response_returns_unknown(self):
        mock = MockLLMClient(response_text="bad json")
        result = extract_claims_for_item("Some answer.", mock)
        assert result.label == "unknown"
        assert result.claims == []

    def test_llm_called_once_for_non_empty_answer(self):
        mock = MockLLMClient(response_text=_VALID_JSON)
        extract_claims_for_item("Some answer.", mock)
        assert len(mock.calls) == 1

    def test_answer_included_in_user_prompt(self):
        mock = MockLLMClient(response_text=_VALID_JSON)
        extract_claims_for_item("My specific answer text.", mock)
        assert "My specific answer text." in mock.calls[0]["user"]

    def test_fenced_response_parsed(self):
        mock = MockLLMClient(response_text=_FENCED_JSON)
        result = extract_claims_for_item("Some answer.", mock)
        assert result.label == "pass"
        assert len(result.claims) == 1

    def test_empty_claims_response_is_pass(self):
        mock = MockLLMClient(response_text=_EMPTY_CLAIMS_JSON)
        result = extract_claims_for_item("Non-empty answer.", mock)
        assert result.label == "pass"
        assert result.claims == []
