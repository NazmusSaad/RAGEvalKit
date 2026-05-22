"""Unit tests for answer relevance evaluator — no real LLM calls."""
import json

import pytest

from rageval.core.llm import MockLLMClient
from rageval.evaluators.answer_relevance import (
    AnswerRelevanceResult,
    evaluate_answer_relevance_for_item,
    parse_answer_relevance_json,
)


# ---------------------------------------------------------------------------
# parse_answer_relevance_json
# ---------------------------------------------------------------------------

class TestParseAnswerRelevanceJson:
    def test_valid_json_score_4(self):
        result = parse_answer_relevance_json('{"score": 4, "reason": "Perfect answer."}')
        assert result.score == pytest.approx(1.0)
        assert result.label == "pass"

    def test_valid_json_score_3(self):
        result = parse_answer_relevance_json('{"score": 3, "reason": "Good answer."}')
        assert result.score == pytest.approx(0.75)
        assert result.label == "pass"

    def test_valid_json_score_2(self):
        result = parse_answer_relevance_json('{"score": 2, "reason": "Partial."}')
        assert result.score == pytest.approx(0.5)
        assert result.label == "fail"

    def test_valid_json_score_1(self):
        result = parse_answer_relevance_json('{"score": 1, "reason": "Off-topic."}')
        assert result.score == pytest.approx(0.25)
        assert result.label == "fail"

    def test_valid_json_score_0(self):
        result = parse_answer_relevance_json('{"score": 0, "reason": "No answer."}')
        assert result.score == pytest.approx(0.0)
        assert result.label == "fail"

    def test_pass_threshold_is_0_75(self):
        # score 3 → 0.75 → exactly at threshold → pass
        assert parse_answer_relevance_json('{"score": 3, "reason": "OK."}').label == "pass"
        # score 2 → 0.5 → below threshold → fail
        assert parse_answer_relevance_json('{"score": 2, "reason": "Partial."}').label == "fail"

    def test_fenced_json_parsed(self):
        fenced = '```json\n{"score": 3, "reason": "OK."}\n```'
        result = parse_answer_relevance_json(fenced)
        assert result.score == pytest.approx(0.75)
        assert result.label == "pass"

    def test_plain_fenced_json(self):
        fenced = '```\n{"score": 4, "reason": "Great."}\n```'
        result = parse_answer_relevance_json(fenced)
        assert result.score == pytest.approx(1.0)

    def test_invalid_json_returns_unknown(self):
        result = parse_answer_relevance_json("not json at all")
        assert result.label == "unknown"
        assert result.score == 0.0

    def test_empty_string_returns_unknown(self):
        result = parse_answer_relevance_json("")
        assert result.label == "unknown"

    def test_missing_score_field_returns_unknown(self):
        result = parse_answer_relevance_json('{"reason": "No score here."}')
        assert result.label == "unknown"

    def test_score_out_of_range_high_returns_unknown(self):
        result = parse_answer_relevance_json('{"score": 5, "reason": "Out of range."}')
        assert result.label == "unknown"

    def test_score_out_of_range_negative_returns_unknown(self):
        result = parse_answer_relevance_json('{"score": -1, "reason": "Negative."}')
        assert result.label == "unknown"

    def test_non_numeric_score_returns_unknown(self):
        result = parse_answer_relevance_json('{"score": "three", "reason": "String."}')
        assert result.label == "unknown"

    def test_reason_extracted(self):
        result = parse_answer_relevance_json('{"score": 4, "reason": "Directly answers."}')
        assert result.reason == "Directly answers."

    def test_missing_reason_defaults_to_empty(self):
        result = parse_answer_relevance_json('{"score": 3}')
        assert result.reason == ""

    def test_raw_json_stored_on_success(self):
        result = parse_answer_relevance_json('{"score": 3, "reason": "Good."}')
        assert result.raw_json == {"score": 3, "reason": "Good."}

    def test_raw_json_stored_on_unknown_with_dict(self):
        result = parse_answer_relevance_json('{"score": 99, "reason": "Bad."}')
        assert result.label == "unknown"
        assert isinstance(result.raw_json, dict)

    def test_metric_is_answer_relevance(self):
        result = parse_answer_relevance_json('{"score": 3, "reason": "Good."}')
        assert result.metric == "answer_relevance"

    def test_returns_answer_relevance_result_object(self):
        result = parse_answer_relevance_json('{"score": 3, "reason": "Good."}')
        assert isinstance(result, AnswerRelevanceResult)

    def test_score_0_is_float(self):
        result = parse_answer_relevance_json('{"score": 0, "reason": "None."}')
        assert isinstance(result.score, float)


# ---------------------------------------------------------------------------
# evaluate_answer_relevance_for_item
# ---------------------------------------------------------------------------

class TestEvaluateAnswerRelevanceForItem:
    def test_returns_result_object(self):
        mock = MockLLMClient(response_text='{"score": 3, "reason": "Good."}')
        result = evaluate_answer_relevance_for_item("What is RAG?", "RAG is...", mock)
        assert isinstance(result, AnswerRelevanceResult)

    def test_valid_response_scored_correctly(self):
        mock = MockLLMClient(response_text='{"score": 4, "reason": "Perfect."}')
        result = evaluate_answer_relevance_for_item("Q?", "A.", mock)
        assert result.score == pytest.approx(1.0)
        assert result.label == "pass"

    def test_invalid_response_returns_unknown(self):
        mock = MockLLMClient(response_text="bad json")
        result = evaluate_answer_relevance_for_item("Q?", "A.", mock)
        assert result.label == "unknown"

    def test_llm_called_once(self):
        mock = MockLLMClient(response_text='{"score": 3, "reason": "OK."}')
        evaluate_answer_relevance_for_item("Q?", "A.", mock)
        assert len(mock.calls) == 1

    def test_question_in_prompt(self):
        mock = MockLLMClient(response_text='{"score": 2, "reason": "Hmm."}')
        evaluate_answer_relevance_for_item("My specific question?", "A.", mock)
        assert "My specific question?" in mock.calls[0]["user"]

    def test_answer_in_prompt(self):
        mock = MockLLMClient(response_text='{"score": 2, "reason": "Hmm."}')
        evaluate_answer_relevance_for_item("Q?", "My specific answer.", mock)
        assert "My specific answer." in mock.calls[0]["user"]

    def test_empty_answer_handled(self):
        mock = MockLLMClient(response_text='{"score": 0, "reason": "Empty."}')
        result = evaluate_answer_relevance_for_item("Q?", "", mock)
        assert result.score == 0.0
        assert result.label == "fail"

    def test_fenced_response_parsed(self):
        fenced = '```json\n{"score": 3, "reason": "OK."}\n```'
        mock = MockLLMClient(response_text=fenced)
        result = evaluate_answer_relevance_for_item("Q?", "A.", mock)
        assert result.label == "pass"
        assert result.score == pytest.approx(0.75)

    def test_pass_label_on_score_3(self):
        mock = MockLLMClient(response_text='{"score": 3, "reason": "OK."}')
        assert evaluate_answer_relevance_for_item("Q?", "A.", mock).label == "pass"

    def test_fail_label_on_score_2(self):
        mock = MockLLMClient(response_text='{"score": 2, "reason": "Partial."}')
        assert evaluate_answer_relevance_for_item("Q?", "A.", mock).label == "fail"
