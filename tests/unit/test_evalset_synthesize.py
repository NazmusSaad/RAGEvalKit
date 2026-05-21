"""Unit tests for evalset synthesis — uses MockLLMClient, no real API calls."""
import json
from pathlib import Path

import pytest

from rageval.core.llm import CompletionResult, MockLLMClient
from rageval.evalset.synthesize import (
    EvalQuestion,
    EvalSetResult,
    _parse_questions,
    _strip_fences,
    generate_evalset_from_chunks,
    write_evalset_jsonl,
)

_EVALSET_ID = "test-evalset-001"
_CHUNK = {"chunk_id": "chunk001", "doc_id": "doc001", "ordinal": 0, "text": "RAG combines retrieval with generation."}


# ---------------------------------------------------------------------------
# Helpers — canned LLM responses
# ---------------------------------------------------------------------------

def _valid_json(n: int = 2) -> str:
    return json.dumps({
        "questions": [
            {
                "question": f"Question {i}?",
                "reference_answer": f"Answer {i}.",
                "qtype": "factoid",
                "difficulty": "easy",
            }
            for i in range(n)
        ]
    })


def _fenced_json(n: int = 1) -> str:
    return f"```json\n{_valid_json(n)}\n```"


def _invalid_json() -> str:
    return "This is definitely not JSON."


def _partial_items() -> str:
    """One good item, two malformed items."""
    return json.dumps({
        "questions": [
            {"question": "Good Q?", "reference_answer": "Good A.", "qtype": "factoid", "difficulty": "easy"},
            {"reference_answer": "No question field here"},
            {"question": "", "reference_answer": "Empty question string"},
        ]
    })


# ---------------------------------------------------------------------------
# _strip_fences
# ---------------------------------------------------------------------------

class TestStripFences:
    def test_no_fences_unchanged(self):
        assert _strip_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fences_stripped(self):
        assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_plain_fences_stripped(self):
        assert _strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_leading_trailing_whitespace_stripped(self):
        assert _strip_fences('  {"a": 1}  ') == '{"a": 1}'


# ---------------------------------------------------------------------------
# _parse_questions
# ---------------------------------------------------------------------------

class TestParseQuestions:
    def test_valid_json_returns_questions(self):
        qs, ok = _parse_questions(_valid_json(2), "c1", _EVALSET_ID)
        assert ok is True
        assert len(qs) == 2

    def test_fenced_json_is_parsed(self):
        qs, ok = _parse_questions(_fenced_json(1), "c1", _EVALSET_ID)
        assert ok is True
        assert len(qs) == 1

    def test_invalid_json_returns_failure(self):
        qs, ok = _parse_questions(_invalid_json(), "c1", _EVALSET_ID)
        assert ok is False
        assert qs == []

    def test_partial_items_keeps_only_valid(self):
        qs, ok = _parse_questions(_partial_items(), "c1", _EVALSET_ID)
        assert ok is True
        assert len(qs) == 1

    def test_source_chunk_ids_set(self):
        qs, _ = _parse_questions(_valid_json(1), "mychunk", _EVALSET_ID)
        assert qs[0].source_chunk_ids == ["mychunk"]

    def test_evalset_id_set(self):
        qs, _ = _parse_questions(_valid_json(1), "c1", _EVALSET_ID)
        assert qs[0].evalset_id == _EVALSET_ID

    def test_question_fields_populated(self):
        qs, _ = _parse_questions(_valid_json(1), "c1", _EVALSET_ID)
        q = qs[0]
        assert q.question == "Question 0?"
        assert q.reference_answer == "Answer 0."
        assert q.qtype == "factoid"
        assert q.difficulty == "easy"

    def test_missing_qtype_defaults_to_factoid(self):
        raw = json.dumps({"questions": [{"question": "Q?", "reference_answer": "A."}]})
        qs, _ = _parse_questions(raw, "c1", _EVALSET_ID)
        assert qs[0].qtype == "factoid"

    def test_missing_difficulty_defaults_to_medium(self):
        raw = json.dumps({"questions": [{"question": "Q?", "reference_answer": "A."}]})
        qs, _ = _parse_questions(raw, "c1", _EVALSET_ID)
        assert qs[0].difficulty == "medium"


# ---------------------------------------------------------------------------
# generate_evalset_from_chunks
# ---------------------------------------------------------------------------

class TestGenerateEvalsetFromChunks:
    def test_empty_chunks_returns_empty_result(self):
        mock = MockLLMClient(response_text=_valid_json())
        result = generate_evalset_from_chunks([], mock, _EVALSET_ID, num_questions=5)
        assert result.questions == []
        assert result.source_chunks_used == 0
        assert result.parse_failures == 0

    def test_returns_eval_set_result(self):
        mock = MockLLMClient(response_text=_valid_json())
        result = generate_evalset_from_chunks([_CHUNK], mock, _EVALSET_ID)
        assert isinstance(result, EvalSetResult)

    def test_questions_generated(self):
        mock = MockLLMClient(response_text=_valid_json(2))
        result = generate_evalset_from_chunks([_CHUNK], mock, _EVALSET_ID, num_questions=2)
        assert len(result.questions) == 2

    def test_caps_at_num_questions(self):
        mock = MockLLMClient(response_text=_valid_json(10))
        result = generate_evalset_from_chunks([_CHUNK], mock, _EVALSET_ID, num_questions=3)
        assert len(result.questions) <= 3

    def test_question_ids_are_unique(self):
        mock = MockLLMClient(response_text=_valid_json(3))
        result = generate_evalset_from_chunks([_CHUNK], mock, _EVALSET_ID, num_questions=3)
        ids = [q.question_id for q in result.questions]
        assert len(ids) == len(set(ids))

    def test_parse_failure_counted(self):
        mock = MockLLMClient(response_text=_invalid_json())
        result = generate_evalset_from_chunks([_CHUNK], mock, _EVALSET_ID, num_questions=5)
        assert result.parse_failures == 1

    def test_parse_failure_does_not_crash(self):
        mock = MockLLMClient(response_text=_invalid_json())
        result = generate_evalset_from_chunks([_CHUNK], mock, _EVALSET_ID, num_questions=5)
        assert result.questions == []

    def test_fenced_json_works(self):
        mock = MockLLMClient(response_text=_fenced_json(1))
        result = generate_evalset_from_chunks([_CHUNK], mock, _EVALSET_ID, num_questions=1)
        assert len(result.questions) == 1

    def test_source_chunks_used(self):
        chunks = [_CHUNK, {"chunk_id": "c2", "doc_id": "d1", "ordinal": 1, "text": "More text."}]
        mock = MockLLMClient(response_text=_valid_json(1))
        result = generate_evalset_from_chunks(chunks, mock, _EVALSET_ID, num_questions=2)
        assert result.source_chunks_used == 2

    def test_model_propagated(self):
        mock = MockLLMClient(response_text=_valid_json())
        result = generate_evalset_from_chunks([_CHUNK], mock, _EVALSET_ID, model="test-model")
        assert result.model == "test-model"

    def test_partial_failure_mixed_chunks(self):
        """One chunk fails, the other succeeds — failures counted, good questions kept."""
        class _AlternatingMock:
            _n = 0

            def complete(self, system: str, user: str, **_) -> CompletionResult:
                self._n += 1
                text = _invalid_json() if self._n == 1 else _valid_json(1)
                return CompletionResult(
                    text=text, prompt_tokens=10, completion_tokens=5,
                    total_tokens=15, cost_usd=None, model="mock",
                )

        chunks = [
            {"chunk_id": "bad_chunk", "doc_id": "d", "ordinal": 0, "text": "text1"},
            {"chunk_id": "good_chunk", "doc_id": "d", "ordinal": 1, "text": "text2"},
        ]
        result = generate_evalset_from_chunks(chunks, _AlternatingMock(), _EVALSET_ID, num_questions=5)
        assert result.parse_failures == 1
        assert len(result.questions) >= 1


# ---------------------------------------------------------------------------
# write_evalset_jsonl
# ---------------------------------------------------------------------------

def _make_question(idx: int = 0) -> EvalQuestion:
    return EvalQuestion(
        question_id=f"q{idx}",
        evalset_id="e1",
        question=f"Question {idx}?",
        reference_answer=f"Answer {idx}.",
        source_chunk_ids=[f"c{idx}"],
        difficulty="easy",
        qtype="factoid",
    )


class TestWriteEvalsetJsonl:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "out" / "test.jsonl"
        write_evalset_jsonl([_make_question()], path)
        assert path.exists()

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "test.jsonl"
        write_evalset_jsonl([_make_question()], path)
        assert path.exists()

    def test_one_line_per_question(self, tmp_path):
        path = tmp_path / "test.jsonl"
        write_evalset_jsonl([_make_question(i) for i in range(3)], path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, tmp_path):
        path = tmp_path / "test.jsonl"
        write_evalset_jsonl([_make_question(i) for i in range(2)], path)
        for line in path.read_text().splitlines():
            if line.strip():
                json.loads(line)  # must not raise

    def test_required_fields_present(self, tmp_path):
        path = tmp_path / "test.jsonl"
        write_evalset_jsonl([_make_question()], path)
        data = json.loads(path.read_text().strip())
        for field in ("question_id", "question", "reference_answer", "source_chunk_ids", "difficulty", "qtype"):
            assert field in data

    def test_source_chunk_ids_is_list(self, tmp_path):
        path = tmp_path / "test.jsonl"
        write_evalset_jsonl([_make_question()], path)
        data = json.loads(path.read_text().strip())
        assert isinstance(data["source_chunk_ids"], list)

    def test_empty_list_creates_empty_file(self, tmp_path):
        path = tmp_path / "test.jsonl"
        write_evalset_jsonl([], path)
        assert path.exists()
        assert path.read_text() == ""
