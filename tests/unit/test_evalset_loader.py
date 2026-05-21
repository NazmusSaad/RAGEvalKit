"""Unit tests for evalset JSONL loader."""
import json

import pytest

from rageval.evalset.loader import load_evalset_from_jsonl
from rageval.evalset.synthesize import EvalQuestion, write_evalset_jsonl

_VALID_Q = {
    "question_id": "q1",
    "question": "What is RAG?",
    "reference_answer": "RAG combines retrieval with generation.",
    "source_chunk_ids": ["c1"],
    "difficulty": "easy",
    "qtype": "factoid",
}


def _write(tmp_path, rows: list[dict], filename: str = "test.jsonl"):
    p = tmp_path / filename
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


class TestLoadEvalsetFromJsonl:
    def test_loads_valid_file(self, tmp_path):
        p = _write(tmp_path, [_VALID_Q])
        qs = load_evalset_from_jsonl(p)
        assert len(qs) == 1

    def test_question_fields(self, tmp_path):
        p = _write(tmp_path, [_VALID_Q])
        q = load_evalset_from_jsonl(p)[0]
        assert q.question_id == "q1"
        assert q.question == "What is RAG?"
        assert q.reference_answer == "RAG combines retrieval with generation."
        assert q.source_chunk_ids == ["c1"]
        assert q.difficulty == "easy"
        assert q.qtype == "factoid"

    def test_returns_eval_question_objects(self, tmp_path):
        p = _write(tmp_path, [_VALID_Q])
        q = load_evalset_from_jsonl(p)[0]
        assert isinstance(q, EvalQuestion)

    def test_multiple_questions(self, tmp_path):
        rows = [{**_VALID_Q, "question_id": f"q{i}"} for i in range(5)]
        p = _write(tmp_path, rows)
        assert len(load_evalset_from_jsonl(p)) == 5

    def test_empty_file_returns_empty_list(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert load_evalset_from_jsonl(p) == []

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "blanks.jsonl"
        p.write_text("\n" + json.dumps(_VALID_Q) + "\n\n")
        assert len(load_evalset_from_jsonl(p)) == 1

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_evalset_from_jsonl(tmp_path / "missing.jsonl")

    def test_invalid_json_raises_value_error(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_evalset_from_jsonl(p)

    def test_invalid_json_mentions_line_number(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps(_VALID_Q) + "\nnot json\n")
        with pytest.raises(ValueError, match="Line 2"):
            load_evalset_from_jsonl(p)

    def test_missing_required_field_raises(self, tmp_path):
        bad = {k: v for k, v in _VALID_Q.items() if k != "question"}
        p = _write(tmp_path, [bad])
        with pytest.raises(ValueError, match="missing required fields"):
            load_evalset_from_jsonl(p)

    def test_missing_field_mentions_field_name(self, tmp_path):
        bad = {k: v for k, v in _VALID_Q.items() if k != "reference_answer"}
        p = _write(tmp_path, [bad])
        with pytest.raises(ValueError, match="reference_answer"):
            load_evalset_from_jsonl(p)

    def test_roundtrip_with_write_evalset_jsonl(self, tmp_path):
        original = [
            EvalQuestion(
                question_id=f"q{i}", evalset_id="e1",
                question=f"Q{i}?", reference_answer=f"A{i}.",
                source_chunk_ids=[f"c{i}"], difficulty="easy", qtype="factoid",
            )
            for i in range(3)
        ]
        path = tmp_path / "round.jsonl"
        write_evalset_jsonl(original, path)
        loaded = load_evalset_from_jsonl(path)
        assert len(loaded) == 3
        assert loaded[0].question == "Q0?"
        assert loaded[2].question_id == "q2"

    def test_optional_evalset_id_defaults_to_empty(self, tmp_path):
        p = _write(tmp_path, [_VALID_Q])  # no evalset_id in data
        q = load_evalset_from_jsonl(p)[0]
        assert q.evalset_id == ""
