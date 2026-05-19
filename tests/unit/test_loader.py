from pathlib import Path

import pytest

from rageval.core.loader import Document, load_document, load_documents


class TestLoadDocument:
    def test_returns_document(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Hello\nWorld", encoding="utf-8")
        doc = load_document(f)
        assert isinstance(doc, Document)

    def test_text_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        doc = load_document(f)
        assert doc.text == "hello world"

    def test_source_path(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("x", encoding="utf-8")
        doc = load_document(f)
        assert doc.source_path == f

    def test_num_chars(self, tmp_path):
        content = "abc"
        f = tmp_path / "test.txt"
        f.write_text(content, encoding="utf-8")
        doc = load_document(f)
        assert doc.num_chars == 3

    def test_doc_id_is_stable(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"stable content")
        assert load_document(f).doc_id == load_document(f).doc_id

    def test_same_content_same_doc_id(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"identical")
        f2.write_bytes(b"identical")
        assert load_document(f1).doc_id == load_document(f2).doc_id

    def test_different_content_different_doc_id(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert load_document(f1).doc_id != load_document(f2).doc_id

    def test_invalid_utf8_handled(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_bytes(b"hello \xff world")
        doc = load_document(f)
        assert "hello" in doc.text


class TestLoadDocuments:
    def test_single_md_file(self, tmp_path):
        f = tmp_path / "a.md"
        f.write_text("markdown content", encoding="utf-8")
        docs = load_documents(f)
        assert len(docs) == 1
        assert docs[0].text == "markdown content"

    def test_single_txt_file(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("text content", encoding="utf-8")
        docs = load_documents(f)
        assert len(docs) == 1

    def test_unsupported_single_file_returns_empty(self, tmp_path):
        f = tmp_path / "file.pdf"
        f.write_bytes(b"%PDF")
        docs = load_documents(f)
        assert docs == []

    def test_directory_with_mixed_files(self, tmp_path):
        (tmp_path / "a.md").write_text("one", encoding="utf-8")
        (tmp_path / "b.txt").write_text("two", encoding="utf-8")
        (tmp_path / "c.pdf").write_bytes(b"%PDF")
        (tmp_path / "d.py").write_text("pass", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert len(docs) == 2

    def test_recursive_directory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.md").write_text("top", encoding="utf-8")
        (sub / "nested.txt").write_text("nested", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert len(docs) == 2

    def test_empty_directory(self, tmp_path):
        assert load_documents(tmp_path) == []

    def test_deeply_nested(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "file.md").write_text("deep", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert len(docs) == 1
        assert docs[0].text == "deep"

    def test_tiny_corpus_fixture(self):
        corpus = Path(__file__).parents[2] / "examples" / "tiny-corpus"
        docs = load_documents(corpus)
        assert len(docs) == 3
        texts = {d.text for d in docs}
        assert any("RAG" in t for t in texts)
