import pytest

from rageval.core.chunker import Chunk, SimpleChunker
from rageval.core.loader import Document


def _doc(text: str, doc_id: str = "testdoc") -> Document:
    return Document(doc_id=doc_id, source_path=None, text=text, num_chars=len(text))


class TestSimpleChunkerConstruction:
    def test_defaults(self):
        c = SimpleChunker()
        assert c.chunk_size == 512
        assert c.chunk_overlap == 64

    def test_custom_params(self):
        c = SimpleChunker(chunk_size=200, chunk_overlap=30)
        assert c.chunk_size == 200
        assert c.chunk_overlap == 30

    def test_overlap_equal_to_size_raises(self):
        with pytest.raises(ValueError):
            SimpleChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError):
            SimpleChunker(chunk_size=100, chunk_overlap=200)

    def test_zero_overlap_is_valid(self):
        c = SimpleChunker(chunk_size=100, chunk_overlap=0)
        assert c.chunk_overlap == 0


class TestChunkDocument:
    def test_small_document_one_chunk(self):
        chunker = SimpleChunker(chunk_size=512, chunk_overlap=64)
        doc = _doc("Short document.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Short document."

    def test_large_document_multiple_chunks(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)
        doc = _doc("x" * 300)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_empty_document_no_chunks(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_document(_doc(""))
        assert chunks == []

    def test_whitespace_only_no_chunks(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_document(_doc("   \n\t  "))
        assert chunks == []

    def test_deterministic(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)
        doc = _doc("a" * 250)
        ids1 = [c.chunk_id for c in chunker.chunk_document(doc)]
        ids2 = [c.chunk_id for c in chunker.chunk_document(doc)]
        assert ids1 == ids2

    def test_ordinals_are_sequential(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)
        doc = _doc("y" * 300)
        chunks = chunker.chunk_document(doc)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_doc_id_propagated(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)
        doc = _doc("z" * 250, doc_id="myid")
        chunks = chunker.chunk_document(doc)
        assert all(c.doc_id == "myid" for c in chunks)

    def test_num_chars_matches_text(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)
        doc = _doc("a" * 250)
        chunks = chunker.chunk_document(doc)
        assert all(c.num_chars == len(c.text) for c in chunks)

    def test_chunks_cover_document_with_no_overlap(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=0)
        text = "a" * 300
        chunks = chunker.chunk_document(_doc(text))
        assert "".join(c.text for c in chunks) == text

    def test_different_docs_produce_different_chunk_ids(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)
        text = "same content " * 20
        doc1 = _doc(text, doc_id="docA")
        doc2 = _doc(text, doc_id="docB")
        ids1 = {c.chunk_id for c in chunker.chunk_document(doc1)}
        ids2 = {c.chunk_id for c in chunker.chunk_document(doc2)}
        assert ids1.isdisjoint(ids2)

    def test_chunk_ids_unique_within_document(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)
        doc = _doc("z" * 400)
        chunks = chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_returns_chunk_dataclass(self):
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=10)
        doc = _doc("hello world")
        chunks = chunker.chunk_document(doc)
        assert isinstance(chunks[0], Chunk)

    def test_tiny_corpus_each_file_one_chunk(self):
        from pathlib import Path
        from rageval.core.loader import load_documents

        corpus = Path(__file__).parents[2] / "examples" / "tiny-corpus"
        docs = load_documents(corpus)
        chunker = SimpleChunker(chunk_size=512, chunk_overlap=64)
        for doc in docs:
            chunks = chunker.chunk_document(doc)
            assert len(chunks) == 1, f"{doc.source_path} produced {len(chunks)} chunks"
