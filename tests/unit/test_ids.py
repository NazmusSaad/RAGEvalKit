from rageval.core.ids import make_chunk_id, make_doc_id, sha256_bytes, sha256_text


class TestSha256Text:
    def test_deterministic(self):
        assert sha256_text("hello") == sha256_text("hello")

    def test_different_inputs_differ(self):
        assert sha256_text("hello") != sha256_text("world")

    def test_returns_64_char_hex(self):
        result = sha256_text("hello")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string(self):
        result = sha256_text("")
        assert len(result) == 64


class TestSha256Bytes:
    def test_deterministic(self):
        assert sha256_bytes(b"hello") == sha256_bytes(b"hello")

    def test_different_inputs_differ(self):
        assert sha256_bytes(b"hello") != sha256_bytes(b"world")

    def test_returns_64_char_hex(self):
        result = sha256_bytes(b"data")
        assert len(result) == 64

    def test_text_and_bytes_agree_for_ascii(self):
        assert sha256_text("hello") == sha256_bytes(b"hello")


class TestMakeDocId:
    def test_deterministic(self):
        content = b"some document content"
        assert make_doc_id(content) == make_doc_id(content)

    def test_same_content_same_id(self):
        assert make_doc_id(b"hello") == make_doc_id(b"hello")

    def test_different_content_different_id(self):
        assert make_doc_id(b"hello") != make_doc_id(b"world")

    def test_returns_hex_string(self):
        result = make_doc_id(b"x")
        assert len(result) == 64


class TestMakeChunkId:
    def test_deterministic(self):
        assert make_chunk_id("doc", 0, "text") == make_chunk_id("doc", 0, "text")

    def test_ordinal_changes_id(self):
        assert make_chunk_id("doc", 0, "text") != make_chunk_id("doc", 1, "text")

    def test_text_changes_id(self):
        assert make_chunk_id("doc", 0, "A") != make_chunk_id("doc", 0, "B")

    def test_doc_id_changes_id(self):
        assert make_chunk_id("doc1", 0, "text") != make_chunk_id("doc2", 0, "text")

    def test_returns_hex_string(self):
        result = make_chunk_id("d", 0, "t")
        assert len(result) == 64
