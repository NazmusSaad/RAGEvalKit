"""Tests for DummyEmbedder — no model downloads required."""
from rageval.core.embedder import DummyEmbedder, Embedder


class TestDummyEmbedderConstruction:
    def test_default_dim(self):
        assert DummyEmbedder().dim == 16

    def test_custom_dim(self):
        assert DummyEmbedder(dim=32).dim == 32

    def test_satisfies_embedder_protocol(self):
        assert isinstance(DummyEmbedder(), Embedder)


class TestDummyEmbedderEmbed:
    def test_returns_list_of_lists(self):
        e = DummyEmbedder(dim=8)
        result = e.embed(["hello"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_vector_length_matches_dim(self):
        for dim in (4, 16, 64):
            e = DummyEmbedder(dim=dim)
            assert len(e.embed(["test"])[0]) == dim

    def test_batch_size_matches_input(self):
        e = DummyEmbedder(dim=16)
        assert len(e.embed(["a", "b", "c"])) == 3

    def test_empty_batch_returns_empty(self):
        assert DummyEmbedder(dim=16).embed([]) == []

    def test_deterministic_same_text(self):
        e = DummyEmbedder(dim=16)
        assert e.embed(["hello"]) == e.embed(["hello"])

    def test_different_texts_produce_different_vectors(self):
        e = DummyEmbedder(dim=16)
        v1 = e.embed(["hello"])[0]
        v2 = e.embed(["world"])[0]
        assert v1 != v2

    def test_l2_normalised(self):
        e = DummyEmbedder(dim=16)
        for text in ("hello", "world", "foo bar baz"):
            v = e.embed([text])[0]
            norm = sum(x * x for x in v) ** 0.5
            assert abs(norm - 1.0) < 1e-6, f"norm={norm} for text={text!r}"

    def test_all_values_are_floats(self):
        e = DummyEmbedder(dim=16)
        v = e.embed(["test"])[0]
        assert all(isinstance(x, float) for x in v)

    def test_dim_property_matches_output_length(self):
        for dim in (4, 16, 64):
            e = DummyEmbedder(dim=dim)
            assert len(e.embed(["x"])[0]) == e.dim
