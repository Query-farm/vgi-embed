"""Tests for the embed scalar functions (compute() array-in / array-out).

Split into two tiers:

* **Pure logic** (no model): ``similarity``, ``embedding_dim``, and the
  NULL/empty masking in ``embed``. These always run.
* **Model-backed** (``@needs_model``): the actual embeddings. Gated on the
  default fastembed model being loadable, so a bare/offline checkout skips them
  cleanly while a provisioned environment runs them.

The model assertions are deliberately *structural / relative* -- exact vector
length, self-similarity ~ 1.0, and a planted related>unrelated comparison
(dog/puppy > dog/airplane) -- never exact float values, which vary by ONNX
build.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

from tests.harness import model_available
from vgi_embed import models
from vgi_embed.scalars import (
    Embed,
    EmbeddingDim,
    EmbedModel,
    EmbedPassage,
    EmbedQuery,
    Similarity,
)

needs_model = pytest.mark.skipif(
    not model_available(), reason="fastembed default model not available (offline / cold cache)"
)

_DIM = 384  # BAAI/bge-small-en-v1.5


# --- startup warm-up (best-effort, never fatal) -----------------------------


class TestWarmUp:
    def test_warm_up_is_idempotent_and_never_raises(self) -> None:
        # Called at worker spawn to move the model load/download off the first
        # query. Must be best-effort: safe to call repeatedly and never raise,
        # even with no model present (the function that needs it raises instead).
        models.warm_up()
        models.warm_up()


# --- similarity (pure arithmetic, always runs) ------------------------------


class TestSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        out = Similarity.compute(
            pa.array([v], type=pa.list_(pa.float32())), pa.array([v], type=pa.list_(pa.float32()))
        ).to_pylist()
        assert math.isclose(out[0], 1.0, abs_tol=1e-6)

    def test_orthogonal_vectors_score_zero(self) -> None:
        a = pa.array([[1.0, 0.0]], type=pa.list_(pa.float32()))
        b = pa.array([[0.0, 1.0]], type=pa.list_(pa.float32()))
        assert math.isclose(Similarity.compute(a, b).to_pylist()[0], 0.0, abs_tol=1e-6)

    def test_opposite_vectors_score_minus_one(self) -> None:
        a = pa.array([[1.0, 2.0]], type=pa.list_(pa.float32()))
        b = pa.array([[-1.0, -2.0]], type=pa.list_(pa.float32()))
        assert math.isclose(Similarity.compute(a, b).to_pylist()[0], -1.0, abs_tol=1e-6)

    def test_null_and_mismatch_and_zero_yield_null(self) -> None:
        lt = pa.list_(pa.float32())
        a = pa.array([None, [1.0, 2.0], [0.0, 0.0]], type=lt)
        b = pa.array([[1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0]], type=lt)
        out = Similarity.compute(a, b).to_pylist()
        # NULL input -> NULL; length mismatch -> NULL; zero magnitude -> NULL.
        assert out == [None, None, None]


# --- embedding_dim (no model) -----------------------------------------------


class TestMetadata:
    def test_embedding_dim_default_and_named(self) -> None:
        out = EmbeddingDim.compute(pa.array(["", "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"])).to_pylist()
        assert out == [384, 384, 768]

    def test_embedding_dim_unknown_model_is_null(self) -> None:
        # Unknown name -> NULL, not a crash.
        out = EmbeddingDim.compute(pa.array(["no/such-model", None])).to_pylist()
        assert out == [None, 384]  # None text falls back to the default model dim


# --- embed NULL/empty masking (no model needed for the NULL rows) -----------


class TestEmbedNullMasking:
    def test_all_null_and_empty_rows_are_null_without_loading_model(self) -> None:
        # Every row is NULL/empty/whitespace -> all-NULL output, and the model is
        # never invoked (so this runs even with no model installed).
        out = Embed.compute(pa.array([None, "", "   "])).to_pylist()
        assert out == [None, None, None]


# --- model-backed embeddings (gated) ----------------------------------------


@needs_model
class TestEmbed:
    def test_returns_fixed_length_float_vector(self) -> None:
        out = Embed.compute(pa.array(["hello"])).to_pylist()
        assert len(out) == 1
        assert len(out[0]) == _DIM
        assert all(isinstance(x, float) for x in out[0])

    def test_self_similarity_is_one(self) -> None:
        v = Embed.compute(pa.array(["a quick brown fox"]))
        sim = Similarity.compute(v, v).to_pylist()[0]
        assert math.isclose(sim, 1.0, abs_tol=1e-5)

    def test_related_pair_beats_unrelated_pair(self) -> None:
        # Planted: dog/puppy (related) should score higher than dog/airplane.
        vecs = Embed.compute(pa.array(["dog", "puppy", "airplane"]))
        dog, puppy, airplane = vecs[0], vecs[1], vecs[2]
        lt = pa.list_(pa.float32())
        related = Similarity.compute(pa.array([dog], type=lt), pa.array([puppy], type=lt)).to_pylist()[0]
        unrelated = Similarity.compute(pa.array([dog], type=lt), pa.array([airplane], type=lt)).to_pylist()[0]
        assert related > unrelated

    def test_null_and_empty_rows_interleave_with_real_ones(self) -> None:
        out = Embed.compute(pa.array(["hello", None, "", "world"])).to_pylist()
        assert len(out[0]) == _DIM
        assert out[1] is None
        assert out[2] is None
        assert len(out[3]) == _DIM

    def test_explicit_model_overload(self) -> None:
        out = EmbedModel.compute(pa.array(["hello"]), "BAAI/bge-small-en-v1.5").to_pylist()
        assert len(out[0]) == _DIM

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(models.ModelNotAvailableError, match="Unknown embedding model"):
            EmbedModel.compute(pa.array(["hello"]), "no/such-model").to_pylist()

    def test_long_text_does_not_crash(self) -> None:
        big = "the cat sat on the mat. " * 2000
        out = Embed.compute(pa.array([big])).to_pylist()
        assert len(out[0]) == _DIM


@needs_model
class TestEmbedQueryPassage:
    def test_query_and_passage_return_fixed_length(self) -> None:
        q = EmbedQuery.compute(pa.array(["how to reset password"])).to_pylist()
        p = EmbedPassage.compute(pa.array(["Click 'forgot password' to reset it."])).to_pylist()
        assert len(q[0]) == _DIM
        assert len(p[0]) == _DIM

    def test_query_prefix_changes_the_vector(self) -> None:
        # embed_query applies the bge instruction prefix; embed_passage does not,
        # so for the same text the two vectors must differ.
        text = pa.array(["reset password"])
        q = EmbedQuery.compute(text).to_pylist()[0]
        p = EmbedPassage.compute(text).to_pylist()[0]
        assert q != p

    def test_query_matches_relevant_passage_best(self) -> None:
        query = EmbedQuery.compute(pa.array(["how do I reset my password"]))
        passages = EmbedPassage.compute(
            pa.array(
                [
                    "To reset your password, click the 'forgot password' link.",
                    "Our office is open from nine to five on weekdays.",
                ]
            )
        )
        lt = pa.list_(pa.float32())
        relevant = Similarity.compute(query, pa.array([passages[0]], type=lt)).to_pylist()[0]
        irrelevant = Similarity.compute(query, pa.array([passages[1]], type=lt)).to_pylist()[0]
        assert relevant > irrelevant

    def test_null_empty_yield_null(self) -> None:
        assert EmbedQuery.compute(pa.array([None, ""])).to_pylist() == [None, None]
        assert EmbedPassage.compute(pa.array([None, "  "])).to_pylist() == [None, None]
