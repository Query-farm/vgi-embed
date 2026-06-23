"""Per-row embedding + similarity as DuckDB scalar functions.

Each function maps one (or two) input value(s) to one output value, so it drops
straight into a ``SELECT`` list or a join/order-by:

    SELECT embed(body) AS v FROM docs;                       -- 384-dim FLOAT[]
    SELECT id FROM docs ORDER BY embed.similarity(embed(body), embed(:q)) DESC;

The embedding functions are model-backed; the model is loaded once per worker
process and cached (see :mod:`vgi_embed.models`). ``similarity`` is pure
arithmetic -- no model.

A note on argument syntax
-------------------------
VGI / DuckDB *scalar* functions take **positional** arguments and resolve
overloads by *arity* -- the ``name := value`` named-argument syntax is a property
of table functions and macros, not scalars. So ``embed`` exposes its optional
``model`` argument as a second arity overload sharing the one name:

    SELECT embed.embed(body)                          FROM docs;  -- default model
    SELECT embed.embed(body, 'BAAI/bge-base-en-v1.5') FROM docs;  -- pick a model

Returns
-------
The embedding functions return ``FLOAT[]`` (``LIST(FLOAT)``), so they declare an
explicit ``Returns(arrow_type=pa.list_(pa.float32()))``. ``similarity`` returns
``DOUBLE``.

NULL semantics: NULL or empty/whitespace-only text yields a NULL vector; a NULL
or malformed vector pair yields a NULL similarity. Nothing here crashes on odd
input.
"""

from __future__ import annotations

from typing import Annotated

import pyarrow as pa
from vgi import Param, Returns, ScalarFunction
from vgi.arguments import ConstParam
from vgi.metadata import FunctionExample

from . import models

# Arrow return type shared by every embedding scalar: a list of float32.
_VECTOR = pa.list_(pa.float32())


def _ex(sql: str, description: str) -> list[FunctionExample]:
    return [FunctionExample(sql=sql, description=description)]


def _embed_array(
    text: pa.StringArray,
    *,
    model: str | None,
    prefix: str = "",
) -> pa.ListArray:
    """Embed a string array to a ``list<float32>`` array, NULL/empty -> NULL.

    NULL and empty/whitespace-only rows map to a NULL vector. The non-empty rows
    are embedded in one batched ``fastembed`` call (order preserved), then spliced
    back into the original row order.
    """
    values = text.to_pylist()
    live_idx: list[int] = []
    live_text: list[str] = []
    for i, t in enumerate(values):
        if t is not None and t.strip():
            live_idx.append(i)
            live_text.append(t)

    vectors: list[list[float] | None] = [None] * len(values)
    if live_text:
        # For the query overloads the prefix may be model-specific; resolve it
        # from the model when not supplied explicitly.
        embedded = models.embed_texts(live_text, model=model, prefix=prefix)
        for i, vec in zip(live_idx, embedded, strict=False):
            vectors[i] = vec
    return pa.array(vectors, type=_VECTOR)


# ===========================================================================
# embed(text) / embed(text, model) -- symmetric sentence embedding (no prefix)
# ===========================================================================


class Embed(ScalarFunction):
    """``embed(text)`` -- sentence embedding with the default model (no prefix)."""

    class Meta:
        name = "embed"
        description = (
            "Embed text into a fixed-length FLOAT[] vector using the default model "
            f"({models.DEFAULT_MODEL}, {models.embedding_dim(None)}-dim). NULL/empty -> NULL."
        )
        categories = ["embedding"]
        examples = _ex(
            "SELECT embed.embed('hello world')",
            "Embed a string with the default model",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to embed")],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_VECTOR)]:
        return _embed_array(text, model=None)


class EmbedModel(ScalarFunction):
    """``embed(text, model)`` -- sentence embedding with an explicit model."""

    class Meta:
        name = "embed"
        description = (
            "Embed text into a FLOAT[] vector with an explicit model (see supported_models()). NULL/empty -> NULL."
        )
        categories = ["embedding"]
        examples = _ex(
            "SELECT embed.embed('hello world', 'BAAI/bge-base-en-v1.5')",
            "Embed a string with a chosen model",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to embed")],
        model: Annotated[str, ConstParam(doc="Model name; see supported_models()")],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_VECTOR)]:
        return _embed_array(text, model=model or None)


# ===========================================================================
# embed_query / embed_passage -- retrieval asymmetry (query gets the prefix)
# ===========================================================================


class EmbedQuery(ScalarFunction):
    """``embed_query(text)`` -- embed a *search query* (applies the model's instruction prefix)."""

    class Meta:
        name = "embed_query"
        description = (
            "Embed a search query with the default model, applying the model's "
            "recommended query instruction prefix (for bge: 'Represent this "
            "sentence for searching relevant passages: '). NULL/empty -> NULL."
        )
        categories = ["embedding", "retrieval"]
        examples = _ex(
            "SELECT embed.embed_query('how do I reset my password')",
            "Embed a retrieval query (query-side prefix applied)",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Search query to embed")],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_VECTOR)]:
        return _embed_array(text, model=None, prefix=models.query_prefix(None))


class EmbedPassage(ScalarFunction):
    """``embed_passage(text)`` -- embed a *document/passage* (no prefix, by design)."""

    class Meta:
        name = "embed_passage"
        description = (
            "Embed a document/passage with the default model. For bge retrieval "
            "models passages get NO instruction prefix (queries do); this mirrors "
            "that. NULL/empty -> NULL."
        )
        categories = ["embedding", "retrieval"]
        examples = _ex(
            "SELECT embed.embed_passage(body) FROM documents",
            "Embed corpus passages (no query prefix)",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Passage/document text to embed")],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_VECTOR)]:
        return _embed_array(text, model=None)


# ===========================================================================
# similarity(a, b) -- pure cosine similarity, no model
# ===========================================================================


class Similarity(ScalarFunction):
    """``similarity(a, b)`` -- cosine similarity of two FLOAT[] vectors, in [-1, 1]."""

    class Meta:
        name = "similarity"
        description = (
            "Cosine similarity of two FLOAT[] vectors, in [-1, 1] (pure arithmetic, "
            "no model). NULL/empty/length-mismatch -> NULL."
        )
        categories = ["similarity"]
        examples = _ex(
            "SELECT embed.similarity(embed.embed('cat'), embed.embed('kitten'))",
            "Cosine similarity between two embeddings",
        )

    @classmethod
    def compute(
        cls,
        a: Annotated[pa.ListArray, Param(arrow_type=_VECTOR, doc="First FLOAT[] vector")],
        b: Annotated[pa.ListArray, Param(arrow_type=_VECTOR, doc="Second FLOAT[] vector")],
    ) -> Annotated[pa.DoubleArray, Returns(arrow_type=pa.float64())]:
        out = [models.cosine_similarity(x, y) for x, y in zip(a.to_pylist(), b.to_pylist(), strict=False)]
        return pa.array(out, type=pa.float64())


# ===========================================================================
# embedding_dim(model) / embed_version() -- metadata helpers
# ===========================================================================


class EmbeddingDim(ScalarFunction):
    """``embedding_dim(model)`` -- the output dimension for a model name."""

    class Meta:
        name = "embedding_dim"
        description = "Output dimension (vector length) for a model name; '' = default model"
        categories = ["embedding", "metadata"]
        examples = _ex(
            "SELECT embed.embedding_dim('BAAI/bge-small-en-v1.5')",
            "Dimension of the default model (384)",
        )

    @classmethod
    def compute(
        cls,
        model: Annotated[pa.StringArray, Param(doc="Model name; '' or NULL = default model")],
    ) -> Annotated[pa.Int32Array, Returns(arrow_type=pa.int32())]:
        out: list[int | None] = []
        for m in model.to_pylist():
            try:
                out.append(models.embedding_dim(m))
            except models.ModelNotAvailableError:
                # Unknown model name -> NULL rather than crashing the whole query.
                out.append(None)
        return pa.array(out, type=pa.int32())


class EmbedVersion(ScalarFunction):
    """``embed_version()`` -- worker + default-model identity string."""

    class Meta:
        name = "embed_version"
        description = "Version string: worker version, fastembed backend, and default model"
        categories = ["metadata"]
        examples = _ex(
            "SELECT embed.embed_version()",
            "Identify the embed worker / default model",
        )

    @classmethod
    def compute(
        cls,
    ) -> Annotated[pa.StringArray, Returns(arrow_type=pa.string())]:
        return pa.array([_version_string()], type=pa.string())


def _version_string() -> str:
    """Build the embed_version() string (worker + fastembed backend + default model)."""
    from . import __version__

    try:
        from importlib.metadata import version as _pkg_version

        backend = f"fastembed {_pkg_version('fastembed')}"
    except Exception:  # noqa: BLE001 - version lookup is best-effort
        backend = "fastembed"
    return f"vgi-embed {__version__} ({backend}; default {models.DEFAULT_MODEL})"


SCALAR_FUNCTIONS: list[type] = [
    Embed,
    EmbedModel,
    EmbedQuery,
    EmbedPassage,
    Similarity,
    EmbeddingDim,
    EmbedVersion,
]
