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

Returns:
-------
The embedding functions return ``FLOAT[]`` (``LIST(FLOAT)``), so they declare an
explicit ``Returns(arrow_type=pa.list_(pa.float32()))``. ``similarity`` returns
``DOUBLE``.

NULL semantics: NULL or empty/whitespace-only text yields a NULL vector; a NULL
or malformed vector pair yields a NULL similarity. Nothing here crashes on odd
input.
"""

from __future__ import annotations

import json
from typing import Annotated

import pyarrow as pa
from vgi import Param, Returns, ScalarFunction
from vgi.arguments import ConstParam
from vgi.metadata import FunctionExample

from . import meta, models

# Arrow return type shared by every embedding scalar: a list of float32.
_VECTOR = pa.list_(pa.float32())


# ---------------------------------------------------------------------------
# Examples (VGI515). The native ``examples=`` carrier (duckdb_functions().examples)
# preserves only each example's SQL and drops its description, so every function
# ALSO emits a byte-identical ``vgi.example_queries`` JSON list that carries the
# ``{"description", "sql"}`` pairs. We build both from one list of (sql,
# description) tuples so they can never drift. For the two ``embed`` arity
# overloads the examples are aggregated by function name (both overloads carry the
# same set), which is how the linter groups them.
# ---------------------------------------------------------------------------


def _fx(pairs: list[tuple[str, str]]) -> list[FunctionExample]:
    """Native ``FunctionExample`` list from ``(sql, description)`` pairs."""
    return [FunctionExample(sql=sql, description=description) for sql, description in pairs]


def _eq(pairs: list[tuple[str, str]]) -> str:
    """``vgi.example_queries`` described-JSON string from ``(sql, description)`` pairs."""
    return json.dumps([{"description": description, "sql": sql} for sql, description in pairs])


# Per-function example sets (each `sql` is self-contained and re-runnable against
# an attached `embed` worker; exact float values vary by ONNX build so the
# examples project a stable shape -- a length, a rounded score, or a boolean --
# rather than a raw vector).
_EMBED_EXAMPLES = [
    (
        "SELECT len(embed.main.embed('hello world')) AS dim",
        "Embed a string into a 384-dim `FLOAT[]` with the default model.",
    ),
    (
        "SELECT len(embed.main.embed('hello world', 'BAAI/bge-base-en-v1.5')) AS dim",
        "Embed text with an explicitly chosen model (768-dim `bge-base`).",
    ),
]
_EMBED_QUERY_EXAMPLES = [
    (
        "SELECT len(embed.main.embed_query('how do I reset my password')) AS dim",
        "Embed a search query with the query-side instruction prefix.",
    ),
]
_EMBED_PASSAGE_EXAMPLES = [
    (
        "SELECT len(embed.main.embed_passage('Reset your password from the account settings page.')) AS dim",
        "Embed a corpus passage (no query prefix) into a 384-dim `FLOAT[]`.",
    ),
]
_SIMILARITY_EXAMPLES = [
    (
        "SELECT ROUND(embed.main.similarity(embed.main.embed('dog'), embed.main.embed('puppy')), 3) AS related",
        "Score how similar two embeddings are (rounded for stable output).",
    ),
    (
        "SELECT embed.main.similarity(embed.main.embed('dog'), embed.main.embed('puppy')) "
        "> embed.main.similarity(embed.main.embed('dog'), embed.main.embed('airplane')) AS related_wins",
        "Confirm a related pair ranks above an unrelated pair.",
    ),
]
_EMBEDDING_DIM_EXAMPLES = [
    (
        "SELECT embed.main.embedding_dim('BAAI/bge-base-en-v1.5') AS dim",
        "Look up a model's output dimension without loading the model.",
    ),
]

# VGI509: at least one guaranteed-runnable, VERIFIED example (with expected_result)
# so an agent has a reference it can execute and check. Both assertions are
# deterministic across ONNX builds: a model's dimension is fixed, and the default
# model always yields a 384-length vector -- neither depends on exact float values.
_VERIFIED_EXAMPLES = json.dumps(
    [
        {
            "description": "The default model produces 384-dimensional vectors.",
            "sql": "SELECT embed.main.embedding_dim('BAAI/bge-small-en-v1.5') AS dim",
            "expected_result": [{"dim": 384}],
        },
        {
            "description": "embed() returns a 384-length FLOAT[] for the default model.",
            "sql": "SELECT len(embed.main.embed('hello world')) AS dim",
            "expected_result": [{"dim": 384}],
        },
    ]
)


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
        """Declarative metadata for the ``embed(text)`` scalar."""

        name = "embed"
        description = (
            "Embed text into a fixed-length FLOAT[] vector using the default model "
            f"({models.DEFAULT_MODEL}, {models.embedding_dim(None)}-dim). NULL/empty -> NULL."
        )
        categories = ["embedding"]
        examples = _fx(_EMBED_EXAMPLES)
        tags = {
            **meta.object_tags(
                title="Embed Text To Vector",
                description_llm=(
                    "## embed(text)\n\n"
                    "Turn a column (or literal) of text into a fixed-length `FLOAT[]` "
                    f"embedding vector using the default model (`{models.DEFAULT_MODEL}`, "
                    f"{models.embedding_dim(None)}-dim). Runs entirely in-process via "
                    "fastembed/ONNX -- no torch, no network call after the one-time model "
                    "download.\n\n"
                    "**When to use.** Use this for *symmetric* embeddings where the two "
                    "sides being compared are the same kind of text (e.g. sentence-to-"
                    "sentence dedup, clustering, classification). For retrieval where a "
                    "short query is matched against longer documents, prefer "
                    "`embed_query` / `embed_passage`.\n\n"
                    "**Input/output.** Input: one `VARCHAR` per row. Output: one "
                    "`FLOAT[]` (`LIST(FLOAT)`) per row. NULL or empty/whitespace-only "
                    "text yields a NULL vector. Score two vectors with `similarity(a, b)`.\n\n"
                    "**Edge cases.** An all-empty batch never loads the model; vectors "
                    "are L2-comparable via cosine similarity."
                ),
                description_md=(
                    "# embed(text)\n\n"
                    "Embed text into a fixed-length `FLOAT[]` vector with the default "
                    f"model (`{models.DEFAULT_MODEL}`, {models.embedding_dim(None)}-dim).\n\n"
                    "## Usage\n\n"
                    "Apply `embed(text)` to a literal or a VARCHAR column to get one "
                    f"{models.embedding_dim(None)}-element `FLOAT[]` per row, then order rows "
                    "by `similarity(...)` against an embedded query vector to rank them. "
                    "Runnable, catalog-qualified examples are attached as example queries.\n\n"
                    "## Notes\n\n"
                    "- Best for symmetric text-to-text comparison; for query/passage "
                    "retrieval use `embed_query` / `embed_passage`.\n"
                    "- NULL or empty input returns NULL.\n"
                    "- Runs locally (fastembed/ONNX); pairs with DuckDB VSS."
                ),
                keywords=[
                    "embed",
                    "embedding",
                    "vector",
                    "fastembed",
                    "onnx",
                    "sentence embedding",
                    "semantic",
                    "text to vector",
                    "float array",
                    "similarity",
                    "rag",
                ],
                category="embedding",
            ),
            "vgi.example_queries": _eq(_EMBED_EXAMPLES),
            "vgi.executable_examples": _VERIFIED_EXAMPLES,
        }

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to embed")],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_VECTOR)]:
        """Embed ``text`` with the default model."""
        return _embed_array(text, model=None)


class EmbedModel(ScalarFunction):
    """``embed(text, model)`` -- sentence embedding with an explicit model."""

    class Meta:
        """Declarative metadata for the ``embed(text, model)`` scalar."""

        name = "embed"
        description = (
            "Embed text into a FLOAT[] vector with an explicit model (see supported_models()). NULL/empty -> NULL."
        )
        categories = ["embedding"]
        examples = _fx(_EMBED_EXAMPLES)
        tags = {
            **meta.object_tags(
                title="Embed Text With Chosen Model",
                description_llm=(
                    "## embed(text, model)\n\n"
                    "The two-argument arity overload of `embed`: embed text into a "
                    "`FLOAT[]` vector using an *explicit* model name instead of the "
                    "default. Call `supported_models()` to discover valid model names "
                    "and their output dimensions.\n\n"
                    "**When to use.** Use this when you need a specific model -- a larger "
                    "model for higher recall, or a model whose dimension matches an "
                    "existing vector index. Otherwise use the one-argument `embed(text)`.\n\n"
                    "**Input/output.** Inputs: `VARCHAR` text and a constant `VARCHAR` model "
                    "name. Output: one `FLOAT[]` per row (NULL/empty text -> NULL). The "
                    "vector length depends on the chosen model; check `embedding_dim(model)`.\n\n"
                    "**Edge cases.** Scalars are positional-only, so the model is a "
                    "second positional argument, not a named one."
                ),
                description_md=(
                    "# embed(text, model)\n\n"
                    "Embed text with an explicitly chosen model (the two-argument arity "
                    "overload of `embed`).\n\n"
                    "## Usage\n\n"
                    "Pass the model name as a second positional argument, e.g. "
                    "`embed(text, 'BAAI/bge-small-en-v1.5')`; browse the `supported_models` "
                    "view for valid names and their vector widths. Runnable, catalog-qualified "
                    "examples are attached as example queries.\n\n"
                    "## Notes\n\n"
                    "- Pick a model from `supported_models()`; vector length varies by model.\n"
                    "- NULL or empty input returns NULL.\n"
                    "- The model name is a positional argument (scalars take no named args)."
                ),
                keywords=[
                    "embed",
                    "embedding",
                    "model",
                    "choose model",
                    "bge",
                    "fastembed",
                    "onnx",
                    "vector",
                    "text to vector",
                    "dimension",
                ],
                category="embedding",
            ),
            "vgi.example_queries": _eq(_EMBED_EXAMPLES),
        }

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to embed")],
        model: Annotated[str, ConstParam(doc="Model name; see supported_models()")],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_VECTOR)]:
        """Embed ``text`` with the explicit ``model``."""
        return _embed_array(text, model=model or None)


# ===========================================================================
# embed_query / embed_passage -- retrieval asymmetry (query gets the prefix)
# ===========================================================================


class EmbedQuery(ScalarFunction):
    """``embed_query(text)`` -- embed a *search query* (applies the model's instruction prefix)."""

    class Meta:
        """Declarative metadata for the ``embed_query(text)`` scalar."""

        name = "embed_query"
        description = (
            "Embed a search query with the default model, applying the model's "
            "recommended query instruction prefix (for bge: 'Represent this "
            "sentence for searching relevant passages: '). NULL/empty -> NULL."
        )
        categories = ["embedding", "retrieval"]
        examples = _fx(_EMBED_QUERY_EXAMPLES)
        tags = {
            **meta.object_tags(
                title="Embed Retrieval Query",
                description_llm=(
                    "## embed_query(text)\n\n"
                    "Embed a *search query* for retrieval, applying the default model's "
                    "recommended query instruction prefix (for bge: "
                    "`Represent this sentence for searching relevant passages: `). This is "
                    "the query side of asymmetric retrieval.\n\n"
                    "**When to use.** Use `embed_query` for the short user query and "
                    "`embed_passage` for the corpus documents, then rank passages by "
                    "`similarity(embed_query(q), embed_passage(doc))`. The prefix nudges "
                    "the query vector toward the passages that answer it, improving recall "
                    "over a plain symmetric `embed`.\n\n"
                    "**Input/output.** Input: one `VARCHAR` query per row. Output: one "
                    "`FLOAT[]` per row (NULL/empty -> NULL), same dimension as `embed`.\n\n"
                    "**Edge cases.** Pair only with `embed_passage` (not `embed`) so query "
                    "and passage live in the same trained retrieval space."
                ),
                description_md=(
                    "# embed_query(text)\n\n"
                    "Embed a search query for retrieval, applying the model's query "
                    "instruction prefix (the query side of asymmetric retrieval).\n\n"
                    "## Usage\n\n"
                    "Embed the short user query with `embed_query(text)` and the corpus "
                    "documents with `embed_passage(text)`, then rank documents by "
                    "`similarity(embed_passage(body), embed_query(q))` descending. Runnable, "
                    "catalog-qualified retrieval examples are attached as example queries.\n\n"
                    "## Notes\n\n"
                    "- Pair with `embed_passage` for the documents, not plain `embed`.\n"
                    "- NULL or empty input returns NULL."
                ),
                keywords=[
                    "embed query",
                    "retrieval",
                    "search",
                    "query embedding",
                    "asymmetric",
                    "instruction prefix",
                    "bge",
                    "rag",
                    "semantic search",
                    "vector",
                ],
                category="retrieval",
            ),
            "vgi.example_queries": _eq(_EMBED_QUERY_EXAMPLES),
        }

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Search query to embed")],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_VECTOR)]:
        """Embed a search query, applying the model's instruction prefix."""
        return _embed_array(text, model=None, prefix=models.query_prefix(None))


class EmbedPassage(ScalarFunction):
    """``embed_passage(text)`` -- embed a *document/passage* (no prefix, by design)."""

    class Meta:
        """Declarative metadata for the ``embed_passage(text)`` scalar."""

        name = "embed_passage"
        description = (
            "Embed a document/passage with the default model. For bge retrieval "
            "models passages get NO instruction prefix (queries do); this mirrors "
            "that. NULL/empty -> NULL."
        )
        categories = ["embedding", "retrieval"]
        examples = _fx(_EMBED_PASSAGE_EXAMPLES)
        tags = {
            **meta.object_tags(
                title="Embed Document Passage",
                description_llm=(
                    "## embed_passage(text)\n\n"
                    "Embed a *document/passage* for retrieval. For bge-style retrieval "
                    "models, passages get **no** instruction prefix (only queries do), and "
                    "this function mirrors that convention exactly.\n\n"
                    "**When to use.** Use `embed_passage` for the corpus side of asymmetric "
                    "retrieval and `embed_query` for the user query, then rank with "
                    "`similarity(embed_query(q), embed_passage(doc))`. Pre-compute passage "
                    "vectors once and store them (e.g. in a DuckDB VSS HNSW index).\n\n"
                    "**Input/output.** Input: one `VARCHAR` passage per row. Output: one "
                    "`FLOAT[]` per row (NULL/empty -> NULL), same dimension as `embed`.\n\n"
                    "**Edge cases.** Numerically this equals plain `embed` for bge (passages "
                    "have no prefix); it exists as a named counterpart to `embed_query` so "
                    "retrieval pipelines read symmetrically."
                ),
                description_md=(
                    "# embed_passage(text)\n\n"
                    "Embed a document/passage for retrieval (the corpus side of asymmetric "
                    "retrieval -- no query prefix).\n\n"
                    "## Usage\n\n"
                    "Materialize passage vectors once with `embed_passage(body)` into a stored "
                    "column (e.g. a DuckDB VSS HNSW index over the resulting `FLOAT[]`), then at "
                    "query time score them against `embed_query(q)`. Runnable, catalog-qualified "
                    "examples are attached as example queries.\n\n"
                    "## Notes\n\n"
                    "- Pair with `embed_query` for the query side.\n"
                    "- For bge, passages get no instruction prefix (by design).\n"
                    "- NULL or empty input returns NULL."
                ),
                keywords=[
                    "embed passage",
                    "document embedding",
                    "corpus",
                    "retrieval",
                    "indexing",
                    "no prefix",
                    "bge",
                    "rag",
                    "semantic search",
                    "vector",
                ],
                category="retrieval",
            ),
            "vgi.example_queries": _eq(_EMBED_PASSAGE_EXAMPLES),
        }

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Passage/document text to embed")],
    ) -> Annotated[pa.ListArray, Returns(arrow_type=_VECTOR)]:
        """Embed a passage/document with no query prefix."""
        return _embed_array(text, model=None)


# ===========================================================================
# similarity(a, b) -- pure cosine similarity, no model
# ===========================================================================


class Similarity(ScalarFunction):
    """``similarity(a, b)`` -- cosine similarity of two FLOAT[] vectors, in [-1, 1]."""

    class Meta:
        """Declarative metadata for the ``similarity(a, b)`` scalar."""

        name = "similarity"
        description = (
            "Cosine similarity of two FLOAT[] vectors, in [-1, 1] (pure arithmetic, "
            "no model). NULL/empty/length-mismatch -> NULL."
        )
        categories = ["similarity"]
        examples = _fx(_SIMILARITY_EXAMPLES)
        tags = {
            **meta.object_tags(
                title="Cosine Similarity Score",
                description_llm=(
                    "## similarity(a, b)\n\n"
                    "Compute the **cosine similarity** of two `FLOAT[]` vectors, a value in "
                    "`[-1, 1]` where 1 means identical direction. This is pure arithmetic "
                    "-- it loads no model and does no I/O -- so it is cheap to call over "
                    "large joins.\n\n"
                    "**When to use.** Use it to rank or threshold embedding pairs: "
                    "`ORDER BY similarity(embed_query(q), embed_passage(doc)) DESC` for "
                    "retrieval, or a `WHERE similarity(...) > 0.8` cutoff for dedup/"
                    "near-duplicate detection.\n\n"
                    "**Input/output.** Inputs: two `FLOAT[]` vectors (typically from "
                    "`embed*`). Output: one `DOUBLE` per row. NULL, empty, or "
                    "length-mismatched vector pairs yield NULL rather than an error.\n\n"
                    "**Edge cases.** A zero vector has undefined direction and yields NULL; "
                    "vectors must be the same length to be comparable."
                ),
                description_md=(
                    "# similarity(a, b)\n\n"
                    "Cosine similarity of two `FLOAT[]` vectors, in `[-1, 1]` (pure "
                    "arithmetic, no model).\n\n"
                    "## Usage\n\n"
                    "Feed two `FLOAT[]` vectors (typically from the `embed*` functions) to "
                    "`similarity(a, b)` to get a `DOUBLE` in `[-1, 1]`; order retrieval "
                    "candidates by it descending, or threshold it (e.g. `> 0.8`) for "
                    "near-duplicate detection. Runnable, catalog-qualified examples are "
                    "attached as example queries.\n\n"
                    "## Notes\n\n"
                    "- 1.0 = identical direction, 0 = orthogonal, -1 = opposite.\n"
                    "- NULL / empty / length-mismatch returns NULL."
                ),
                keywords=[
                    "similarity",
                    "cosine similarity",
                    "cosine",
                    "distance",
                    "score",
                    "rank",
                    "compare vectors",
                    "dot product",
                    "nearest neighbor",
                    "semantic search",
                ],
                category="similarity",
            ),
            "vgi.example_queries": _eq(_SIMILARITY_EXAMPLES),
        }

    @classmethod
    def compute(
        cls,
        a: Annotated[pa.ListArray, Param(arrow_type=_VECTOR, doc="First embedding vector to compare")],
        b: Annotated[
            pa.ListArray, Param(arrow_type=_VECTOR, doc="Second embedding vector, compared against the first")
        ],
    ) -> Annotated[pa.DoubleArray, Returns(arrow_type=pa.float64())]:
        """Cosine similarity of each ``(a, b)`` vector pair."""
        out = [models.cosine_similarity(x, y) for x, y in zip(a.to_pylist(), b.to_pylist(), strict=False)]
        return pa.array(out, type=pa.float64())


# ===========================================================================
# embedding_dim(model) -- metadata helper
# ===========================================================================


class EmbeddingDim(ScalarFunction):
    """``embedding_dim(model)`` -- the output dimension for a model name."""

    class Meta:
        """Declarative metadata for the ``embedding_dim(model)`` scalar."""

        name = "embedding_dim"
        description = "Output dimension (vector length) for a model name; '' = default model"
        categories = ["embedding", "metadata"]
        examples = _fx(_EMBEDDING_DIM_EXAMPLES)
        tags = {
            **meta.object_tags(
                title="Embedding Dimension Lookup",
                description_llm=(
                    "## embedding_dim(model)\n\n"
                    "Return the output dimension (the `FLOAT[]` length) a model produces, "
                    "**without** loading the model or embedding anything. Pass `''` or NULL "
                    "to get the default model's dimension.\n\n"
                    "**When to use.** Use it to size a vector column or index ahead of time "
                    "(e.g. an `ARRAY[FLOAT, N]` / VSS HNSW index), or to validate that a "
                    "chosen model matches an existing index width before re-embedding a "
                    "corpus.\n\n"
                    "**Input/output.** Input: a `VARCHAR` model name (`''`/NULL = default). "
                    "Output: `INTEGER` dimension, or NULL for an unknown model name (it does "
                    "not raise).\n\n"
                    "**Edge cases.** Unknown model names return NULL rather than erroring, "
                    "so a dirty model column won't abort a scan."
                ),
                description_md=(
                    "# embedding_dim(model)\n\n"
                    "Output dimension (vector length) for a model name -- `''`/NULL = the "
                    "default model. No model is loaded.\n\n"
                    "## Usage\n\n"
                    "```sql\n"
                    "SELECT embed.embedding_dim('BAAI/bge-small-en-v1.5');  -- 384\n"
                    "SELECT embed.embedding_dim('');                        -- default model dim\n"
                    "```\n\n"
                    "## Notes\n\n"
                    "- Unknown model name -> NULL (no error).\n"
                    "- Use to size a vector column / VSS index before embedding."
                ),
                keywords=[
                    "embedding dim",
                    "dimension",
                    "vector length",
                    "model size",
                    "float array length",
                    "index width",
                    "metadata",
                    "supported models",
                ],
                category="discovery",
            ),
            "vgi.example_queries": _eq(_EMBEDDING_DIM_EXAMPLES),
        }

    @classmethod
    def compute(
        cls,
        model: Annotated[pa.StringArray, Param(doc="Model name; '' or NULL = default model")],
    ) -> Annotated[pa.Int32Array, Returns(arrow_type=pa.int32())]:
        """Output dimension per model name (unknown -> NULL)."""
        out: list[int | None] = []
        for m in model.to_pylist():
            try:
                out.append(models.embedding_dim(m))
            except models.ModelNotAvailableError:
                # Unknown model name -> NULL rather than crashing the whole query.
                out.append(None)
        return pa.array(out, type=pa.int32())


SCALAR_FUNCTIONS: list[type] = [
    Embed,
    EmbedModel,
    EmbedQuery,
    EmbedPassage,
    Similarity,
    EmbeddingDim,
]
