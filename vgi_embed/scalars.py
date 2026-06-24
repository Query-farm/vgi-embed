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

from typing import Annotated

import pyarrow as pa
from vgi import Param, Returns, ScalarFunction
from vgi.arguments import ConstParam
from vgi.metadata import FunctionExample

from . import meta, models

_SRC = "vgi_embed/scalars.py"

# Arrow return type shared by every embedding scalar: a list of float32.
_VECTOR = pa.list_(pa.float32())

# VGI509 guaranteed-runnable, catalog-qualified examples. Each ``sql`` is
# self-contained and re-runnable against an attached ``embed`` worker; we omit
# ``expected_result`` deliberately (exact float values vary by ONNX build, and
# the linter only needs each query to execute cleanly).
_EXECUTABLE_EXAMPLES = """[
  {
    "description": "Embed a string into a 384-dim FLOAT[] with the default model.",
    "sql": "SELECT len(embed.main.embed('hello world')) AS dim"
  },
  {
    "description": "A sentence is more similar to itself than to an unrelated one.",
    "sql": "SELECT ROUND(embed.main.similarity(embed.main.embed('dog'), embed.main.embed('puppy')), 3) AS related"
  },
  {
    "description": "Embed a search query with the query-side instruction prefix.",
    "sql": "SELECT len(embed.main.embed_query('how do I reset my password')) AS dim"
  },
  {
    "description": "List the supported embedding models and their dimensions.",
    "sql": "SELECT model, dim FROM embed.main.supported_models() ORDER BY model"
  }
]"""


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
        """Declarative metadata for the ``embed(text)`` scalar."""

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
                    "**Input/output.** Input: one VARCHAR per row. Output: one "
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
                    "```sql\n"
                    "SELECT embed.embed('hello world');            -- FLOAT[384]\n"
                    "SELECT id FROM docs ORDER BY\n"
                    "  embed.similarity(embed.embed(title), embed.embed('reset password')) DESC;\n"
                    "```\n\n"
                    "## Notes\n\n"
                    "- Best for symmetric text-to-text comparison; for query/passage "
                    "retrieval use `embed_query` / `embed_passage`.\n"
                    "- NULL or empty input returns NULL.\n"
                    "- Runs locally (fastembed/ONNX); pairs with DuckDB VSS."
                ),
                keywords=(
                    "embed, embedding, vector, fastembed, onnx, sentence embedding, "
                    "semantic, text to vector, float array, similarity, rag"
                ),
                relative_path=_SRC,
            ),
            "vgi.executable_examples": _EXECUTABLE_EXAMPLES,
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
        examples = _ex(
            "SELECT embed.embed('hello world', 'BAAI/bge-small-en-v1.5')",
            "Embed a string with a chosen model",
        )
        tags = meta.object_tags(
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
                "**Input/output.** Inputs: VARCHAR text and a constant VARCHAR model "
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
                "```sql\n"
                "SELECT embed.embed('hello world', 'BAAI/bge-small-en-v1.5');\n"
                "SELECT model, dim FROM embed.supported_models();\n"
                "```\n\n"
                "## Notes\n\n"
                "- Pick a model from `supported_models()`; vector length varies by model.\n"
                "- NULL or empty input returns NULL.\n"
                "- The model name is a positional argument (scalars take no named args)."
            ),
            keywords=("embed, embedding, model, choose model, bge, fastembed, onnx, vector, text to vector, dimension"),
            relative_path=_SRC,
        )

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
        examples = _ex(
            "SELECT embed.embed_query('how do I reset my password')",
            "Embed a retrieval query (query-side prefix applied)",
        )
        tags = meta.object_tags(
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
                "**Input/output.** Input: one VARCHAR query per row. Output: one "
                "`FLOAT[]` per row (NULL/empty -> NULL), same dimension as `embed`.\n\n"
                "**Edge cases.** Pair only with `embed_passage` (not `embed`) so query "
                "and passage live in the same trained retrieval space."
            ),
            description_md=(
                "# embed_query(text)\n\n"
                "Embed a search query for retrieval, applying the model's query "
                "instruction prefix (the query side of asymmetric retrieval).\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT id FROM docs ORDER BY\n"
                "  embed.similarity(embed.embed_passage(body),\n"
                "                   embed.embed_query('reset password')) DESC\n"
                "LIMIT 5;\n"
                "```\n\n"
                "## Notes\n\n"
                "- Pair with `embed_passage` for the documents, not plain `embed`.\n"
                "- NULL or empty input returns NULL."
            ),
            keywords=(
                "embed query, retrieval, search, query embedding, asymmetric, "
                "instruction prefix, bge, rag, semantic search, vector"
            ),
            relative_path=_SRC,
        )

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
        examples = _ex(
            "SELECT embed.embed_passage('Reset your password from the account settings page.')",
            "Embed a corpus passage (no query prefix)",
        )
        tags = meta.object_tags(
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
                "**Input/output.** Input: one VARCHAR passage per row. Output: one "
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
                "```sql\n"
                "-- pre-compute and store passage vectors\n"
                "CREATE TABLE doc_vecs AS\n"
                "  SELECT id, embed.embed_passage(body) AS v FROM docs;\n"
                "```\n\n"
                "## Notes\n\n"
                "- Pair with `embed_query` for the query side.\n"
                "- For bge, passages get no instruction prefix (by design).\n"
                "- NULL or empty input returns NULL."
            ),
            keywords=(
                "embed passage, document embedding, corpus, retrieval, indexing, "
                "no prefix, bge, rag, semantic search, vector"
            ),
            relative_path=_SRC,
        )

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
        examples = _ex(
            "SELECT embed.similarity(embed.embed('cat'), embed.embed('kitten'))",
            "Cosine similarity between two embeddings",
        )
        tags = meta.object_tags(
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
                "`embed*`). Output: one DOUBLE per row. NULL, empty, or "
                "length-mismatched vector pairs yield NULL rather than an error.\n\n"
                "**Edge cases.** A zero vector has undefined direction and yields NULL; "
                "vectors must be the same length to be comparable."
            ),
            description_md=(
                "# similarity(a, b)\n\n"
                "Cosine similarity of two `FLOAT[]` vectors, in `[-1, 1]` (pure "
                "arithmetic, no model).\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT embed.similarity(embed.embed('cat'), embed.embed('kitten'));\n"
                "SELECT id FROM docs ORDER BY\n"
                "  embed.similarity(embed.embed_passage(body), embed.embed_query(:q)) DESC;\n"
                "```\n\n"
                "## Notes\n\n"
                "- 1.0 = identical direction, 0 = orthogonal, -1 = opposite.\n"
                "- NULL / empty / length-mismatch returns NULL."
            ),
            keywords=(
                "similarity, cosine similarity, cosine, distance, score, rank, "
                "compare vectors, dot product, nearest neighbor, semantic search"
            ),
            relative_path=_SRC,
        )

    @classmethod
    def compute(
        cls,
        a: Annotated[pa.ListArray, Param(arrow_type=_VECTOR, doc="First FLOAT[] vector")],
        b: Annotated[pa.ListArray, Param(arrow_type=_VECTOR, doc="Second FLOAT[] vector")],
    ) -> Annotated[pa.DoubleArray, Returns(arrow_type=pa.float64())]:
        """Cosine similarity of each ``(a, b)`` vector pair."""
        out = [models.cosine_similarity(x, y) for x, y in zip(a.to_pylist(), b.to_pylist(), strict=False)]
        return pa.array(out, type=pa.float64())


# ===========================================================================
# embedding_dim(model) / embed_version() -- metadata helpers
# ===========================================================================


class EmbeddingDim(ScalarFunction):
    """``embedding_dim(model)`` -- the output dimension for a model name."""

    class Meta:
        """Declarative metadata for the ``embedding_dim(model)`` scalar."""

        name = "embedding_dim"
        description = "Output dimension (vector length) for a model name; '' = default model"
        categories = ["embedding", "metadata"]
        examples = _ex(
            "SELECT embed.embedding_dim('BAAI/bge-small-en-v1.5')",
            "Dimension of the default model (384)",
        )
        tags = meta.object_tags(
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
                "**Input/output.** Input: a VARCHAR model name (`''`/NULL = default). "
                "Output: INTEGER dimension, or NULL for an unknown model name (it does "
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
            keywords=(
                "embedding dim, dimension, vector length, model size, float array "
                "length, index width, metadata, supported models"
            ),
            relative_path=_SRC,
        )

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


class EmbedVersion(ScalarFunction):
    """``embed_version()`` -- worker + default-model identity string."""

    class Meta:
        """Declarative metadata for the ``embed_version()`` scalar."""

        name = "embed_version"
        description = "Version string: worker version, fastembed backend, and default model"
        categories = ["metadata"]
        examples = _ex(
            "SELECT embed.embed_version()",
            "Identify the embed worker / default model",
        )
        tags = meta.object_tags(
            title="Embed Worker Version",
            description_llm=(
                "## embed_version()\n\n"
                "Return a single human-readable identity string for the worker: its "
                "`vgi-embed` version, the `fastembed` backend version, and the default "
                "model name. Takes no arguments.\n\n"
                "**When to use.** Use it for diagnostics and reproducibility -- to "
                "record which worker/model produced a set of vectors, or to confirm "
                "an attached worker is the build you expect before a batch job.\n\n"
                "**Input/output.** No input. Output: one VARCHAR row, e.g. "
                "`vgi-embed 0.1.0 (fastembed 0.3.x; default BAAI/bge-small-en-v1.5)`.\n\n"
                "**Edge cases.** The backend version is best-effort; if it can't be "
                "resolved the string still returns with a bare `fastembed` token."
            ),
            description_md=(
                "# embed_version()\n\n"
                "Identity string: the worker version, the fastembed backend version, "
                "and the default model.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT embed.embed_version();\n"
                "-- vgi-embed 0.1.0 (fastembed 0.3.x; default BAAI/bge-small-en-v1.5)\n"
                "```\n\n"
                "## Notes\n\n"
                "- Useful for diagnostics and reproducibility.\n"
                "- Backend version lookup is best-effort."
            ),
            keywords=(
                "version, embed_version, build, diagnostics, fastembed, default "
                "model, worker info, reproducibility, metadata"
            ),
            relative_path=_SRC,
        )

    @classmethod
    def compute(
        cls,
    ) -> Annotated[pa.StringArray, Returns(arrow_type=pa.string())]:
        """Return the worker/default-model identity string."""
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
