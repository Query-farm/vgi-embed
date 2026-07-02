# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.9.0",
#     "fastembed>=0.3",
# ]
# ///
"""VGI worker exposing local text embeddings (fastembed/ONNX) to DuckDB/SQL.

Assembles the scalar and table functions in ``vgi_embed`` into a single ``embed``
catalog and runs the worker over stdio (DuckDB subprocess) or HTTP (via serve.py).

The embeddings are generated locally with `fastembed` (Qdrant, Apache-2.0), which
runs sentence-transformer models through ONNX Runtime -- **no torch**. The default
model ``BAAI/bge-small-en-v1.5`` (384-dim, MIT) is downloaded on first use and
cached (gitignored); see ``vgi_embed/models.py``.

Usage:
    uv run embed_worker.py               # serve over stdio (DuckDB subprocess)
    python serve.py --port 8000          # serve over HTTP

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'embed' (TYPE vgi, LOCATION 'uv run embed_worker.py');

    SELECT embed.embed('hello world');                          -- FLOAT[384]
    SELECT embed.similarity(embed.embed('cat'), embed.embed('kitten'));
    SELECT id FROM docs
      ORDER BY embed.similarity(embed.embed_passage(body),
                                embed.embed_query('reset password')) DESC
      LIMIT 5;
    SELECT * FROM embed.supported_models();
"""

from __future__ import annotations

import json
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, Schema, View

from vgi_embed import models
from vgi_embed.scalars import SCALAR_FUNCTIONS
from vgi_embed.tables import TABLE_FUNCTIONS

# VGI311 -- the parameterless table function `supported_models()` always returns
# the same rows, so we also expose it as a plain VIEW of the same name. That lets
# consumers write `SELECT * FROM embed.main.supported_models` (no parentheses);
# the view simply scans the backing table function.
_SUPPORTED_MODELS_VIEW = View(
    name="supported_models",
    definition="SELECT model, dim FROM embed.main.supported_models()",
    comment="Discovery table of every (model, dim) the embed worker supports.",
    column_comments={
        "model": "Model name to pass to embed(text, model) or embedding_dim(model).",
        "dim": "Embedding dimension (FLOAT[] length) the model produces.",
    },
    tags={
        "vgi.title": "Supported Models (table)",
        "vgi.doc_llm": (
            "A ready-to-scan **discovery table** of every `(model, dim)` pair the embed "
            "worker supports, so you can find the valid model names to pass as the second "
            "argument of `embed(text, model)` and to `embedding_dim(model)`, along with the "
            "`FLOAT[]` length each model produces. This is the no-argument table form of the "
            "`supported_models()` table function -- query it directly (no parentheses):\n\n"
            "```sql\n"
            "SELECT * FROM embed.main.supported_models;\n"
            "```"
        ),
        "vgi.doc_md": (
            "## supported_models (view)\n\n"
            "Every `(model, dim)` the embed worker supports, as a plain table.\n\n"
            "`model` is the name to pass to `embed(text, model)` / `embedding_dim(model)`; "
            "`dim` is the `FLOAT[]` length the model produces. The no-argument table form of "
            "`supported_models()` -- scan it directly:\n\n"
            "```sql\n"
            "SELECT * FROM embed.main.supported_models;\n"
            "```"
        ),
        "vgi.keywords": json.dumps(
            [
                "supported models",
                "list models",
                "available models",
                "model catalog",
                "discovery",
                "dimension",
                "embedding models",
                "bge",
                "what models",
            ]
        ),
        "domain": "machine-learning",
        "category": "embeddings",
        "topic": "supported-models",
        "vgi.category": "discovery",
        "vgi.example_queries": json.dumps(
            [
                {
                    "description": "List the supported embedding models and their dimensions.",
                    "sql": "SELECT model, dim FROM embed.main.supported_models ORDER BY model",
                },
                {
                    "description": "Dimension of the default model.",
                    "sql": "SELECT dim FROM embed.main.supported_models WHERE model = 'BAAI/bge-small-en-v1.5'",
                },
            ]
        ),
    },
)


_EMBED_CATALOG = Catalog(
    name="embed",
    default_schema="main",
    comment="Local text embeddings (fastembed/ONNX) + cosine similarity for semantic search / RAG.",
    tags={
        "vgi.title": "Local Text Embeddings & Similarity",
        "vgi.keywords": json.dumps(
            [
                "embeddings",
                "embed",
                "text embedding",
                "vector",
                "fastembed",
                "onnx",
                "cosine similarity",
                "semantic search",
                "retrieval",
                "rag",
                "bge",
                "sentence transformer",
                "nearest neighbor",
                "vss",
            ]
        ),
        "vgi.doc_llm": (
            "Turn text into fixed-length FLOAT[] embedding vectors entirely in-process "
            "(fastembed/ONNX, no torch, no network call after a one-time model download) "
            "and compare them with cosine similarity. This is the offline, in-database "
            "building block for semantic search, retrieval-augmented generation (RAG), "
            "clustering, and near-duplicate detection: embed a corpus once, embed a "
            "query, and rank rows by vector similarity -- pairing naturally with the "
            "DuckDB VSS extension for indexed nearest-neighbor search. It covers both "
            "symmetric text-to-text embedding and the asymmetric query/passage "
            "convention that retrieval models expect, plus a pure-arithmetic similarity "
            "score. Reach for it whenever you want vectors and ranking inside SQL "
            "instead of standing up a separate embedding service. List this catalog's "
            "schema to discover the embedding, retrieval, similarity, and "
            "model-discovery functions it provides."
        ),
        "vgi.doc_md": (
            "# Local Text Embeddings & Semantic Search in SQL\n\n"
            "![fastembed logo](https://qdrant.tech/images/logo_with_text.png)\n\n"
            "**Turn text into vector embeddings directly in DuckDB SQL** — generate "
            "fixed-length `FLOAT[]` embedding vectors and score them with cosine "
            "similarity for semantic search, retrieval-augmented generation (RAG), "
            "and nearest-neighbor ranking, all without leaving your query and "
            "without sending a single byte to an external API.\n\n"
            "The **embed** extension brings sentence-transformer text embeddings to "
            "SQL for data engineers, RAG builders, and search teams who want vector "
            "search inside the database instead of a separate embedding service. "
            "Embeddings are computed **entirely in-process and offline** — there is "
            "no torch dependency, no GPU requirement, and no network call after the "
            "one-time model download — so the same query runs identically on a "
            "laptop, in CI, and in production.\n\n"
            "Under the hood it is powered by [fastembed](https://github.com/qdrant/fastembed) "
            "from [Qdrant](https://qdrant.tech) (Apache-2.0), which runs quantized "
            "sentence-transformer models through [ONNX Runtime](https://onnxruntime.ai) "
            "([source](https://github.com/microsoft/onnxruntime)). The default model is "
            "[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) "
            "(384-dimensional, MIT licensed), downloaded on first use and cached "
            "locally; larger `bge` models and `all-MiniLM-L6-v2` are also supported. "
            "Vectors are returned over Apache Arrow as native DuckDB `FLOAT[]` lists, "
            "ready to index and rank with the "
            "[DuckDB VSS extension](https://duckdb.org/docs/extensions/vss.html).\n\n"
            "## How it works\n\n"
            "There are two embedding modes. *Symmetric* embedding maps any text to a "
            "vector for text-to-text comparison (dedup, clustering, classification). "
            "*Asymmetric* retrieval follows the query/passage convention these models "
            "are trained on: a short search query and a longer document are embedded "
            "with slightly different instruction prefixes so they land in the same "
            "space, which improves recall over plain symmetric embedding. Ranking is "
            "then just cosine similarity between vectors -- pure arithmetic, no model "
            "load. Model discovery and dimension lookups let you size a vector column "
            "or VSS index ahead of time.\n\n"
            "```sql\n"
            "-- A sentence is more similar to itself than to an unrelated one\n"
            "SELECT embed.similarity(embed.embed('cat'), embed.embed('kitten')) AS score;\n"
            "```\n\n"
            "NULL or empty input text yields a NULL vector, and the model is loaded "
            "once and amortized across every row in a scan."
        ),
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "MIT",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-embed/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-embed/blob/main/README.md",
        # VGI152/VGI920 -- a fixed analyst-task suite `vgi-lint simulate` runs to
        # measure how well an agent can actually drive this worker. Every task has
        # a deterministic reference despite embeddings being float-valued: we
        # assert vector *length*, self-similarity rounded to 1.0, a planted
        # related>unrelated ordering, and the discovery table's exact rows.
        "vgi.agent_test_tasks": json.dumps(
            [
                {
                    "name": "default_model_dimension",
                    "prompt": (
                        "What is the vector length (embedding dimension) of the default model, BAAI/bge-small-en-v1.5?"
                    ),
                    "reference_sql": (
                        "SELECT dim FROM embed.main.supported_models() WHERE model = 'BAAI/bge-small-en-v1.5'"
                    ),
                    "success_criteria": "Returns the default model's dimension, 384.",
                    "ignore_column_names": True,
                },
                {
                    "name": "list_supported_models",
                    "prompt": "List the names of every embedding model this worker supports.",
                    "reference_sql": "SELECT model FROM embed.main.supported_models() ORDER BY model",
                    "success_criteria": "Returns the set of supported model names.",
                    "unordered": True,
                    "ignore_column_names": True,
                },
                {
                    "name": "self_similarity_is_one",
                    "prompt": (
                        "Confirm that a piece of text is maximally similar to itself: "
                        "compute the cosine similarity of the embedding of the word "
                        "'database' with itself, rounded to 3 decimal places."
                    ),
                    "reference_sql": (
                        "SELECT ROUND(embed.main.similarity("
                        "embed.main.embed('database'), embed.main.embed('database')), 3) AS sim"
                    ),
                    "success_criteria": "Returns 1.0 (a vector is identical to itself).",
                    "ignore_column_names": True,
                },
                {
                    "name": "related_more_similar_than_unrelated",
                    "prompt": (
                        "Is the word 'dog' more semantically similar to 'puppy' than to "
                        "'airplane'? Return a single boolean."
                    ),
                    "reference_sql": (
                        "SELECT embed.main.similarity(embed.main.embed('dog'), embed.main.embed('puppy')) "
                        "> embed.main.similarity(embed.main.embed('dog'), embed.main.embed('airplane')) "
                        "AS related_more_similar"
                    ),
                    "success_criteria": "Returns true; a related pair scores higher than an unrelated pair.",
                    "ignore_column_names": True,
                },
            ]
        ),
    },
    source_url="https://github.com/Query-farm/vgi-embed",
    schemas=[
        Schema(
            name="main",
            comment="Local text embeddings (fastembed/ONNX) + cosine similarity for SQL",
            tags={
                "vgi.title": "Embed — main schema",
                "vgi.keywords": json.dumps(
                    [
                        "embed",
                        "embed_query",
                        "embed_passage",
                        "similarity",
                        "embedding_dim",
                        "embed_version",
                        "supported_models",
                        "embeddings",
                        "vector",
                        "cosine",
                        "semantic search",
                        "retrieval",
                        "rag",
                    ]
                ),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "machine-learning",
                "category": "embeddings",
                "topic": "semantic-search",
                # VGI413 navigation/SEO registry: ordered categories every object
                # (function/view) is assigned to via its own `vgi.category` tag.
                "vgi.categories": json.dumps(
                    [
                        {
                            "name": "embedding",
                            "title": "Embedding",
                            "description": "Turn text into fixed-length FLOAT[] vectors for symmetric text-to-text comparison.",
                        },
                        {
                            "name": "retrieval",
                            "title": "Retrieval",
                            "description": "Asymmetric query/passage embedding for semantic search and RAG.",
                        },
                        {
                            "name": "similarity",
                            "title": "Similarity",
                            "description": "Score and rank embedding vectors by cosine similarity.",
                        },
                        {
                            "name": "discovery",
                            "title": "Discovery",
                            "description": "Introspect supported models, vector dimensions, and worker identity.",
                        },
                    ]
                ),
                "vgi.doc_llm": (
                    "## embed.main schema\n\n"
                    "The single schema of the embed worker. It groups the local "
                    "text-embedding and vector-similarity surface into a few concepts: "
                    "symmetric text-to-text embedding, the asymmetric query/passage "
                    "convention used for retrieval, a pure-arithmetic cosine-similarity "
                    "score, and model-discovery/dimension helpers for sizing vector "
                    "columns and indexes.\n\n"
                    "Everything runs in-process via fastembed/ONNX (no torch, no "
                    "network after the one-time model download). Use it to build "
                    "semantic search and RAG pipelines that store and rank vectors "
                    "with DuckDB VSS, all in SQL. List the schema to see the exact "
                    "functions and their signatures."
                ),
                "vgi.doc_md": (
                    "# embed.main\n\n"
                    "Local text-embedding and cosine-similarity functions over Apache "
                    "Arrow, for semantic search and RAG with DuckDB.\n\n"
                    "## Overview\n\n"
                    "This schema holds the worker's scalar embedding and similarity "
                    "helpers plus a model-discovery table. Embeddings are generated "
                    "locally with fastembed on ONNX Runtime; the default model is "
                    "`BAAI/bge-small-en-v1.5` (384-dim). Symmetric embedding is for "
                    "text-to-text comparison; the query/passage pair is for asymmetric "
                    "retrieval.\n\n"
                    "## Usage\n\n"
                    "```sql\n"
                    "SELECT embed.similarity(embed.embed('cat'), embed.embed('kitten')) AS score;\n"
                    "```\n\n"
                    "## Notes\n\n"
                    "- Use the query/passage pair for retrieval; use plain symmetric "
                    "embedding for text-to-text comparison.\n"
                    "- NULL or empty text yields a NULL vector."
                ),
                # VGI506 representative, catalog-qualified example queries for the schema.
                "vgi.example_queries": (
                    "SELECT embed.main.embed('hello world');\n"
                    "SELECT embed.main.embed('hello world', 'BAAI/bge-small-en-v1.5');\n"
                    "SELECT embed.main.embed_query('how do I reset my password');\n"
                    "SELECT embed.main.embed_passage('Reset your password in account settings.');\n"
                    "SELECT embed.main.similarity(embed.main.embed('cat'), embed.main.embed('kitten'));\n"
                    "SELECT embed.main.embedding_dim('BAAI/bge-small-en-v1.5');\n"
                    "SELECT embed.main.embed_version();\n"
                    "SELECT * FROM embed.main.supported_models() ORDER BY model;"
                ),
            },
            functions=[*SCALAR_FUNCTIONS, *TABLE_FUNCTIONS],
            views=[_SUPPORTED_MODELS_VIEW],
        ),
    ],
)


class EmbedWorker(Worker):
    """Worker process hosting the ``embed`` catalog."""

    catalog = _EMBED_CATALOG

    def run(self, otel_config: Any = None) -> None:
        """Warm the default model, then serve.

        Loading (and, on a cold cache, *downloading*) the ONNX model is lazy, so
        without this the first query of every ATTACH pays that multi-second cost
        inline -- a window in which a worker-pool teardown SIGTERM (or a heavily
        loaded host) can kill the run mid-assertion and record a spurious E2E
        failure. Warming at spawn moves that one-time cost ahead of any query,
        keeping the SQL suite deterministic without changing a single output
        value. Best-effort; never fatal.
        """
        models.warm_up()
        super().run(otel_config=otel_config)


def main() -> None:
    """Run the embed worker process (stdio or, via flags, HTTP)."""
    EmbedWorker.main()


if __name__ == "__main__":
    main()
