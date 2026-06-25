# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.5",
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
            "`supported_models()` table function -- query it directly with "
            "`SELECT * FROM embed.main.supported_models` (no parentheses)."
        ),
        "vgi.doc_md": (
            "## supported_models (view)\n\n"
            "Every `(model, dim)` the embed worker supports, as a plain table.\n\n"
            "`model` is the name to pass to `embed(text, model)` / `embedding_dim(model)`; "
            "`dim` is the `FLOAT[]` length the model produces. The no-argument table form of "
            "`supported_models()` -- scan it with `SELECT * FROM embed.main.supported_models`."
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
            "(fastembed/ONNX, no torch, no network) and compare them with cosine similarity. "
            "Use embed(text) for symmetric embeddings, embed_query(text)/embed_passage(text) "
            "for retrieval asymmetry, similarity(a, b) to score two vectors, and "
            "supported_models() to discover available models. Pairs with DuckDB VSS for "
            "semantic search and RAG."
        ),
        "vgi.doc_md": (
            "# embed\n\n"
            "Local text embeddings (fastembed/ONNX, no torch) and cosine similarity over "
            "Apache Arrow, for semantic search / RAG with DuckDB VSS.\n\n"
            "Scalars: `embed`, `embed_query`, `embed_passage`, `similarity`, "
            "`embedding_dim`, `embed_version`. Table: `supported_models`.\n\n"
            "The default model is `BAAI/bge-small-en-v1.5` (384-dim, MIT), downloaded on "
            "first use and cached locally."
        ),
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "MIT",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-embed/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-embed/blob/main/README.md",
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
                "vgi.doc_llm": (
                    "## embed.main schema\n\n"
                    "The single schema of the `embed` worker. It groups all local "
                    "text-embedding and vector-similarity functions:\n\n"
                    "- `embed(text)` / `embed(text, model)` -- symmetric sentence "
                    "embeddings into `FLOAT[]` vectors.\n"
                    "- `embed_query(text)` / `embed_passage(text)` -- the query and "
                    "passage sides of asymmetric retrieval.\n"
                    "- `similarity(a, b)` -- cosine similarity of two vectors.\n"
                    "- `embedding_dim(model)` -- a model's output dimension.\n"
                    "- `embed_version()` -- worker/model identity.\n"
                    "- `supported_models()` -- table of available `(model, dim)` pairs.\n\n"
                    "Everything runs in-process via fastembed/ONNX (no torch, no "
                    "network after the one-time model download). Use it to build "
                    "semantic search and RAG pipelines that store and rank vectors "
                    "with DuckDB VSS, all in SQL."
                ),
                "vgi.doc_md": (
                    "# embed.main\n\n"
                    "Local text-embedding and cosine-similarity functions over Apache "
                    "Arrow, for semantic search and RAG with DuckDB.\n\n"
                    "## Overview\n\n"
                    "This schema holds the worker's scalar embedding/similarity helpers "
                    "and the `supported_models()` discovery table. Embeddings are "
                    "generated locally with fastembed on ONNX Runtime; the default "
                    "model is `BAAI/bge-small-en-v1.5` (384-dim).\n\n"
                    "## Usage\n\n"
                    "```sql\n"
                    "SELECT embed.main.embed('hello world');\n"
                    "SELECT embed.main.similarity(\n"
                    "  embed.main.embed_query('reset password'),\n"
                    "  embed.main.embed_passage(body)) FROM docs;\n"
                    "```\n\n"
                    "## Notes\n\n"
                    "- Pair `embed_query` with `embed_passage` for retrieval; use plain "
                    "`embed` for symmetric text-to-text comparison.\n"
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
