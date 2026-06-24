# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.4",
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

from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_embed import models
from vgi_embed.scalars import SCALAR_FUNCTIONS
from vgi_embed.tables import TABLE_FUNCTIONS

_EMBED_CATALOG = Catalog(
    name="embed",
    default_schema="main",
    comment="Local text embeddings (fastembed/ONNX) + cosine similarity for semantic search / RAG.",
    tags={
        "vgi.title": "Local Text Embeddings & Similarity",
        "vgi.keywords": (
            "embeddings, embed, text embedding, vector, fastembed, onnx, cosine "
            "similarity, semantic search, retrieval, rag, bge, sentence transformer, "
            "nearest neighbor, vss"
        ),
        "vgi.description_llm": (
            "Turn text into fixed-length FLOAT[] embedding vectors entirely in-process "
            "(fastembed/ONNX, no torch, no network) and compare them with cosine similarity. "
            "Use embed(text) for symmetric embeddings, embed_query(text)/embed_passage(text) "
            "for retrieval asymmetry, similarity(a, b) to score two vectors, and "
            "supported_models() to discover available models. Pairs with DuckDB VSS for "
            "semantic search and RAG."
        ),
        "vgi.description_md": (
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
                "vgi.keywords": (
                    "embed, embed_query, embed_passage, similarity, embedding_dim, "
                    "embed_version, supported_models, embeddings, vector, cosine, "
                    "semantic search, retrieval, rag"
                ),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "machine-learning",
                "category": "embeddings",
                "topic": "semantic-search",
                "vgi.source_url": "https://github.com/Query-farm/vgi-embed/blob/main/embed_worker.py",
                "vgi.description_llm": (
                    "Local text-embedding and similarity functions: embed text into FLOAT[] "
                    "vectors with the default or a chosen model, apply retrieval query/passage "
                    "prefixes, compute cosine similarity between vectors, look up a model's "
                    "embedding dimension, and list the supported models."
                ),
                "vgi.description_md": ("Local text-embedding and cosine-similarity functions over Apache Arrow."),
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
