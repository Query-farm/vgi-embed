# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python",
#     "fastembed>=0.3",
# ]
#
# [tool.uv.sources]
# vgi-python = { path = "../vgi-python" }
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
    schemas=[
        Schema(
            name="main",
            comment="Local text embeddings (fastembed/ONNX) + cosine similarity for SQL",
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
