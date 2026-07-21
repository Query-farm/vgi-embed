# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.16.0",
#     "fastembed>=0.3",
# ]
# ///
"""Thin PEP 723 stdio shim for the embed worker.

The catalog assembly, :class:`~vgi_embed.worker.EmbedWorker`, and ``main()`` now
live in the wheel-importable :mod:`vgi_embed.worker` module (so the built wheel
contains the worker, and the console script ``vgi-embed-worker`` and the Docker
HTTP entrypoint can import it). This repo-root script re-exports them and keeps
the inline PEP 723 dependency pins, so ``uv run embed_worker.py`` (used by the
Makefile, ``ci/run-integration.sh``, and the tests as the stdio ATTACH command)
works exactly as before.

Usage:
    uv run embed_worker.py               # serve over stdio (DuckDB subprocess)
    python serve.py --port 8000          # serve over HTTP

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'embed' (TYPE vgi, LOCATION 'uv run embed_worker.py');

    SELECT embed.embed('hello world');                          -- FLOAT[384]
    SELECT embed.similarity(embed.embed('cat'), embed.embed('kitten'));
    SELECT * FROM embed.supported_models();
"""

from __future__ import annotations

from vgi_embed.worker import EmbedWorker, main

__all__ = ["EmbedWorker", "main"]


if __name__ == "__main__":
    main()
