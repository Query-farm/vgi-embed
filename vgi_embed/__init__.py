"""vgi-embed: local text embeddings (fastembed/ONNX) as DuckDB SQL functions.

Exposes ``embed`` (and ``embed_query`` / ``embed_passage``), ``similarity``,
``embedding_dim``, ``embed_version``, and the ``supported_models()`` discovery
table function for semantic search / RAG with DuckDB VSS.
"""

from __future__ import annotations

__version__ = "0.1.0"
