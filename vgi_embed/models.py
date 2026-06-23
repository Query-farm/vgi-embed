"""Model lifecycle: load a fastembed ONNX model once, cache it in the worker process.

VGI keeps the worker process alive across queries, so the expensive thing an
embedding worker does -- loading the ONNX model (and, on first use ever,
*downloading* it) -- happens once and is amortised over every row of every query.
This module centralises that caching: callers just ask for "the embedder for
model X" and get a ready ``fastembed.TextEmbedding`` back.

Why fastembed
-------------
`fastembed` (Qdrant, Apache-2.0) runs sentence-transformer models through ONNX
Runtime -- **no torch**. On first use it downloads a small, quantised ONNX model
to a local cache and reuses it forever after. That makes the worker light to
install and fast to start once the model is cached.

Default model
-------------
``BAAI/bge-small-en-v1.5`` -- 384-dim, **MIT licensed**, strong general-purpose
English retrieval/semantic-search embeddings. Downloaded on first use to the
fastembed cache dir (``~/.cache/...`` by default, or ``VGI_EMBED_CACHE_DIR`` /
``FASTEMBED_CACHE_PATH`` -- see :func:`_cache_dir`). The cache is gitignored.

Retrieval prefixes
------------------
bge retrieval models recommend prefixing *queries* (not passages) with an
instruction. For ``bge-*`` we apply::

    Represent this sentence for searching relevant passages: <query>

to queries only; passages are embedded as-is. ``embed_query`` / ``embed_passage``
expose that asymmetry; plain ``embed`` applies no prefix (symmetric similarity).

Everything here is lazy: importing this module is cheap; nothing is loaded or
downloaded until the first row needs it (or :func:`warm_up` is called at startup).
A model that cannot be loaded raises a clear, actionable error rather than a deep
library traceback.
"""

from __future__ import annotations

import contextlib
import math
import os
import threading
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastembed import TextEmbedding


# ---------------------------------------------------------------------------
# Supported models. Keyed by the name users pass to embed(text, model) and to
# embedding_dim(model). Each entry: (output dimension, query-instruction prefix).
# The prefix is applied by embed_query only; embed/embed_passage never prefix.
# ---------------------------------------------------------------------------

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# (dim, query_prefix). All are fastembed-supported ONNX models with permissive
# licenses; bge-small-en-v1.5 is the default (MIT, 384-dim).
_SUPPORTED_MODELS: dict[str, tuple[int, str]] = {
    "BAAI/bge-small-en-v1.5": (384, _BGE_QUERY_PREFIX),
    "BAAI/bge-base-en-v1.5": (768, _BGE_QUERY_PREFIX),
    "BAAI/bge-small-en": (384, _BGE_QUERY_PREFIX),
    "sentence-transformers/all-MiniLM-L6-v2": (384, ""),
}

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_CACHE_DIR_ENV = "VGI_EMBED_CACHE_DIR"

_lock = threading.Lock()


class ModelNotAvailableError(RuntimeError):
    """A requested embedding model is unknown or could not be loaded/downloaded.

    Carries an actionable hint (the supported model list, or that the first use
    needs network access to download) so the DuckDB-side error tells the user how
    to fix it.
    """


def supported_models() -> list[tuple[str, int]]:
    """Every ``(model, dim)`` the worker can produce, sorted by model name."""
    return sorted((name, dim) for name, (dim, _prefix) in _SUPPORTED_MODELS.items())


def resolve_model(model: str | None) -> str:
    """Normalise a requested model name, defaulting empty/None to the default."""
    name = (model or "").strip() or DEFAULT_MODEL
    if name not in _SUPPORTED_MODELS:
        raise ModelNotAvailableError(
            f"Unknown embedding model {name!r}. Supported models: {', '.join(sorted(_SUPPORTED_MODELS))}."
        )
    return name


def embedding_dim(model: str | None) -> int:
    """Output dimension for ``model`` (defaulting empty/None to the default model)."""
    return _SUPPORTED_MODELS[resolve_model(model)][0]


def query_prefix(model: str | None) -> str:
    """The query-instruction prefix for ``model`` ('' if the model uses none)."""
    return _SUPPORTED_MODELS[resolve_model(model)][1]


def _cache_dir() -> str | None:
    """Where fastembed should cache downloaded ONNX models.

    ``VGI_EMBED_CACHE_DIR`` wins; otherwise we honour fastembed's own
    ``FASTEMBED_CACHE_PATH``; otherwise ``None`` lets fastembed pick its default
    (a temp/cache dir under the user's home). The dir is created on demand.
    """
    explicit = os.environ.get(_CACHE_DIR_ENV) or os.environ.get("FASTEMBED_CACHE_PATH")
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    return None


@cache
def _load_model(model_name: str) -> TextEmbedding:
    """Load (and cache) a fastembed ``TextEmbedding`` by name.

    First-ever use downloads the quantised ONNX model to the fastembed cache; all
    later worker processes that share the cache load it from disk. A download
    failure (e.g. offline on a cold cache) is surfaced as a clear error.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - dependency always present in prod
        raise ModelNotAvailableError("fastembed is not installed. Install it with: uv pip install fastembed") from exc

    try:
        return TextEmbedding(model_name=model_name, cache_dir=_cache_dir())
    except Exception as exc:  # noqa: BLE001 - turn any backend failure into an actionable error
        raise ModelNotAvailableError(
            f"Could not load embedding model {model_name!r}. The model is downloaded "
            f"on first use, so this needs network access on a cold cache; afterwards it "
            f"is served from the fastembed cache "
            f"(override with {_CACHE_DIR_ENV}). Original error: {exc}"
        ) from exc


def get_model(model: str | None) -> TextEmbedding:
    """Get the cached fastembed embedder for ``model`` (thread-safe first load)."""
    name = resolve_model(model)
    with _lock:
        return _load_model(name)


def embed_texts(texts: list[str], *, model: str | None, prefix: str = "") -> list[list[float]]:
    """Embed a list of (already non-empty) strings, returning one vector each.

    ``prefix`` is prepended to every text (used by ``embed_query`` to apply the
    model's query instruction). Order is preserved. The caller is responsible for
    masking out NULL/empty rows and re-inserting NULLs.
    """
    if not texts:
        return []
    embedder = get_model(model)
    prepared = [prefix + t for t in texts] if prefix else texts
    # fastembed yields numpy arrays; convert to plain Python float lists for Arrow.
    return [vec.tolist() for vec in embedder.embed(prepared)]


# ---------------------------------------------------------------------------
# Pure cosine similarity (no model)
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float | None:
    """Cosine similarity of two vectors in [-1, 1]; ``None`` on NULL/empty/mismatch.

    Pure arithmetic -- never touches a model. Returns ``None`` (rather than raising)
    for NULL inputs, empty vectors, length mismatches, or a zero-magnitude vector,
    so it is robust to odd input straight out of SQL.
    """
    if a is None or b is None:
        return None
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return None
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        # A NULL element inside the list makes the whole comparison undefined.
        if x is None or y is None:
            return None
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------------------------------------------------------
# Startup warm-up
# ---------------------------------------------------------------------------


def warm_up() -> None:
    """Load (and, if needed, download) the default model once at worker startup.

    Everything in this module is lazy by design, so the *first* query of every
    ATTACH otherwise pays the model load -- and on a cold cache the multi-second
    *download* -- inline. Under the end-to-end SQL suite that happens while the
    runner is mid-assertion on the first file: a long window in which a worker-pool
    teardown SIGTERM (or a heavily-loaded host) can kill the run and record a
    spurious failure, even though every embedding is deterministic.

    Warming here moves that one-time cost to process spawn (before any query), so
    each per-file first query is fast and the vulnerable window shrinks to near
    zero. It only populates the existing cache -- it never changes any output.
    Best-effort: if the model can't be loaded (e.g. offline on a cold cache) it is
    not fatal here -- the function that needs it will raise its own actionable
    error if actually invoked, so a worker still starts cleanly.
    """
    with contextlib.suppress(Exception):
        # Touch the embedder so the ONNX session is built and cached now.
        embedder = _load_model(DEFAULT_MODEL)
        next(iter(embedder.embed(["warm up"])), None)
