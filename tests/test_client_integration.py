"""End-to-end tests through ``vgi.client.Client``, spawning the real worker.

These exercise the full Arrow-IPC round trip the way DuckDB would: the worker
runs as a subprocess and we drive it over stdin/stdout. Gated on the default
fastembed model being available, so a bare/offline checkout stays green.
"""

from __future__ import annotations

import math
import os
import shlex
import sys

import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client

from tests.harness import model_available

_WORKER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embed_worker.py")

needs_model = pytest.mark.skipif(
    not model_available(), reason="fastembed default model not available (offline / cold cache)"
)


def _client() -> Client:
    # Launch the worker with the same interpreter running the tests so it sees the
    # installed deps (rather than going through `uv run`). Client wants a
    # shell-style command string.
    return Client(f"{shlex.quote(sys.executable)} {shlex.quote(_WORKER)}")


@needs_model
def test_embed_scalar_end_to_end() -> None:
    # New-API scalars bind their column Params from the input batch by position;
    # only ConstParam args go in arguments.positional. `embed(text)` has no
    # ConstParam, so positional is empty (passing the column name would route to
    # the `embed(text, model)` overload and treat it as a model name).
    batch = pa.RecordBatch.from_pydict({"text": ["hello world", None]})
    with _client() as client:
        results = list(
            client.scalar_function(
                function_name="embed",
                input=iter([batch]),
                arguments=Arguments(positional=[]),
            )
        )
    vectors = results[0]["result"].to_pylist()
    assert len(vectors[0]) == 384
    assert vectors[1] is None


@needs_model
def test_embed_with_explicit_model_overload_end_to_end() -> None:
    # The 2-arity overload: the ConstParam model name goes in positional.
    batch = pa.RecordBatch.from_pydict({"text": ["hello world"]})
    with _client() as client:
        results = list(
            client.scalar_function(
                function_name="embed",
                input=iter([batch]),
                arguments=Arguments(positional=[pa.scalar("BAAI/bge-small-en-v1.5")]),
            )
        )
    assert len(results[0]["result"].to_pylist()[0]) == 384


@needs_model
def test_similarity_self_is_one_end_to_end() -> None:
    batch = pa.RecordBatch.from_pydict({"text": ["a quick brown fox"]})
    with _client() as client:
        emb = list(
            client.scalar_function(
                function_name="embed",
                input=iter([batch]),
                arguments=Arguments(positional=[]),
            )
        )
        vec = emb[0]["result"]
        sim_batch = pa.RecordBatch.from_arrays([vec, vec], names=["a", "b"])
        sim = list(
            client.scalar_function(
                function_name="similarity",
                input=iter([sim_batch]),
                arguments=Arguments(positional=[]),
            )
        )
    assert math.isclose(sim[0]["result"].to_pylist()[0], 1.0, abs_tol=1e-5)
