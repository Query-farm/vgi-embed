"""Discovery table function for the embed worker.

``supported_models()`` expands to **many rows** (one per known model), so it is a
**table function** -- the form that accepts DuckDB ``name := value`` arguments
(this one takes none, but the table-function shape is still its right home). The
per-row embedding/similarity functions are *scalars* and live in
:mod:`vgi_embed.scalars`.

    SELECT * FROM embed.supported_models() ORDER BY model;
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pyarrow as pa
from vgi.metadata import FunctionExample
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi_rpc.rpc import OutputCollector

from . import models
from .schema_utils import field


@dataclass(kw_only=True)
class _NoArgs:
    """A discovery table function that takes no arguments."""


_SUPPORTED_MODELS_SCHEMA = pa.schema(
    [
        field("model", pa.string(), "Model name to pass to embed(text, model).", nullable=False),
        field("dim", pa.int32(), "Embedding dimension (FLOAT[] length) the model produces.", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class SupportedModelsFunction(TableFunctionGenerator[_NoArgs]):
    """Every ``(model, dim)`` the worker can produce, one per row.

    ``model`` is the value you pass as the optional second argument to
    ``embed(text, model)`` (and to ``embedding_dim(model)``); ``dim`` is the
    length of the FLOAT[] vector it returns.
    """

    FIXED_SCHEMA: ClassVar[pa.Schema] = _SUPPORTED_MODELS_SCHEMA

    class Meta:
        """Declarative metadata for the ``supported_models()`` table function."""

        name = "supported_models"
        description = "Every (model, dim) the embed worker supports"
        categories = ["embedding", "metadata"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM embed.supported_models() ORDER BY model",
                description="List the supported embedding models and their dimensions",
            ),
            FunctionExample(
                sql="SELECT dim FROM embed.supported_models() WHERE model = 'BAAI/bge-small-en-v1.5'",
                description="Dimension of the default model",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_NoArgs]) -> TableCardinality:
        """Exact row count: one per supported model."""
        n = len(models.supported_models())
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[_NoArgs], state: None, out: OutputCollector) -> None:
        """Emit one ``(model, dim)`` row per supported model, then finish."""
        rows = models.supported_models()
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "model": [r[0] for r in rows],
                    "dim": [r[1] for r in rows],
                },
                schema=params.output_schema,
            )
        )
        out.finish()


TABLE_FUNCTIONS: list[type] = [
    SupportedModelsFunction,
]
