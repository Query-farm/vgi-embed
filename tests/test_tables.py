"""Integration test for the ``supported_models()`` discovery table function.

Drives it through the real bind -> init -> process lifecycle in-process (no worker
subprocess). No model is loaded -- the catalog of supported models is static -- so
this always runs.
"""

from __future__ import annotations

from vgi_embed import models
from vgi_embed.tables import SupportedModelsFunction

from .harness import invoke_table_function


class TestSupportedModels:
    def test_columns_and_nonempty(self) -> None:
        table = invoke_table_function(SupportedModelsFunction)
        assert table.column_names == ["model", "dim"]
        assert table.num_rows >= 1

    def test_default_model_present_with_384_dim(self) -> None:
        table = invoke_table_function(SupportedModelsFunction)
        mapping = dict(
            zip(table.column("model").to_pylist(), table.column("dim").to_pylist(), strict=True)
        )
        assert mapping[models.DEFAULT_MODEL] == 384

    def test_matches_models_module(self) -> None:
        table = invoke_table_function(SupportedModelsFunction)
        rows = list(zip(table.column("model").to_pylist(), table.column("dim").to_pylist(), strict=True))
        assert rows == models.supported_models()

    def test_sorted_by_model(self) -> None:
        table = invoke_table_function(SupportedModelsFunction)
        names = table.column("model").to_pylist()
        assert names == sorted(names)
