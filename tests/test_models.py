"""Worker-startup configuration and model metadata tests without model downloads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from vgi_embed import models


@pytest.mark.parametrize(
    ("configured", "expected", "dim"),
    [
        (None, "BAAI/bge-small-en-v1.5", 384),
        ("", "BAAI/bge-small-en-v1.5", 384),
        ("  ", "BAAI/bge-small-en-v1.5", 384),
        (" BAAI/bge-base-en-v1.5 ", "BAAI/bge-base-en-v1.5", 768),
        ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", 384),
    ],
)
def test_default_model_configuration(configured: str | None, expected: str, dim: int) -> None:
    env = os.environ.copy()
    env.pop("VGI_EMBED_DEFAULT_MODEL", None)
    if configured is not None:
        env["VGI_EMBED_DEFAULT_MODEL"] = configured
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from vgi_embed import models; from vgi_embed.scalars import Embed; "
            "print(json.dumps([models.resolve_model(None), models.embedding_dim(''), "
            "models.resolve_model('BAAI/bge-small-en-v1.5'), Embed.Meta.description, "
            "json.loads(Embed.Meta.tags['vgi.executable_examples'])]))",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    name, actual_dim, explicit, description, examples = json.loads(result.stdout)
    assert (name, actual_dim) == (expected, dim)
    assert explicit == "BAAI/bge-small-en-v1.5"
    assert expected in description
    assert f"{dim}-dim" in description
    assert all(example["expected_result"] == [{"dim": dim}] for example in examples)


def test_invalid_default_reports_supported_models() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from vgi_embed.worker import EmbedWorker"],
        env={**os.environ, "VGI_EMBED_DEFAULT_MODEL": "no/such-model"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Unknown embedding model 'no/such-model'" in result.stderr
    assert "Supported models:" in result.stderr


def test_warm_up_uses_configured_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "DEFAULT_MODEL", "BAAI/bge-base-en-v1.5")
    with patch.object(models, "_load_model") as load:
        load.return_value.embed.return_value = iter([])
        models.warm_up()
    load.assert_called_once_with("BAAI/bge-base-en-v1.5")
