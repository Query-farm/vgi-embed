# Copyright 2026 Query Farm LLC - https://query.farm
"""Prepare hash-verified immutable ONNX/tokenizer assets for the unchanged worker."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen


class EmbedAssets:
    """Download exact model bytes, verify their hashes and configure offline worker loading."""

    def __init__(self):
        """Prepare only the pinned model snapshot; preserve existing environment settings."""
        directory = Path(__file__).resolve().parent
        self.manifest = json.loads((directory / "model-manifest.json").read_text())
        root = directory / "validation/model-cache"
        repository = self.manifest["repository"]
        revision = self.manifest["revision"]
        model_root = root / ("models--" + repository.replace("/", "--"))
        snapshot = model_root / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, expected in self.manifest["files"].items():
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and self._digest(destination) == expected["sha256"]:
                continue
            url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=snapshot, delete=False) as output:
                    temporary = Path(output.name)
                    with urlopen(url, timeout=90) as response:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                if (
                    temporary.stat().st_size != expected["bytes"]
                    or self._digest(temporary) != expected["sha256"]
                ):
                    raise RuntimeError(f"Pinned embedding asset failed verification: {name}")
                temporary.replace(destination)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        refs = model_root / "refs"
        refs.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=refs, mode="w", delete=False) as output:
            output.write(revision)
            temporary = Path(output.name)
        temporary.replace(refs / "main")
        settings = {
            "VGI_EMBED_DEFAULT_MODEL": self.manifest["model_name"],
            "VGI_EMBED_CACHE_DIR": str(root),
            "FASTEMBED_CACHE_PATH": str(root),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        self.previous = {name: os.environ.get(name) for name in settings}
        os.environ.update(settings)

    @staticmethod
    def _digest(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    def close(self):
        """Restore the caller's environment after closing its worker connection."""
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
