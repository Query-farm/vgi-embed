# CI: the vgi-embed worker integration suite

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the unit tests
and this repo's sqllogictest suite (`test/sql/*.test`) against the vgi-embed
VGI worker through the **real DuckDB `vgi` extension** on every push / PR.

## How it works (no C++ build)

Rather than building the vgi DuckDB extension from source, CI drives a
**prebuilt** standalone `haybarn-unittest` (the DuckDB/Haybarn sqllogictest
runner, published in Haybarn's releases) and installs the **signed** `vgi`
extension from the Haybarn community channel:

1. **Install the worker** — `uv sync --frozen` into a venv. `embed_worker.py`
   is a self-contained PEP 723 stdio worker the extension can spawn via
   `uv run --no-sync --python 3.13 embed_worker.py`.
2. **Download the runner** — the matching `haybarn_unittest-*` asset per
   platform from the latest Haybarn release.
3. **Preprocess** — the standalone runner links none of the extensions the
   tests gate on, so [`preprocess-require.awk`](preprocess-require.awk) rewrites
   each `require <ext>` into an explicit signed `INSTALL <ext> FROM
   {community,core}; LOAD <ext>;`. These tests skip `require vgi` (haybarn
   silently SKIPs it) and `LOAD vgi;` directly, so the awk also injects an
   `INSTALL vgi FROM community;` right before each bare `LOAD vgi;`. `require-env`
   and everything else pass through untouched.
4. **Run** — [`run-integration.sh`](run-integration.sh) stages the preprocessed
   tree, resolves `VGI_EMBED_WORKER` per `$TRANSPORT`, warms the extension cache
   once, then runs the suite in a single `haybarn-unittest` invocation. Any
   failed assertion exits non-zero and fails the job.

## Transports (subprocess / http / unix)

The same suite runs over every VGI transport. The vgi extension picks the
transport from the ATTACH `LOCATION` string, which `run-integration.sh` builds
from `$TRANSPORT` (default `subprocess`):

- **subprocess** — `VGI_EMBED_WORKER` is the stdio command (`uv run --no-sync
  --python 3.13 embed_worker.py`); the extension spawns the worker per query and
  talks Arrow IPC over stdin/stdout.
- **http** — the script boots `embed_worker.py --http --port 0 --port-file <f>`
  with cwd = the stage dir, polls the port-file (generous 180s timeout — the
  worker warms its ONNX model at spawn), and sets `VGI_EMBED_WORKER` to the bare
  `http://127.0.0.1:<port>` (no path suffix). HTTP mode needs the `http` extra
  (waitress) — `pyproject.toml` lists `vgi-python[http]`, the PEP 723 header in
  `embed_worker.py` does too, and the integration job installs `--extra http`.
  Over `http://` the vgi extension routes worker-RPC through DuckDB's httpfs, so
  the script injects `INSTALL httpfs FROM core; LOAD httpfs;` after each `LOAD
  vgi;` in the staged tests (http leg only).
- **unix** — the script boots `embed_worker.py --unix <sock>` with cwd = the
  stage dir, polls for the socket, and sets `VGI_EMBED_WORKER` to
  `unix://<sock>`.

The runner SILENTLY SKIPS (exit 0) any test whose error message contains "HTTP"
or "Unable to connect", so a broken http setup would otherwise fake-pass with
"All tests were skipped". The script guards against that: it captures the run
log and fails the leg if every test was skipped.

CI ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs the
`integration` job as a `{os} x {transport}` matrix
(`[ubuntu-latest, macos-latest] x [subprocess, http, unix]`).

## Run it locally

```bash
uv sync --python 3.13                       # install the worker + deps
# point HAYBARN_UNITTEST at a haybarn-unittest binary (or a local DuckDB
# `unittest` built with the vgi extension), and the worker at the stdio command:
HAYBARN_UNITTEST=/path/to/haybarn-unittest \
VGI_EMBED_WORKER="uv run --no-sync --python 3.13 embed_worker.py" \
  ci/run-integration.sh
```

Or use the Makefile target `make test-sql`, which installs `haybarn-unittest`
as a uv tool and points the worker at `uv run --no-sync --python 3.13 embed_worker.py`.
