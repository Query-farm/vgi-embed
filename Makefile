# vgi-embed worker -- dev and test targets.
#
# Usage:
#   make test        # unit (pytest) + SQL (end-to-end via haybarn-unittest)
#   make test-unit   # pytest only
#   make test-sql    # SQL end-to-end only (runs the haybarn glob)
#   make models      # warm the fastembed cache (download the default ONNX model)
#
# The SQL suite drives the worker as a real subprocess over stdio: haybarn-unittest
# ATTACHes `${VGI_EMBED_WORKER}`, then runs the .test files in test/sql/.
#
# The default model (BAAI/bge-small-en-v1.5, MIT, 384-dim) is downloaded by
# fastembed on first use and cached (gitignored). `make models` pre-warms it so
# the first SQL/unit run isn't paying the download inline.

# Worker stdio command (overridable). The PEP-723 header in embed_worker.py pins
# fastembed, so `uv run` gives the worker its dependency.
WORKER_STDIO   ?= uv run --python 3.13 embed_worker.py

# haybarn-unittest: the DuckDB sqllogictest runner (uv tool install haybarn-unittest).
HAYBARN        ?= haybarn-unittest
TEST_DIR        = .
TEST_PATTERN    = test/sql/*

.PHONY: test test-unit test-sql pytest models lint typecheck

test: test-unit test-sql

test-unit: pytest

pytest:
	uv run --no-sync pytest -q

# End-to-end SQL: run the haybarn glob with the worker command exported.
# `uv run embed_worker.py` resolves fastembed from the script's pinned deps.
test-sql:
	@command -v $(HAYBARN) >/dev/null 2>&1 || { \
		echo "ERROR: $(HAYBARN) not found. Install it with:" >&2; \
		echo "  uv tool install haybarn-unittest" >&2; \
		echo "  (then ensure ~/.local/bin is on PATH)" >&2; \
		exit 1; \
	}
	VGI_EMBED_WORKER="$(WORKER_STDIO)" $(HAYBARN) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

# Pre-warm the fastembed cache by downloading the default ONNX model once.
models:
	uv run --no-sync python -c "from vgi_embed import models; models.warm_up(); print('default model cached:', models.DEFAULT_MODEL)"

lint:
	uv run --no-sync ruff check .

typecheck:
	uv run --no-sync mypy vgi_embed/
