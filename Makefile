# vgi-embed worker -- dev and test targets.
#
# Usage:
#   make test        # unit (pytest) + SQL (end-to-end via haybarn-unittest)
#   make test-unit   # pytest only
#   make test-sql    # SQL end-to-end only (runs the haybarn glob)
#   make models      # warm the fastembed cache (download the default ONNX model)

# SQL tests: ci/run-integration.sh (adapted from the vgi-starter template) runs
# test/sql/*.test through the published haybarn-unittest with the signed community
# `vgi` extension, over stdio, launch (AF_UNIX, pooled) and http. CI runs the same
# script per transport. `make test-sql TRANSPORTS=launch` runs one.
HAYBARN_VERSION  ?= 2.0.1rc3
WORKER           ?= $(CURDIR)/.venv/bin/vgi-embed-worker
TOOLS            := $(CURDIR)/.tools
HAYBARN_UNITTEST := $(TOOLS)/bin/haybarn-unittest
TRANSPORTS       ?= stdio launch http

.PHONY: build tools check test test-unit test-sql pytest models lint typecheck

build: ## Install the worker (with every extra) into .venv
	uv sync --locked --all-extras

tools: ## Install the pinned SQL test runner into .tools
	@test -x $(HAYBARN_UNITTEST) || { uv venv -q $(TOOLS) && uv pip install -q -p $(TOOLS) haybarn-unittest==$(HAYBARN_VERSION); }

test-sql: build tools ## SQL tests through DuckDB over each of $(TRANSPORTS)
	@for t in $(TRANSPORTS); do \
		WORKER="$(WORKER)" HAYBARN_UNITTEST="$(HAYBARN_UNITTEST)" TRANSPORT=$$t ci/run-integration.sh || exit 1; \
	done

test: test-unit test-sql

test-unit: pytest

pytest:
	uv run --no-sync pytest -q

# Pre-warm the fastembed cache by downloading the default ONNX model once.
models:
	uv run --no-sync python -c "from vgi_embed import models; models.warm_up(); print('default model cached:', models.DEFAULT_MODEL)"

lint:
	uv run --no-sync ruff check .

typecheck:
	uv run --no-sync mypy vgi_embed/

check: lint test ## Everything CI runs
