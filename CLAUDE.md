# CLAUDE.md — vgi-embed

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://query.farm) worker exposing **local text embeddings** to
DuckDB/SQL for semantic search / RAG with DuckDB VSS. `embed_worker.py` assembles
every function into one `embed` catalog (single `main` schema) and runs it over
stdio. Embeddings are generated locally via `fastembed` (Qdrant, Apache-2.0) on
**ONNX Runtime — no torch**; the default model is `BAAI/bge-small-en-v1.5`
(384-dim, **MIT**).

## Layout

```
embed_worker.py        repo-root stdio entry; PEP 723 inline deps; warms the model then serves; main()
serve.py               HTTP entry shim
vgi_embed/
  models.py            loaded-once-and-cached fastembed lifecycle + warm_up(); pure cosine_similarity; supported-model table
  scalars.py           7 scalar functions (embed/embed+model arity overload, embed_query/passage, similarity, embedding_dim, embed_version)
  tables.py            supported_models() discovery table function
  schema_utils.py      pa.Field comment helper
tests/                 pytest: scalars / tables / Client integration (model-gated tests self-skip)
test/sql/*.test        haybarn-unittest sqllogictest — authoritative E2E
Makefile               test / test-unit / test-sql / models / lint / typecheck
```

## Core conventions (read first)

- **Scalars are positional-only** (no `name := value`). `embed`'s optional
  `model` is therefore a second **arity overload** (`embed(text)` /
  `embed(text, model)`) sharing one name — same idiom as vgi-nlp's `lemmatize`.
- **`FLOAT[]` returns need an explicit return type.** The embedding scalars
  declare `Returns(arrow_type=pa.list_(pa.float32()))`; `similarity` declares
  `Returns(arrow_type=pa.float64())`. Without the explicit `arrow_type`, a
  `LIST(FLOAT)` output can't be inferred.
- **`supported_models()` is a table function** (rows out), so it lives in
  `tables.py` with the `@init_single_worker` / `@bind_fixed_schema` pattern.
- **NULL/empty → NULL.** Every embedding scalar masks NULL and
  whitespace-only rows to a NULL vector *before* calling the model, and splices
  them back by index — so the model is only ever handed real strings, and an
  all-empty input batch never loads the model at all.

## Sharp edges

1. **Expensive init: load the model ONCE.** `models._load_model` is
   `@lru_cache`'d and guarded by a lock; the whole point is that VGI keeps the
   worker alive so the ONNX session is built once and amortised over every row.
   Never construct `TextEmbedding` per call.
2. **First use downloads the model.** `fastembed` fetches a quantised ONNX model
   on first use and caches it (gitignored). On a **cold cache offline**, model
   calls raise an actionable `ModelNotAvailableError`. `warm_up()` (called at
   worker spawn from `EmbedWorker.run`) moves that download/load off the first
   query; it's best-effort and never fatal. `make models` pre-warms for dev/CI.
3. **`haybarn-unittest` skips `require vgi`** — use explicit `statement ok` /
   `LOAD vgi;` in `.test` files (the ones here do).
4. **Determinism in assertions.** Embeddings are deterministic per model, but
   exact float values vary by ONNX build/platform. Assert the vector **length**
   exactly, `similarity(embed(a), embed(a)) ≈ 1.0` (via `ROUND(..., 3) = 1.0`),
   and a planted **related > unrelated** comparison (`dog/puppy > dog/airplane`).
   Never assert exact floats.
5. **Model-gated tests.** Unit/integration tests that actually embed are guarded
   by `@needs_model` (default model loadable) so a bare/offline checkout stays
   green; the pure-logic tests (`similarity`, `embedding_dim`, `supported_models`,
   NULL masking) always run.

## Testing

```sh
uv run --no-sync pytest -q     # unit (model-gated tests self-skip on a cold/offline checkout)
make models                    # pre-warm the fastembed cache for local dev
make test-sql                  # E2E: haybarn-unittest over test/sql/*  (authoritative)
make test                      # both
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_embed/
```

`make test-sql` exports `VGI_EMBED_WORKER="uv run --python 3.13 embed_worker.py"`
and runs `haybarn-unittest --test-dir . "test/sql/*"` (install once:
`uv tool install haybarn-unittest`). **The SQL suite is authoritative** — it
exercises the real RPC + model path. CI runs unit + lint plus a gated `e2e` job
(installs worker deps from PyPI, warms the model, launches the worker from the
prepared venv).
```
