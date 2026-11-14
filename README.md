<p align="center">
  <img src="https://query.farm/media-kit/products/vgi/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# Local Text Embeddings for RAG in DuckDB

> **vgi-embed** · a [Query.Farm](https://query.farm) VGI worker · powered by fastembed (ONNX)

Local **text embeddings** as DuckDB SQL functions — for semantic search and
retrieval-augmented generation (RAG) with [DuckDB VSS](https://duckdb.org/docs/extensions/vss).

A [VGI](https://query.farm) worker that turns text into fixed-length `FLOAT[]`
vectors entirely **on your machine** — no API keys, no network at query time. It
runs sentence-transformer models through [`fastembed`](https://github.com/qdrant/fastembed)
(Qdrant, Apache-2.0), which uses **ONNX Runtime — no torch**, so it installs light
and starts fast.

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'embed' (TYPE vgi, LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-embed@v0.5.0 vgi-embed-worker');

-- A 384-dimensional vector.
SELECT embed.embed('hello world');

-- Cosine similarity (pure, no model).
SELECT embed.similarity(embed.embed('cat'), embed.embed('kitten'));

-- Top-k semantic search over a column.
SELECT id, body
FROM docs
ORDER BY embed.similarity(embed.embed_passage(body),
                          embed.embed_query('how do I reset my password')) DESC
LIMIT 5;

-- What models are available?
SELECT * FROM embed.supported_models();
```

## Quick start

The worker isn't on PyPI; [uv](https://docs.astral.sh/uv/) runs it straight from a tagged release on GitHub. `uvx` builds an isolated environment on the first ATTACH and reuses it afterwards:

```sql
INSTALL vgi FROM community;
LOAD vgi;

ATTACH 'embed' AS embed (TYPE vgi, LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-embed@v0.5.0 vgi-embed-worker');

-- Embed text into a FLOAT[] vector and check its length.
SELECT len(embed.main.embed('hello world')) AS dim;

-- Embed text with an explicitly chosen model.
SELECT len(embed.main.embed('hello world', 'BAAI/bge-base-en-v1.5')) AS dim;

-- Zero-shot classification: label each ticket with its most similar category.
WITH tickets(ticket) AS (VALUES
    ('The app crashes every time I open the settings screen.'),
    ('I was charged twice for my subscription this month.'),
    ('Could you add a dark mode option?')),
labels(label) AS (VALUES ('bug report'), ('billing problem'), ('feature request'))
SELECT t.ticket,
       arg_max(l.label, embed.main.similarity(embed.main.embed(t.ticket), embed.main.embed(l.label))) AS label
FROM tickets AS t CROSS JOIN labels AS l
GROUP BY t.ticket ORDER BY t.ticket;
```

The embedding model downloads on first use and is cached — see [The model](#the-model).

Change `@v0.5.0` to move to a newer release.

### Other ways to run it

- **Docker** (the image bundles the worker and its dependencies):

  ```sql
  ATTACH 'embed' AS embed (TYPE vgi, LOCATION 'docker://ghcr.io/query-farm/vgi-embed:0.5.0');
  ```

- **From a local checkout** (`git clone https://github.com/Query-farm/vgi-embed`), point `LOCATION` at `uv run --project /path/to/vgi-embed vgi-embed-worker`.

### Build from source

To build the worker yourself, see [Development](#development).

## The model

| | |
|---|---|
| **Default model** | `BAAI/bge-small-en-v1.5` |
| **Dimension** | 384 (`FLOAT[384]`) |
| **Model license** | **MIT** (commercial use permitted) — see the [model card](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| **Runtime** | `fastembed` (Apache-2.0) on ONNX Runtime — no torch |

The model is **downloaded on first use** (a small quantised ONNX file) and cached
on disk; later runs load it locally. The cache directory is gitignored. Override
it with `VGI_EMBED_CACHE_DIR` (or fastembed's own `FASTEMBED_CACHE_PATH`).
Pre-warm it once with `make models`.

Choose a model with the second positional argument to `embed`, `embed_query`, or
`embed_passage`. Use the **same model for queries and documents**; matching vector
dimensions alone does not make different models compatible.

```sql
SELECT embed.similarity(
    embed.embed_query('reset password', 'BAAI/bge-base-en-v1.5'),
    embed.embed_passage('Click forgot password to reset it.', 'BAAI/bge-base-en-v1.5')
);
```

Set `VGI_EMBED_DEFAULT_MODEL` in the worker's environment before startup to change
the shared default for all three functions, `embedding_dim('')`, and model
warm-up. For example:

```sh
VGI_EMBED_DEFAULT_MODEL=BAAI/bge-base-en-v1.5 uv run vgi-embed-worker
```

Explicit model arguments override this setting. Omitting the argument or passing
an empty/whitespace-only model name uses the configured default. An unset or blank
environment variable retains `BAAI/bge-small-en-v1.5`; an unsupported default
produces an error at worker startup. Restart the worker after changing this setting.
Stored vectors and query vectors must still use the same model; re-embed existing
data when switching models.

Supported models (pass as the second argument to any embedding function, or query
`supported_models()`):

| model | dim | license |
|---|---|---|
| `BAAI/bge-small-en-v1.5` (default) | 384 | MIT |
| `BAAI/bge-base-en-v1.5` | 768 | MIT |
| `BAAI/bge-small-en` | 384 | MIT |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Apache-2.0 |

## Functions

### Scalars

| function | signature | notes |
|---|---|---|
| `embed(text)` | `VARCHAR → FLOAT[]` | sentence embedding, default model |
| `embed(text, model)` | `(VARCHAR, VARCHAR) → FLOAT[]` | explicit model (arity overload) |
| `embed_query(text)` | `VARCHAR → FLOAT[]` | embed a **search query**; applies the model's query instruction prefix |
| `embed_query(text, model)` | `(VARCHAR, VARCHAR) → FLOAT[]` | explicit model and its query instruction prefix |
| `embed_passage(text)` | `VARCHAR → FLOAT[]` | embed a **document/passage**; no prefix (by design) |
| `embed_passage(text, model)` | `(VARCHAR, VARCHAR) → FLOAT[]` | explicit model; no query prefix |
| `similarity(a, b)` | `(FLOAT[], FLOAT[]) → DOUBLE` | cosine similarity in `[-1, 1]`; **pure, no model** |
| `embedding_dim(model)` | `VARCHAR → INT` | vector length for a model name (`''` = default) |

### Table function

| function | columns |
|---|---|
| `supported_models()` | `(model VARCHAR, dim INT)` |

All three embedding functions expose their optional `model` via an **arity overload** because VGI scalar
functions are positional-only (`name := value` is a table-function feature).

### Query vs. passage (retrieval asymmetry)

bge retrieval models recommend prefixing **queries** — not passages — with an
instruction. For `bge-*`, `embed_query` prepends:

> `Represent this sentence for searching relevant passages: `

`embed_passage` (and plain `embed`) apply **no** prefix. For a symmetric
similarity comparison use `embed` on both sides; for query→document retrieval use
`embed_query` on the query and `embed_passage` on the corpus.

### NULL / robustness semantics

NULL or empty/whitespace-only text → a NULL vector. `similarity` returns NULL for
NULL inputs, empty vectors, length mismatches, or a zero-magnitude vector. Nothing
crashes on odd input.

## Using with DuckDB VSS

Embed once into a column, build an HNSW index, query with a vector:

```sql
INSTALL vss; LOAD vss;
ALTER TABLE docs ADD COLUMN v FLOAT[384];
UPDATE docs SET v = embed.embed_passage(body);
CREATE INDEX docs_hnsw ON docs USING HNSW (v) WITH (metric = 'cosine');

SELECT id, body
FROM docs
ORDER BY array_cosine_distance(v, embed.embed_query('reset my password')::FLOAT[384])
LIMIT 5;
```

## Development

```bash
uv sync --extra dev
make models                  # pre-warm the fastembed cache (downloads the default model)
uv run --no-sync pytest -q   # unit (model-gated tests self-skip on a cold/offline checkout)
make test-sql                # E2E: haybarn-unittest over test/sql/* (authoritative)
make test                    # both
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_embed/
```

`make test-sql` exports `VGI_EMBED_WORKER=".venv/bin/python embed_worker.py"` (the synced project venv)
and runs `haybarn-unittest --keep-home --test-dir . "test/sql/*"` (install once:
`uv tool install --prerelease allow 'haybarn-unittest>=2.0.0rc11,<2.1'`; haybarn 2.0 is required for the worker's VGI protocol vgi.v3 / 3.3, then put `~/.local/bin` on `PATH`).

## License

Worker code: **MIT** (see [LICENSE](LICENSE)). The default model
`BAAI/bge-small-en-v1.5` is MIT-licensed; `fastembed` is Apache-2.0. The `vgi`
DuckDB extension and `vgi-python` are licensed separately by Query Farm.

---

## Authorship & License

Written by [Query.Farm](https://query.farm).

Copyright 2026 Query Farm LLC - https://query.farm


## Executed notebooks

Explore [retrieval depth, cosine checks and precision/recall](notebooks/README.md) with the released worker.
