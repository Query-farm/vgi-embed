# vgi-embed

Local **text embeddings** as DuckDB SQL functions — for semantic search and
retrieval-augmented generation (RAG) with [DuckDB VSS](https://duckdb.org/docs/extensions/vss).

A [VGI](https://query.farm) worker that turns text into fixed-length `FLOAT[]`
vectors entirely **on your machine** — no API keys, no network at query time. It
runs sentence-transformer models through [`fastembed`](https://github.com/qdrant/fastembed)
(Qdrant, Apache-2.0), which uses **ONNX Runtime — no torch**, so it installs light
and starts fast.

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'embed' (TYPE vgi, LOCATION 'uv run embed_worker.py');

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

Other supported models (pass as the second argument to `embed`, or query
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
| `embed_passage(text)` | `VARCHAR → FLOAT[]` | embed a **document/passage**; no prefix (by design) |
| `similarity(a, b)` | `(FLOAT[], FLOAT[]) → DOUBLE` | cosine similarity in `[-1, 1]`; **pure, no model** |
| `embedding_dim(model)` | `VARCHAR → INT` | vector length for a model name (`''` = default) |
| `embed_version()` | `→ VARCHAR` | worker + backend + default-model identity |

### Table function

| function | columns |
|---|---|
| `supported_models()` | `(model VARCHAR, dim INT)` |

`embed` exposes its optional `model` via an **arity overload** because VGI scalar
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

`make test-sql` exports `VGI_EMBED_WORKER="uv run --python 3.13 embed_worker.py"`
and runs `haybarn-unittest --test-dir . "test/sql/*"` (install once:
`uv tool install haybarn-unittest`, then put `~/.local/bin` on `PATH`).

## License

Worker code: **MIT** (see [LICENSE](LICENSE)). The default model
`BAAI/bge-small-en-v1.5` is MIT-licensed; `fastembed` is Apache-2.0. The `vgi`
DuckDB extension and `vgi-python` are licensed separately by Query Farm.
