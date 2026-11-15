# Local text embedding examples

Two independent notebooks distinguish candidate ranking from review decisions:

- [How many passages should a search return?](01_audit_retrieval_depth.ipynb) preserves the retrieval-depth walkthrough: two authored questions, eight substantive passages, missing-text controls, exhaustive scores and complete precision/recall curves. Model registry and scalar metadata validate the chosen vector width.
- [Which title pairs deserve a duplicate review?](02_similarity_is_a_review_signal.ipynb) uses symmetric embeddings for seven authored titles, accounts for all 21 unordered pairs, and separates exact text, authored paraphrases and related tasks before applying a review cutoff.

## Run locally or in Colab

Use each notebook's Colab badge and run all cells; setup installs only the base shared package and retrieves three hash-verified companions. Locally, copy this directory, install [uv](https://docs.astral.sh/uv/) and Git, then:

```bash
uv sync --locked
uv run --locked vgi-notebooks 01_audit_retrieval_depth.ipynb
uv run --locked vgi-notebooks 02_similarity_is_a_review_signal.ipynb
```

The notebook lock is separate from the worker's root dependencies. Its shared `[runner]` extra supports local publication; Colab installs the base package without replacing its kernel/Jupyter client. For an interactive local kernel:

```bash
uv run --locked python -m ipykernel install --user --name vgi-embed-notebooks --display-name 'Python (VGI Embed examples)'
```

Both books visibly attach the released worker through the same short command:

```sql
ATTACH 'embed' (TYPE vgi, LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-embed@v0.5.0 vgi-embed-worker');
```

No local worker checkout, manually compiled binary or transitive dependency overrides are needed. First use requires network for Git, packages, the published Haybarn/VGI extension and about 67.4 MB of public model assets. Subsequent inference runs locally. Fresh temporary SQL-secret storage with persistent secrets disabled prevents saved credentials from affecting attachment; the notebooks verify an empty secret inventory before and after work.

[model-manifest.json](model-manifest.json) pins fastembed's `BAAI/bge-small-en-v1.5` assets to **Qdrant/bge-small-en-v1.5-onnx-Q revision `aa8f8b060edb00e03bfdd08813a2949946c8ba55`**. [model_assets.py](model_assets.py) downloads seven exact public files, verifies SHA-256 and size, atomically publishes them under ignored `validation/model-cache`, sets the worker's supported cache/default model settings and enables offline model loading. Cleanup restores prior settings. It does not generate vectors or change worker code. The ONNX file is 66,465,124 bytes. An unavailable/corrupt download fails, with no synthetic vector or alternate-model fallback.

## One-parameter experiments

```bash
uv run --locked vgi-notebooks 01_audit_retrieval_depth.ipynb \
  --parameters '{"TOP_K":4}' --output-dir artifacts/top-four
uv run --locked vgi-notebooks 02_similarity_is_a_review_signal.ipynb \
  --parameters '{"REVIEW_THRESHOLD":0.70}' --output-dir artifacts/lower-cutoff
```

The first changes returned passages per question from 2 to 4; the second lowers the dimensionless cosine threshold from 0.85 to 0.70. Full scores remain materialized before these selections. The runner rejects missing/same-directory destinations and unknown parameters, preserves successful defaults, and writes experiment copies separately. Restore each default before publication.

## Data, semantics and coverage

All text and relevance/pair-role labels are invented and visible in SQL. There are no production logs, observed events or benchmark labels. In the retrieval example, unlisted substantive pairs are explicitly irrelevant within this completely judged toy set. NULL/blank passages are unavailable, not zero-score results. In the title example, roles interpret the output without changing scores; a threshold flags a pair for reading and never authorizes deletion, merging or transitive equivalence.

The actual **v0.5.0 catalog has six logical names**: `embed`, `embed_query`, `embed_passage`, `similarity`, `embedding_dim`, and `supported_models`. Three embedding functions have arity overloads, and the registry's function/view alias counts once. There are no physical tables, COPY handlers, authenticated or write operations. The two stories meaningfully cover **6/6 released surfaces and 6/6 reviewed current-interface surfaces**; exact-current runtime count remains **0**. [coverage.json](coverage.json) binds complete runtime exports, signatures, visible SQL, interpreted results and successful execution receipts.

Current public source `b77b166a8d437c144e39b55b5ad4f483f915f836` adds composed example/agent metadata but preserves compute methods, registration, model registry and root dependencies. [Source compatibility](coverage/source-compatibility.json) and the retained source diff document that review. Catalog title/subtitle come from actual release metadata. Four advertised models do not imply four tested runtimes: only the pinned BGE-small snapshot runs. Same vector width does not make different models comparable. Cosine scores are dimensionless and uncalibrated; exact floats can vary by runtime/platform, so invariants use tolerances. The short Git launcher resolves the release's declared dependencies; tested versions are recorded, and changing pins requires rerunning validation.

## Validation

Both saved defaults and separate experiments execute real model inference. The retrieval book has 11 invariants, including independent SQL cosine reconstruction; the title book has 10 invariants plus population reconciliation and undefined-vector controls. A fresh notebook-only Python 3.12 kernel preloads eight plotting/Jupyter libraries before setup and verifies unchanged versions, new companion/model downloads and `pip check`. This is a Colab-compatible setup test, not manual operation of hosted Colab. All changed figures are visually reviewed; SVG geometry/text is preserved and committed content-addressed charts support GitHub. Logs, model caches, PNG previews and experiment copies remain ignored.

Worker source is unchanged. Existing [green source CI](https://github.com/Query-farm/vgi-embed/actions/runs/43567732728) is reused under the shared notebook-only workflow; no worker rebuild or repeated source/transport suite is performed. [STANDARD.md](STANDARD.md) links the maintained shared specification.
