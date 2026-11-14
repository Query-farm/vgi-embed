# Executed embedding walkthrough

[How many passages should a search return?](01_audit_retrieval_depth.ipynb) embeds two synthetic support questions and eight substantive passages, preserves two missing/blank input cases, verifies worker cosine scores with independent SQL arithmetic, and explores precision/recall at different retrieval counts. All relevance judgments are explicitly authored, not benchmark labels.

## Run from this directory

Install [uv](https://docs.astral.sh/uv/) and use Python 3.13. First use needs network access for the pinned worker, dependencies, VGI extension and BGE model weights. No API key or worker checkout is required. Copy the entire `notebooks/` directory to run elsewhere.

```sh
cd notebooks
uv sync --frozen
uv run --frozen python execute.py --timeout 300
```

The separate environment pins Haybarn `2.0.0rc11`; it does not alter root dependencies. The runner creates a temporary kernel with its own interpreter and writes outputs atomically only after a successful run. A failed execution preserves the last successful notebook.

For an interactive editor, register this environment once and select its kernel:

```sh
uv run --frozen python -m ipykernel install --user --name vgi-embed-notebooks --display-name 'Python (vgi-embed notebooks)'
```

The notebook expects its working directory to be `notebooks/`. Helpers, images and documentation are local companions; no root `.venv` or source discovery is used.

## One-parameter experiment

`TOP_K` is an integer 1–8, in passages per question. The default is 2. Increase it to 4, leaving everything else unchanged:

```sh
uv run --frozen python execute.py --timeout 300 --parameters '{"TOP_K":4}' --output-dir artifacts/top-four
```

Overrides require a separate output directory. The runner copies the assets, helper, environment lock, runner and this README into successful experiment directories. Logs, extracted charts, kernels and experiment copies are ignored by Git. Reset `TOP_K = 2` for the default.

## What executes

The visible SQL uses this tested literal remote launch:

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'embed' (TYPE vgi, LOCATION 'uvx --python 3.13 --with vgi-python==0.51.0 --with vgi-rpc==0.53.1 --with pyarrow==26.0.1 --with fastembed==0.9.1 --with onnxruntime==1.31.1 --from git+https://github.com/Query-farm/vgi-embed@v0.5.0 vgi-embed-worker');
```

Tag `v0.5.0` resolves to source revision `a0ad9a77a0aae6b55fb0f2d5ed3292473c6d6743`. The explicit model is `BAAI/bge-small-en-v1.5`, with 384-dimensional FLOAT vectors. The helper sets the same startup default, then restores the previous setting on cleanup. Optional `VGI_EMBED_CACHE_DIR` / `FASTEMBED_CACHE_PATH` select the model cache. A cold offline cache fails rather than substituting vectors. Model calls remain local after weights are available; first-use download reaches the upstream model host. No text is sent to a paid embedding API.

The runtime package versions are pinned; fastembed's named model resolver does not pin an immutable weight revision here. Exact floats can vary with model artifacts, ONNX build and platform. Validate dimensions, cosine identities and retrieval behavior when upgrading the worker, runtime or model. Re-embed both sides if switching model; equal dimensions alone do not imply compatibility.

All vectors and worker similarities are materialized before DataFrame capture. SQL performs the joins, ranking, independent cosine check and precision/recall arithmetic; Python handles validation, plotting and result-derived prose. No approximate search index or Python embedding library is used in the notebook environment.

## Interpretation

The collection is synthetic and completely judged by the notebook author. Missing and blank text remain unavailable, not zero scores. Unlisted substantive pairs are deliberately labeled irrelevant in this closed example. Both queries have two relevant passages. Precision divides by returned count; recall divides by the complete relevant count. Scores are dimensionless cosine similarities, not probabilities, and no numeric threshold is calibrated. These two questions cannot establish general retrieval quality.

The separate experiment changes only the selected prefix. Recall cannot fall when k increases; precision can rise or fall. Counts, IDs and highlighted curve positions come from SQL results. Worker/SQL cosine agreement uses a `1e-6` tolerance rather than exact float assertions.

## Validation

Verified on macOS arm64 with the published engine and remote release:

- Default and `TOP_K=4` experiment both ran from a copied `notebooks/` directory without repository source (3.8 s and 3.6 s). Each has 15 code cells, seven SQL cells, two charts and nine SQL invariants. All four charts were inspected.
- At `k=2`, both questions have 100% authored-label precision and recall. At `k=4`, recall stays 100% while precision falls to 50%; the same sixteen scores and complete curves are used. These are demonstration results, not measured general search quality.
- Worker and independent SQL cosine differ by at most `1.1e-15` in this run. Missing/blank handling, dimensions, pair coverage, symmetry, self-similarity and metric bounds pass.
- All three runner guards pass: missing output destination, same-directory destination, and invalid `TOP_K=9` fail without changing the successful default. Notebook structural review and Ruff format/lint pass.
- Current source checks, separate from the release: 58 pytest tests; 51 SQL assertions in two cases; Ruff lint/format and strict mypy (seven source files); metadata score 100, L2, zero findings. Metadata execution was enabled; external-link checks and paid AI passes were not run.
- The root environment's additional pydoclint invocation fails to import `DocstringYields` due to an existing dependency combination. An isolated `uvx pydoclint vgi_embed` run passes; root dependencies and lock are unchanged.

Validation logs, extracted charts and experiment copies are under ignored `artifacts/`.
