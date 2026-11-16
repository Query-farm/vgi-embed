# CI: the vgi-embed SQL suite

[`run-integration.sh`](run-integration.sh), adapted from the vgi-starter template,
runs `test/sql/*.test` through the published `haybarn-unittest` sqllogictest runner
with the signed community `vgi` extension (no C++ build), over one transport:

| `TRANSPORT` | `VGI_EMBED_WORKER` (the ATTACH LOCATION) | How DuckDB reaches the worker |
|-------------|-------------------------------|-------------------------------|
| `stdio`     | `$WORKER`                     | spawns it per connection, Arrow IPC over stdin/stdout |
| `launch`    | `launch:$WORKER`              | starts it once on an AF_UNIX socket and pools connections |
| `http`      | `http://127.0.0.1:<port>`     | the script starts `$WORKER --http --port 0` and reads `PORT:<n>` |

`WORKER` is an executable path: the project venv's `vgi-embed-worker` console script. The
script rewrites `require <ext>` into a `LOAD` (the runner links no extensions),
adds `LOAD httpfs` for the http leg, installs the extensions once, and fails on
any unexpected skip (`EXPECTED_SKIP_REASONS`) or if fewer than `MIN_EXECUTED`
test files ran: the runner otherwise turns errors mentioning "HTTP" into silent
skips and exits 0 when nothing matched.

CI (`.github/workflows/ci.yml`, job `integration`) runs it as a
`transport: [stdio, launch, http]` matrix. Locally:

```bash
make test-sql                      # all three transports
make test-sql TRANSPORTS=launch    # one
```
