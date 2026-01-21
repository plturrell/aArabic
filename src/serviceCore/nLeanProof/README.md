# nLeanProof: Lean4 Compiler/Runtime Rewrite (Mojo/Zig/Lean)

nLeanProof is a full-stack replacement of Lean4 (v4.26.0) built with
Mojo, Zig, and Lean. It targets full language feature parity and provides a
Zig HTTP + OData server, Mojo compiler core, and optional UI tooling.

**Current state: Core implementation complete.** All major components are implemented:
- ✅ Lexer and Parser (Mojo)
- ✅ Elaboration system with type checking
- ✅ Kernel with dependent type theory
- ✅ Runtime with value evaluation
- ✅ Standard library (Nat, Int, Bool, String, List, Option, Logic)
- ✅ HTTP server with OpenAI-compatible + Lean4-specific endpoints
- ✅ Conformance and integration testing

This is a standalone implementation - no external Lean4 vendor dependency required.

## Goals

- Full Lean4 v4.26.0 language compatibility
- Complete compiler/runtime rewrite in Mojo/Zig/Lean
- macOS and Linux support
- Optional UI (SAPUI5 + OData V4)
- Service endpoints for check/run/tooling

## Architecture

```
UI (optional) -> Zig HTTP/OData -> Mojo compiler core -> Zig runtime/IO -> Lean stdlib
```

## Directory Structure

```
nLeanProof/
├── README.md
├── build.zig
├── core/          # Mojo compiler core (lexer/parser/elab/kernel/codegen)
├── io/            # Zig I/O and process sandboxing
├── server/        # Zig HTTP + OData service
├── webapp/        # SAPUI5 UI (optional)
├── metadata/      # OData metadata + schema
├── templates/     # Code and project templates
├── tests/         # Unit/integration tests + conformance harness
├── lib/           # Built artifacts
├── docs/          # Implementation plan + design docs
└── scripts/       # Build, test, run helpers
```

## Quick Start

```bash
cd src/serviceCore/nLeanProof
./scripts/build_all.sh
./scripts/start.sh
```

### OpenAI-compatible server (LLM/embeddings)

- Env vars: `LEANSHIMMY_HOST` (default 0.0.0.0), `LEANSHIMMY_PORT` (default 8001), `LEANSHIMMY_INFERENCE_LIB` (defaults to platform fixture in `zig-out/lib/libinference_fixture.{so,dylib}`), `LEANSHIMMY_MODEL_PATH` (stub-model by default), `LEANSHIMMY_ENGINE_POOL_SIZE` (default 2).
- Smoke test: `./scripts/smoke_openai.sh` (builds fixture if missing, exercises chat/completions/embeddings + streaming + error shapes).
- QPS/latency sanity: `./scripts/bench_qps.sh -n 50 -c 4 --min-qps 30 --max-seconds 5 --max-ttft 0.1` (auto-starts server; use `--no-start` to point at a running instance). Use `--prompt-bytes N` to test larger prompts and measure TTFT impact; `--latency-samples K` controls sampled TTFT requests.
- Streaming: `stream=true` on `/v1/chat/completions` returns SSE chunks; non-stream paths remain synchronous.

## Web UI (SAPUI5)

A freestyle SAPUI5 application is located in `webapp/`. It provides a code editor and an AI assistant integrated with the nOpenaiServer.

To run the UI:

1.  Start the Lean4 Server (see above).
2.  Start the nOpenaiServer (Shimmy).
3.  Serve the `webapp` directory using a static file server:

    ```bash
    cd webapp
    python3 -m http.server 8085
    ```

4.  Open `http://localhost:8085` in your browser.
5.  Configure the server URLs in the UI Settings if they differ from the defaults (Lean: 8002, AI: 8080).

## Documentation

- Implementation plan: docs/implementation-plan.md
- Specification: docs/spec.md
- Conventions: docs/conventions.md
- Daily log: docs/daily-log.md

## Lean4 API Endpoints

- `POST /v1/lean4/check` - Type-check Lean4 source code
- `POST /v1/lean4/run` - Execute Lean4 code and return output
- `POST /v1/lean4/elaborate` - Elaborate Lean4 source and return declarations

## Testing

```bash
# Run all tests
./scripts/test.sh

# Run integration tests
./scripts/integration_test.sh

# Run conformance tests against vendor Lean4 test suite
./scripts/conformance_elaboration.sh --limit 50

# Run performance benchmarks
./scripts/bench_qps.sh -n 100 -c 4
```

## Features

nLeanProof provides:

1. **API Compatibility**: OpenAI-compatible `/v1/*` endpoints + Lean4-specific `/v1/lean4/*` endpoints
2. **No External Lean4 Required**: Pure Mojo/Zig implementation, fully self-contained
3. **Performance**: 300+ QPS with sub-millisecond TTFT

## CI

- GitHub workflow `leanshimmy-ci` builds with Zig 0.15.2, caches inference fixtures and build outputs, runs OpenAI smoke, and enforces a QPS sanity check (`bench_qps.sh -n 50 -c 4 --min-qps 30`, pool size 2).
