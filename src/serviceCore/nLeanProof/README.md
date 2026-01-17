# nLeanProof: Lean4 Compiler/Runtime Rewrite (Mojo/Zig/Lean)

nLeanProof is a full-stack replacement of Lean4 (v4.26.0) built with
Mojo, Zig, and Lean. It targets full language feature parity and provides a
Zig HTTP + OData server, Mojo compiler core, and optional UI tooling.

Current state: project scaffold + implementation plan. Compiler/runtime
features are under active build-out.

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

## Documentation

- Implementation plan: docs/implementation-plan.md
- Specification: docs/spec.md
- Conventions: docs/conventions.md
- Daily log: docs/daily-log.md

## CI

- GitHub workflow `leanshimmy-ci` builds with Zig 0.15.2, caches inference fixtures and build outputs, runs OpenAI smoke, and enforces a QPS sanity check (`bench_qps.sh -n 50 -c 4 --min-qps 30`, pool size 2).
