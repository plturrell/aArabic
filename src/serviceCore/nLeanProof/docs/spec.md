# nLeanProof Specification (v0.1)

Target: full Lean4 v4.26.0 compiler/runtime parity.

## Compatibility

- Language: Lean4 v4.26.0, full feature parity
- OS: macOS + Linux only
- UI: optional (SAPUI5 + OData V4)

## Architecture

- Zig server (HTTP/OData) for service and tooling endpoints
- Mojo compiler core (lexer, parser, elaborator, kernel, codegen)
- Zig runtime/IO layer (filesystem, process, sandboxing)
- Lean stdlib rebuilds on top of the new runtime

## Non-goals (initial)

- Windows support
- Backward compatibility with Lean3
- Embedded Python/JS tooling

## Conformance

- Upstream test suite in `vendor/layerIntelligence/lean4/tests` is the source
  of truth for behavior and regressions.
- Differential testing against upstream Lean4 is allowed as a test oracle, but
  the new compiler/runtime must not link or depend on upstream Lean4 at runtime.

## Service Endpoints

Base service endpoints (subject to expansion):

- `GET /health`
- `GET /version`
- `POST /v1/lean4/check`
- `POST /v1/lean4/run`

