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

- Test suite in `tests/lean4` contains conformance tests for Lean4 behavior.
- This is a standalone implementation with no external Lean4 dependency.

## Service Endpoints

Base service endpoints (subject to expansion):

- `GET /health`
- `GET /version`
- `POST /v1/lean4/check`
- `POST /v1/lean4/run`

