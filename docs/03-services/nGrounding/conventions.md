# nLeanProof Conventions

## Language + Build

- ASCII by default
- Zig 0.15.2+ and Mojo 0.26+
- No Python in production paths
- Prefer Zig for IO/runtime and Mojo for compiler core

## Layout

- `core/`: compiler core (lexer/parser/elab/kernel/codegen)
- `io/`: Zig IO + sandboxing utilities
- `server/`: Zig HTTP/OData server
- `webapp/`: optional SAPUI5 UI
- `tests/`: unit + integration + conformance harness

## API + Error Handling

- JSON everywhere
- Stable error schemas
- Explicit versioning in responses

## Testing (daily)

- Run `zig build test` for Zig units
- Run `mojo test tests/mojo` when present
- Run a small upstream conformance subset daily

## Coding Style

- Small, focused modules
- No implicit globals
- Clear ownership of allocations
- Use tests for regressions

