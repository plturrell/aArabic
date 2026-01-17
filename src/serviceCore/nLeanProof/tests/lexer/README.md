# Lexer Golden Tests

Golden tests run the Mojo lexer CLI against fixtures in `tests/lexer/golden`.

## Run

```bash
./scripts/lexer_golden.sh
```

## Notes

- Fixtures are single-line for now (CLI uses `--text` input).
- Multiline inputs are supported via `--text`; keep fixtures small.
- ASCII-only lexer behavior until Unicode handling lands (non-ASCII treated as identifiers).
- Use `--disable-crashpad` if Mojo crashpad cannot start in your environment.
- Use `--mojo-home PATH` to redirect Mojo runtime state to a writable directory.
