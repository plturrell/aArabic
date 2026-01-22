# Parser Golden Tests

Golden tests run the Mojo parser CLI against fixtures in `tests/parser/golden`.

## Run

```bash
./scripts/parser_golden.sh
```

## Notes

- Output is a compact s-expression via `node_to_string`.
- Use `--disable-crashpad` if Mojo crashpad cannot start in your environment.
- Use `--mojo-home PATH` to redirect Mojo runtime state to a writable directory.
