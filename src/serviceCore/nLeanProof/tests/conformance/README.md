# Conformance Harness

Discovery tool for Lean4 upstream tests.

## Usage

```bash
zig build conformance-discover -- --root tests/lean4 --suite lean --limit 10
```

## Baseline Report

```bash
zig build conformance-baseline -- --root tests/lean4 --suite lean --sample 10
```

## Manifest

```bash
zig build conformance-manifest -- --root tests/lean4 --suite lean --output tmp/conformance/manifest.json
```

## Summary

```bash
zig build conformance-summary -- --root tests/lean4 --suite lean --output tmp/conformance/summary.json
```

## Oracle + Diff

```bash
./scripts/conformance_oracle.sh --limit 5
./scripts/conformance_diff.sh
```

## Lexer Output Capture

```bash
./scripts/conformance_lexer.sh --limit 10
```

## Parser Output Capture + Diff

```bash
./scripts/conformance_parser.sh --limit 10
./scripts/conformance_parser_baseline.sh --limit 10
./scripts/conformance_parser_diff.sh
```

Use `--disable-crashpad` on these scripts if Mojo crashpad cannot start.
Use `--mojo-home PATH` to redirect Mojo runtime state to a writable directory.

## JSON Output

```bash
zig build conformance-discover -- --json --limit 5
```
