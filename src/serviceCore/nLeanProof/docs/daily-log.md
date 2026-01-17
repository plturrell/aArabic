# nLeanProof Daily Log

## Day 1

Focus:
- Freeze spec and repo conventions
- Verify build scripts and smoke server

Results:
- Spec and conventions added
- Zig build succeeded (`zig build -Doptimize=ReleaseFast`)
- Smoke server failed in sandbox: `AccessDenied` on listen (unable to bind)

## Day 2

Focus:
- Conformance harness skeleton (test discovery)
- Baseline discovery run

Results:
- Added conformance discovery tool + build step
- Discovery run succeeded for 10 tests under `tests/lean`
- Zig unit tests passed (`zig build test`)

## Day 3

Focus:
- Baseline report generation
- Oracle runner for upstream Lean4
- Diff tooling for triage

Results:
- Baseline report tool added and executed (sample size 5)
- Oracle subset run succeeded (5 tests)
- Diff tool produced 4 diffs, 1 missing expected output

## Day 4

Focus:
- CI wiring for macOS/Linux
- Unified CI script for local and GitHub Actions

Results:
- Added leanShimmy GitHub Actions workflow for Zig build/tests
- Added `scripts/ci.sh` runner (build/test/conformance spot-check)

## Day 5

Focus:
- Conformance manifest + summary tooling
- Lexer scaffolding and golden tests
- CI cache/artifacts for Zig builds

Results:
- Added manifest/summary Zig tools and CI artifact upload for conformance reports
- Added Mojo lexer token definitions, lexer skeleton, CLI, and golden fixtures
- Added CI caching for Zig build artifacts and global cache

## Day 6

Focus:
- Expand lexer rules (comments, literals, keywords)
- Add Lean-flavored golden fixtures

Results:
- Added nested block comment support and basic literal lexing
- Added keyword recognition and additional golden lexer cases

## Day 7

Focus:
- Parser skeleton and CST output
- Parser golden tests

Results:
- Added parser node types, parser scaffold, and CLI
- Added parser golden fixtures and test script

## Day 8

Focus:
- Conformance lexer integration script

Results:
- Added conformance lexer runner for subset capture

## Day 9

Focus:
- Parser expansion (import/namespace/theorem, precedence/app)
- File-based CLI input and parser conformance capture/diff

Results:
- Added parser grammar coverage + golden fixtures
- Added file input shim and updated lexer/parser CLIs
- Added parser conformance capture and diff scripts

## Day 10

Focus:
- Wire Zig file I/O for CLI --file support
- Extend parser blocks and by-bodies
- Add parser conformance baseline script

Results:
- Added Zig IO library and wired Mojo file reader to it
- Added namespace/section blocks and theorem-by parsing with golden fixtures
- Added parser baseline capture script

## Day 11

Focus:
- Add crashpad disable hooks for Mojo scripts

Results:
- Added `--disable-crashpad` and env passthrough to Mojo test and conformance scripts

## Day 12

Focus:
- Add Mojo home override support for scripts

Results:
- Added `--mojo-home` option and HOME/XDG overrides for Mojo runners
