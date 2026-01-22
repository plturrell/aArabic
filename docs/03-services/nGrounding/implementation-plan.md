# nLeanProof Implementation Plan (Project Scope)

## Objectives
- Rebuild Lean4 compiler/runtime stack in Mojo/Zig with OpenAI-compatible APIs.
- Replace legacy Python/Lean artifacts with pure Mojo/Zig/Lean components.
- Deliver production-grade HTTP services (LLM, embeddings) with low TTFT/high QPS.

## Scope
- Front-end: Lexer/parser (Lean4 grammar coverage), elaboration/typechecker scaffolding, codegen/bytecode plan.
- Services: OpenAI-compatible HTTP server, embedding endpoints, model management, CI smoke + conformance harness.
- Bridge: Zig inference engine shared library (load/generate/embed) with Mojo bindings.
- SDK/Clients: Lean/Mojo client stubs, clear API contracts.
- Perf: Async runtime tuning, load/TTFT profiling and fixes.

## Deliverables
- Production Zig server with inference bridge and OpenAI-compatible endpoints.
- Mojo front-end (lexer/parser) plus elaboration/typechecker scaffolding, moving toward full Lean4 parity.
- Conformance harness wired to Lean4 corpus; smoke, perf, and load test suites.
- SDK/client stubs and API contracts; architecture/ABI docs and daily logs.
- Measurable perf targets: initial targets TTFT <= 300 ms P50, QPS >= 50 on baseline hw (define in perf section); adjust with profiling.

## Timeline (120 days, phased)
- Days 1–15: Foundation — toolchain green (Mojo/Zig), lexer/parser golden tests, IO shim, basic OpenAI endpoints and stub inference (completed).
- Days 16–30: Inference integration — real `libinference` ABI stabilized, request validation/escaping, CI smoke with real outputs, streaming plan.
- Days 31–60: Lean4 front-end growth — elaboration/typechecker scaffolding, core language features, conformance harness on subset corpus, expand tests and SDK stubs.
- Days 61–90: Compiler/runtime path — codegen/bytecode plan, runtime shims, deeper Lean4 feature coverage; API hardening and error models.
- Days 91–110: Performance — QPS/TTFT tuning, async/runtime tuning, load/perf tests, regression suite; embedding/LLM perf baselines.
- Days 111–120: Release hardening — broaden conformance coverage, docs (release notes, architecture/ABI), CI polish, final SDK/client examples.

Daily logs should track progress against these phases; adjust scope/risk weekly based on integration findings.

## Quantitative Targets (initial; refine as data arrives)
- Perf: TTFT <= 300 ms P50, <= 600 ms P90; QPS >= 50 on baseline hw (document CPU/GPU, batch sizes). Embedding latency <= 150 ms P50.
- Conformance: By Day 60, pass parser/lexer conformance on subset Lean4 corpus; by Day 120, target >= 90% of agreed Lean4 feature list.
- Stability: CI green with smoke + conformance subset; zero regressions on streaming/chat/completions/embeddings endpoints.

## Tooling (current)
- `scripts/smoke_openai.sh` — OpenAI-style chat/completion/embed smoke + streaming and error-shape checks.
- `scripts/bench_qps.sh` — quick QPS/latency sanity (configurable requests/concurrency, auto-starts fixture-backed server).
- Concurrency baseline: round-robin engine pool (size from `LEANSHIMMY_ENGINE_POOL_SIZE`, default 2) with per-connection threads; future work: per-engine async/io tuning.
- CI: `leanshimmy-ci` workflow builds, runs smoke, and enforces a conservative QPS sanity (`bench_qps.sh -n 50 -c 4 --min-qps 20`) with cached inference fixture binaries.

## Phase Exit Criteria & Risks
- Day 30 exit: `libinference` ABI frozen; smoke test runs against real lib; streaming plan drafted. Risk: ABI churn/blockers on model availability. Mitigation: freeze a minimal ABI and ship a dummy lib for CI.
- Day 60 exit: Elaborator/typechecker scaffolding complete; subset conformance suite passes; SDK stubs published. Risk: Lean4 semantics scope creep. Mitigation: define a minimal feature list (imports/namespaces/defs/theorems/core typing) and lock it.
- Day 90 exit: Compiler/runtime plan and initial codegen hooks; expanded conformance; error model documented. Risk: runtime complexity. Mitigation: prototype bytecode/IR early.
- Day 110 exit: Perf targets met on baseline hw with documented runs; load test scripts in CI optional job. Risk: hardware variance. Mitigation: fix a reference machine/profile and capture configs.
- Day 120 exit: Release notes, architecture/ABI docs, SDK examples; CI with smoke + subset conformance; known issues list.

## Ownership & Dependencies
- Owners: (fill with responsible leads for inference, front-end, perf, CI/docs).
- Dependencies: `libinference` availability; Lean4 corpus access; baseline perf hardware definition; CI runners with required toolchains.

## Day-by-Day (next 30 days, detailed)
- Day 16: Freeze `libinference` C ABI (signatures for load/generate/embed/info); write an ABI doc.
- Day 17: Implement metadata/info call in `libinference` fixture (max_output, embedding_dims); adjust `InferenceEngine` to consume.
- Day 18: Publish a minimal `libinference` fixture (CI-friendly); add build script and include in repo.
- Day 19: Harden HTTP parsing: fail invalid JSON with 400; ensure output escaping; add unit tests for validation paths.
- Day 20: Add structured error codes/messages (e.g., `invalid_json`, `prompt_too_large`, `model_not_loaded`); update smoke to assert error shapes.
- Day 21: Design streaming for chat (SSE or chunked); choose protocol; draft interface and test plan.
- Day 22: Implement streaming prototype for chat; add streamed smoke test hitting /v1/chat/completions with `stream=true`.
- Day 23: Refine streaming (backpressure, partials); add regression tests for both streamed/non-streamed paths.
- Day 24: Concurrency plan: document thread/async strategy; add mutex coverage for shared state; code review.
- Day 25: Implement per-connection engine guarding or pooled engines; measure basic QPS sanity under concurrency.
- Day 26: Add QPS micro-benchmark script; capture baseline numbers; adjust buffer sizes or thread settings.
- Day 27: Wire smoke/CI to run against the real `libinference` fixture; update CI cache/artifacts.
- Day 28: Add minimal perf smoke (latency budget check) and fail CI if P50/P90 exceed thresholds (configurable).
- Day 29: Define baseline hardware profile (CPU/GPU, threads); document perf test setup.
- Day 30: Review ABI/doc + streaming + perf smoke; lock Day 30 exit criteria.

## Day-by-Day (Lean4 front-end, Days 31–60)
- Day 31: Lock minimal feature list for Lean4 subset (imports, namespaces, defs, theorems with simple types).
- Day 32: Elaborator scaffolding: contexts, symbols, environments; stub typechecker entrypoints.
- Day 33: Add basic type representations and unification stubs; unit tests for context ops.
- Day 34: Wire parser/AST to elaborator entrypoints; smoke elaboration on tiny inputs.
- Day 35: Extend parser/AST for needed Lean4 syntax (binders, simple tactics if required); add tests.
- Day 36: Add error reporting structure (locations, codes); unit tests.
- Day 37: Implement type inference for defs with simple types; tests on sample Lean code.
- Day 38: Theorem handling: simple proof-by-body or `by` placeholder; tests.
- Day 39: Integrate elaboration results into conformance harness; run subset corpus locally.
- Day 40: Add Lean/Mojo SDK stubs for submitting Lean code; document API contract.
- Day 41: SDK examples + tests; CI job to run elaboration subset.
- Day 42: Expand conformance subset coverage; track pass/fail metrics.
- Day 43: Fill parser/elaborator gaps based on failures; iterate.
- Day 44: Refine error messages; ensure CI artifacts capture failures.
- Day 45: Assess progress vs 60-day target; adjust backlog.
- Day 46–50: Broaden syntax/typing coverage based on conformance failures; add regression tests.
- Day 51–55: Finalize SDK/client stubs; ensure examples pass in CI.
- Day 56–60: Target >= subset pass rate; document remaining feature gaps and plan for Day 61–90.

## Day-by-Day (Compiler/Runtime/Perf, Days 61–110)
- Day 61: Publish codegen/bytecode plan and IR sketch; review.
- Day 62–64: Prototype IR/bytecode emitter skeleton; unit tests on tiny programs.
- Day 65: Runtime shim design (memory, GC/arena strategy); doc + review.
- Day 66–68: Implement minimal runtime hooks; wire IR emitter to produce stubs.
- Day 69–70: Integrate with conformance harness where possible; track coverage.
- Day 71–75: Expand Lean4 feature coverage; close gaps identified in Day 56–60; error model refinement.
- Day 76–78: Improve diagnostics/logging; add regression tests for new features.
- Day 79–80: Run conformance and capture metrics; plan next fixes.
- Day 81–83: Initial perf profiling (HTTP + inference); identify bottlenecks (buffer, locking, thread).
- Day 84–86: Optimize based on profiling; tune buffer sizes, thread model, async runtime if used.
- Day 87–90: Perf tuning for TTFT/QPS targets; add load/perf test scripts; benchmark on baseline hw; set thresholds.
- Day 91–94: Regression suite for perf/stability; include streaming robustness tests.
- Day 95–97: Harden error codes/logging; stress test under load; fix crashes/leaks.
- Day 98–100: Verify perf targets; adjust thresholds or code as needed; update docs.
- Day 101–105: Broaden perf/load scenarios; capture reports; update CI optional perf job.
- Day 106–110: Prepare for hardening: finalize perf regressions, clean up tech debt flagged by profiling.

## Day-by-Day (Hardening, Days 111–120)
- Day 111: Audit conformance gaps; prioritize fixes; assign owners.
- Day 112–114: Fix top conformance blockers; rerun suites; document deltas.
- Day 115: Update architecture/ABI docs and release notes draft.
- Day 116: Finalize SDK/client examples; ensure they run in CI.
- Day 117: CI polish (artifacts, logs, perf optional job stability).
- Day 118: Compile known-issues list and mitigations.
- Day 119: Release checklist execution (tests, docs, versioning).
- Day 120: Final sign-off; publish release notes and artifacts.
