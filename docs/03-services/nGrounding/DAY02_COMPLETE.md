# Day 2 (Complete)

## Focus
- Expose OpenAI-compatible HTTP endpoints.
- Stub inference bridge and CI smoke coverage.

## Done
- Added chat/completions/embeddings routes (default port 8001) with JSON bodies.
- Stub inference engine hooked; dynamic lib hooks in place; model preload envs added.
- Smoke test script added; CI script updated to run it.

## Next
- Integrate real `libinference`, tighten validation/escaping, and strengthen smoke assertions (Day 3).
