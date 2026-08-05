# forme

Thin generic runtime between `rig-core` (LLM) and `graph-flow` (execution).

**Destination:** typed prompt registry (State × Event → Prompt), pluggable context builder trait, generic policy trait (allowed tools per State). No teaching logic. No wayfinder types.

See:
- BLUEPRINT: ../forme-BLUEPRINT.md
- Process: ../forme-build-process.md

## Crates.io
`forme` is free (404 on crates.io) — candidate name: archaic printing term, type locked ready to press.

## Build waves (BFS)
Wave 0: core — Event, State, PromptKey, FormeError
Wave 1: prompt-registry, policy, context-builder trait, persistence trait
Wave 2: rig-adapter, context-builder impls, persistence in-mem
Wave 3: runtime
Wave 4: examples + qa

Each wave must pass `cargo test` before next.

## Quick start
cargo test
