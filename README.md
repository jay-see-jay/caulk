# forme

Thin generic runtime between `rig-core` (LLM) and `graph-flow` (execution).

**Purpose:** typed prompt registry (`State × Event → Prompt`), pluggable context builder trait, generic policy trait (allowed tools per `State`). No teaching logic. Planners stay in product layer, not in library.

`forme` is a publishing word — an archaic printing term meaning a form locked ready to press. `forme` is free (404 on crates.io) candidate.

## Quick start

```bash
cargo test --lib                # 80 unit tests, core + registry + policy + runtime
cargo run --example hello_world # Idle/Done FSM demo with MockLlm, no network
cargo run --example support_bot # 3-state refund FSM, policy deny demo
cargo run --bin qa_runner       # replay determinism + policy deny + snapshot
cargo doc --no-deps             # public API leak check
cargo fmt --check && cargo clippy -- -D warnings
```

### Use as library

```toml
[dependencies]
forme = { path = "../forme" }  # or 0.1.0 from crates.io once published
```

```rust
use forme::{Runner, PromptKey};
use forme::prompt_registry::InMemoryRegistry;
use forme::context_builder::NoopBuilder;
use forme::policy::AllowAllPolicy;
use forme::rig_adapter::MockLlm;
use std::{collections::HashMap, sync::Arc};
```

### Generic design

- Core `Event`, `State` are user-defined traits (`Send+Sync+Clone+Debug+Display+Serialize`)
- `PromptKey::key_for(&state, &event)` canonical -> `"{state}::{event}"`
- `Registry` trait: `FsRegistry` (`root/<state>/<event>.md`) + `InMemoryRegistry` (HashMap) with mtime cache
- `ContextBuilder<E>`: `NoopBuilder`, `FileSnippetBuilder` (50-line window), `ParagraphBuilder` (last N paras)
- `Policy<S>`: `AllowAllPolicy`, `DenyListPolicy`, `AllowListPolicy`, or custom per-state allowlist (see `support_bot`)
- `LlmAdapter<E>`: async trait `call(prompt, context) -> String`, `MockLlm` with canned or closure
- `Runner<S,E,R,B,P,L>`: deterministic `prepare` (pure) + `step` (async LLM), `handle_edge` for graph-flow

## Build waves (BFS)

- Wave 0: core — Event, State, PromptKey, ToolId, FormeError (14 tests)
- Wave 1: prompt_registry (FsRegistry + InMemoryRegistry), policy (AllowAll/DenyList/AllowList), persistence trait (8+8 tests)
- Wave 2: rig_adapter MockLlm, FileSnippetBuilder, ParagraphBuilder, InMemoryCheckpointer (30+ tests)
- Wave 3: runtime Runner::prepare/step/handle_edge, ToolPlan (12 tests)
- Wave 4: examples (`hello_world`, `support_bot`) + `qa_runner` + fixtures/snapshots (80+ total lib tests)

Each wave must pass `cargo test` before next.

## Crates.io ready

- name: `forme` (still free candidate)
- description: "Thin generic runtime between rig-core and graph-flow — typed prompt registry (State x Event), pluggable context builder, policy allowlist"
- license: `MIT OR Apache-2.0`
- keywords: `llm, agent, rig, prompt, state-machine`
- categories: `asynchronous`
- edition: 2021, rust-version tested 1.97.1

## Repo layout

```
src/
  core/          # Event/State/PromptKey/ToolId/FormeError
  prompt_registry/ # Registry trait + FsRegistry + InMemoryRegistry
  policy/        # Policy trait + implementations
  context_builder/ # Noop, FileSnippet, Paragraph
  persistence/   # Checkpointer + InMemoryCheckpointer
  rig_adapter/   # LlmAdapter + MockLlm
  runtime/       # Runner, ToolPlan, StepOutput, NextAction
examples/
  hello_world.rs # Idle/Done, NoopBuilder, AllowAll, MockLlm
  support_bot.rs # 3-state refund, ParagraphBuilder, custom policy
tests/
  fixtures/
    event-log.json
    prompts/        # FsRegistry integration
  snapshots/
    hello-world.txt # expected hello_world output
```

See also: `../forme-BLUEPRINT.md` for spec.
