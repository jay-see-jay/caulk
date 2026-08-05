# forme core

**Scope:** `src/core/mod.rs` — zero-dep kernel (except `thiserror`, `serde`). Defines `Event`, `State`, `PromptKey`, `ToolId`, `FormeError`. No I/O, no Rig, no graph-flow. Every other module depends on this.

## Why

Rig provides `preamble` per agent; `graph_flow` provides `Context.get/set` KV-bag. Neither provides per-State prompt registry / tool allow-list. `forme` fills gap generically: user-defined `State × Event → Prompt`, generic `ToolId`, pluggable registry/builder/policy traits.

## Traits

### `Event`
```rust
pub trait Event: Clone + Debug + Display + Send + Sync + 'static + Serialize + DeserializeOwned
```
Blanket impl for any `T` meeting bounds. **NOT object-safe** (`Clone` returns `Self`, serde). Use as generic `E: Event`, not `dyn Event`. Keep `Display` stable — used for canonical `PromptKey`.

### `State`
```rust
pub trait State: Clone + Debug + Display + Eq + Hash + Send + Sync + 'static + Serialize + DeserializeOwned
```
Same object-safety & blanket impl notes. `Display` should be canonical (e.g. `"Idle"`). Adds `Eq+Hash` vs `Event` because it is used as map-key prefix.

## Concrete Types — Not Generic PromptKey

Decision: `PromptKey { state: String, event: String }` concrete, not `PromptKey<S,E>` generic. Avoids combinatorial explosion in registry signatures (`HashMap<PromptKey<S,E>,String>` would require generics everywhere). Construction via `Display`:

- `PromptKey::new(state: impl Into<String>, event: impl Into<String>)` — raw
- `PromptKey::key_for<S:State,E:Event>(&S,&E)` — typed via `to_string()`
- `canonical() -> "{state}::{event}"`, `Display` = canonical, `Debug` = struct-like differing from Display

`From<(String,String)>` and `From<(&str,&str)>` for ergonomics. Empty strings allowed but discouraged.

`HashMap`/`HashSet` usable — `Eq+Hash` derived, tests cover `prompt_key_equality`, `hash_eq`, `hashset_lookup`, `display_canonical`, `key_for_generic`.

## `ToolId`

Thin newtype `pub struct ToolId(pub String)` around tool identifier (`"write_code"`, `"search_web"`). `Display`, `From<&str>`, `From<String>`, `AsRef<str>`, `Eq+Hash+Serialize`. Gives type safety in policy APIs vs raw `String`.

## `FormeError`

Zero I/O, data-only, `thiserror`:

```rust
pub enum FormeError {
  RegistryNotFound(PromptKey),       // prompt not found for {key}
  ContextBuildFailed(String),        // context build failed: {msg}
  PolicyDenied(ToolId),              // policy denied tool {tool}
  PersistenceFailed(String),         // persistence failed: {msg}
  LlmFailed(String),                 // llm call failed: {msg}
  InvalidInput(String),              // invalid input: {msg}
}
```
Helpers `is_not_found()`, `is_denied()` for test ergonomic branching. Display contains payload, required by QA snapshot. Tests `error_display_*` assert message contains key/tool payload.

## Blanket Impl / Ergonomics

No user boilerplate: `impl State` / `Event` auto via blanket. Example:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum MyState { Idle, Active }
impl Display for MyState { ... } // now S: State automatically
```

## Testing (Wave0 gate)

- `cargo test --lib` 14 passing on cargo 1.75.0 (apt) and 1.97.1 (`~/.cargo/bin/cargo`)
- `cargo clippy --lib` clean (0 warnings after `#[allow]` for `from_iter`)
- `cargo fmt --check` passes
- Tests cover: equality, inequality, display/canonical stability, hash eq/hashset lookup, `key_for` generic, all error variants payload containment, `ToolId` basic/hashset/serde, empty keys allowed, `From` tuple, trait bounds compile check, `serde_roundtrip`.

## Wave0 Leaf Size

S (15-30 min) — 1 trait pair + 2 concrete types + 1 error enum + 14 tests.

## No Teaching / No Wayfinder

`forme-teacher` subcrate explicitly out of scope per user feedback. Wayfinder types (`Destination`, `Notes`, `Decisions`) are product planning only, not shipped. Public API generic only.

## Usage With New Toolchain

```bash
export PATH="$HOME/.cargo/bin:$PATH"   # cargo 1.97.1, rustc 1.97.1
cargo test --lib
```

Commit: `673f782 Add core: Event, State, PromptKey, ToolId, FormeError` + style fixes `835ec4d`.

