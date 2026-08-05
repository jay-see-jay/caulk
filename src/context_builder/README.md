# context_builder — trait

Wave 1 = trait + NoopBuilder only. No product impls (FileSnippet, Paragraph = Wave 2).

Goal: pluggable context from current event + sliding window history, feeding rig `preamble`.

Trait sketch:

pub trait ContextBuilder<E: Event> : Send + Sync {
    fn build(&self, event: &E, history: &[E]) -> Result<String, FormeError>;
}

- `Context = String` for Wave 1 (simplest). Future associated type = serde_json::Value possible, but String keeps Rig `preamble: &str` happy.
- `history: &[E]` last N events, caller decides window.
- `NoopBuilder` returns "" .

Error: `ContextBuildFailed(String)`.

Tests:
- Noop returns empty
- custom builder concatenates history deterministic
- snapshot test deterministic JSON
- error propagation returns ContextBuildFailed
- Send+Sync bound compile check
- generic over custom Event

Object-safe? No because generic E; users needing dyn use enum wrapper.


