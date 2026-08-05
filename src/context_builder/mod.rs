//! context_builder — trait + NoopBuilder (Wave 1)
//!
//! Pluggable context from current event + sliding window history,
//! feeding Rig preamble (`&str`). `Context = String` for simplicity.

use crate::core::{Event, FormeError};

/// Builds context string from current event and history.
///
/// `E: Event` generic, not object-safe. Users needing dynamic dispatch
/// should use an enum wrapper for `E`.
///
/// Implementors must be `Send + Sync`.
pub trait ContextBuilder<E: Event>: Send + Sync {
    /// Build context for `event` given `history`.
    ///
    /// `history` is a caller-chosen sliding window (last N events).
    /// Returns `FormeError::ContextBuildFailed` on failure.
    fn build(&self, event: &E, history: &[E]) -> Result<String, FormeError>;
}

/// No-op builder — returns empty string.
///
/// Useful as default when no external context is needed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NoopBuilder;

impl<E: Event> ContextBuilder<E> for NoopBuilder {
    fn build(&self, _event: &E, _history: &[E]) -> Result<String, FormeError> {
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    use serde::{Deserialize, Serialize};

    // ── helpers ─────────────────────────────────────────────────────

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct SimpleEvent(String);

    impl fmt::Display for SimpleEvent {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct CustomEvent {
        id: u32,
        text: String,
    }

    impl fmt::Display for CustomEvent {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}:{}", self.id, self.text)
        }
    }

    /// Concatenates history with "\n" and appends current event.
    #[derive(Clone, Debug, Default)]
    struct ConcatBuilder;

    impl<E: Event> ContextBuilder<E> for ConcatBuilder {
        fn build(&self, event: &E, history: &[E]) -> Result<String, FormeError> {
            let mut parts: Vec<String> = Vec::with_capacity(history.len() + 1);
            for h in history {
                let s: String = h.to_string();
                parts.push(s);
            }
            let cur: String = event.to_string();
            parts.push(cur);
            let joined: String = parts.join("\n");
            Ok(joined)
        }
    }

    #[derive(Clone, Debug, Default)]
    struct FailingBuilder;

    impl<E: Event> ContextBuilder<E> for FailingBuilder {
        fn build(&self, _event: &E, _history: &[E]) -> Result<String, FormeError> {
            Err(FormeError::ContextBuildFailed("forced failure".into()))
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    // ── tests ───────────────────────────────────────────────────────

    #[test]
    fn noop_returns_empty() {
        let builder: NoopBuilder = NoopBuilder;
        let event: SimpleEvent = SimpleEvent("hello".into());
        let history: Vec<SimpleEvent> = vec![SimpleEvent("one".into()), SimpleEvent("two".into())];
        let result: Result<String, FormeError> = builder.build(&event, &history);
        assert!(result.is_ok());
        let ctx: String = result.unwrap();
        assert_eq!(ctx, "");
    }

    #[test]
    fn noop_returns_empty_no_history() {
        let builder: NoopBuilder = NoopBuilder::default();
        let event: SimpleEvent = SimpleEvent("ev".into());
        let history: Vec<SimpleEvent> = Vec::new();
        let ctx: String = builder.build(&event, &history).unwrap();
        assert_eq!(ctx.len(), 0);
        assert!(ctx.is_empty());
    }

    #[test]
    fn custom_builder_concatenates_history_deterministic() {
        let builder: ConcatBuilder = ConcatBuilder;
        let history: Vec<SimpleEvent> = vec![
            SimpleEvent("a".into()),
            SimpleEvent("b".into()),
            SimpleEvent("c".into()),
        ];
        let event: SimpleEvent = SimpleEvent("d".into());
        let ctx: String = builder.build(&event, &history).unwrap();
        assert_eq!(ctx, "a\nb\nc\nd");

        // deterministic: same input -> same join
        let ctx2: String = builder.build(&event, &history).unwrap();
        assert_eq!(ctx, ctx2);
    }

    #[test]
    fn snapshot_test_deterministic() {
        let builder: ConcatBuilder = ConcatBuilder::default();
        let history: Vec<SimpleEvent> =
            vec![SimpleEvent("first".into()), SimpleEvent("second".into())];
        let event: SimpleEvent = SimpleEvent("current".into());

        let out1: String = builder.build(&event, &history).unwrap();
        let out2: String = builder.build(&event, &history).unwrap();

        // snapshot: fully deterministic string
        let expected: String = "first\nsecond\ncurrent".to_string();
        assert_eq!(out1, expected);
        assert_eq!(out2, expected);
        assert_eq!(out1, out2);
    }

    #[test]
    fn error_propagation_returns_context_build_failed() {
        let builder: FailingBuilder = FailingBuilder;
        let event: SimpleEvent = SimpleEvent("x".into());
        let history: Vec<SimpleEvent> = vec![];
        let result: Result<String, FormeError> = builder.build(&event, &history);
        assert!(result.is_err());
        let err: FormeError = result.unwrap_err();
        let msg: String = err.to_string();
        assert!(msg.contains("context build failed"), "msg was: {msg}");
        assert!(msg.contains("forced failure"), "msg was: {msg}");
        match err {
            FormeError::ContextBuildFailed(s) => {
                assert_eq!(s, "forced failure");
            }
            _ => panic!("expected ContextBuildFailed, got {:?}", err),
        }
    }

    #[test]
    fn send_sync_compile_check() {
        // compile-time bound check: if this compiles, NoopBuilder is Send+Sync
        assert_send_sync::<NoopBuilder>();
        assert_send_sync::<ConcatBuilder>();
        assert_send_sync::<FailingBuilder>();

        fn assert_builder_send_sync<E: Event, B: ContextBuilder<E> + Send + Sync>() {}
        assert_builder_send_sync::<SimpleEvent, NoopBuilder>();
        assert_builder_send_sync::<CustomEvent, ConcatBuilder>();
    }

    #[test]
    fn generic_over_custom_event() {
        let builder: ConcatBuilder = ConcatBuilder;
        let history: Vec<CustomEvent> = vec![
            CustomEvent {
                id: 1,
                text: "hello".into(),
            },
            CustomEvent {
                id: 2,
                text: "world".into(),
            },
        ];
        let event: CustomEvent = CustomEvent {
            id: 3,
            text: "now".into(),
        };
        let ctx: String = builder.build(&event, &history).unwrap();
        // Display is "id:text"
        assert_eq!(ctx, "1:hello\n2:world\n3:now");

        // also Noop works over custom event
        let noop: NoopBuilder = NoopBuilder;
        let empty: String = noop.build(&event, &history).unwrap();
        assert_eq!(empty, "");
    }

    #[test]
    fn noop_generic_over_string_event() {
        // String itself satisfies Event via blanket impl
        let builder: NoopBuilder = NoopBuilder;
        let event: String = "ev".to_string();
        let history: Vec<String> = vec!["a".to_string(), "b".to_string()];
        let ctx: String = builder.build(&event, &history).unwrap();
        assert_eq!(ctx, "");
    }
}
