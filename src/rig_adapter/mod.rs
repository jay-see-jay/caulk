//! rig_adapter — generic LLM adapter, mockable for tests
//!
//! Wave 2: thin async trait over any LLM client (Rig, OpenAI, mock).
//! Real Rig integration lives behind the `rig` feature (Wave 3). For now
//! this module provides the trait and a `MockLlm` that satisfies it.
//!
//! Why generic over `E: Event`?
//! Products (Ferriswheel, Inkwell, Loom) define their own `Event` types.
//! A teaching policy or a cursor-aware context builder might want different
//! adapter behaviour per event shape, but the LLM call itself is just
//! `prompt + context → completion`. Making the adapter generic over `E`
//! lets a product provide an event-specific policy while the core runtime
//! (`forme`) stays unaware of product events.
//!
//! This mirrors how `ContextBuilder<E>` and `Policy<S>` are generic.
//! `MockLlm` implements `LlmAdapter<E>` for *any* `E`, so tests stay generic.
//! Real adapters in Wave 3 will wrap `rig-core`’s `Agent` and forward:
//! `agent.preamble(prompt).context(&context).prompt(event_text).call()`.
//! Error contract: all I/O or model errors map to `FormeError::LlmFailed`.

use std::sync::Arc;

use crate::core::{Event, FormeError};

/// Async LLM call — prompt + context → completion.
///
/// `E: Event` is phantom for product specialization. The method itself
/// does not take `&E` because Rig’s preamble/context pattern already carries
/// the event text via the caller (the runtime glues `registry.get(key)` +
/// `builder.build(event, history)` into prompt/context). Keeping `E` on the trait
/// allows a product to impl `LlmAdapter<MyEvent>` differently without changing
/// runtime call sites.
///
/// Implementations must be `Send + Sync` (supertrait) so they can be shared
/// across async tasks and graph-flow edges.
pub trait LlmAdapter<E: Event>: Send + Sync {
    /// Call the underlying model.
    ///
    /// `prompt` — resolved from `PromptKey` (typed state × event).
    /// `context` — built by `ContextBuilder<E>`.
    /// Returns the raw completion string.
    fn call(
        &self,
        prompt: String,
        context: String,
    ) -> impl std::future::Future<Output = Result<String, FormeError>> + Send;
}

// ── MockLlm ───────────────────────────────────────────────────────────────

/// In-memory mock LLM for tests and examples.
///
/// Configurable per construction:
///
/// * `MockLlm::with_response("canned")` → always `Ok(canned.clone())`
/// * `MockLlm::with_fn(|prompt, ctx| { … })` → custom closure, can inspect
///   prompt/context and return `Ok` or `Err`
/// * `MockLlm::with_error("boom")` → always `Err(LlmFailed("boom"))`
///
/// `MockLlm` is `Clone` (Arc-shared) so a single configured mock can be
/// shared across threads, tasks, or cloned into a runner. Clones share the
/// same handler but are independently `Send + Sync`.
///
/// `with_fn` closure must be `Send + Sync + 'static` to satisfy the
/// `LlmAdapter` supertrait.
pub struct MockLlm {
    handler: Arc<dyn Fn(String, String) -> Result<String, FormeError> + Send + Sync>,
}

impl std::fmt::Debug for MockLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockLlm").finish_non_exhaustive()
    }
}

impl Clone for MockLlm {
    fn clone(&self) -> Self {
        Self {
            handler: Arc::clone(&self.handler),
        }
    }
}

impl MockLlm {
    /// Empty mock returning empty string.
    pub fn new() -> Self {
        Self::with_response("")
    }

    /// Always returns `response.clone()`.
    pub fn with_response(response: impl Into<String>) -> Self {
        let resp = response.into();
        Self {
            handler: Arc::new(move |_, _| Ok(resp.clone())),
        }
    }

    /// Custom function `(prompt, context) -> Result`.
    ///
    /// Example: `MockLlm::with_fn(|p, c| Ok(format!("p={p} c={c}")))`
    pub fn with_fn<F>(func: F) -> Self
    where
        F: Fn(String, String) -> Result<String, FormeError> + Send + Sync + 'static,
    {
        Self {
            handler: Arc::new(func),
        }
    }

    /// Always errors with `LlmFailed(msg)`.
    pub fn with_error(msg: impl Into<String>) -> Self {
        let msg = msg.into();
        Self {
            handler: Arc::new(move |_, _| Err(FormeError::LlmFailed(msg.clone()))),
        }
    }
}

impl Default for MockLlm {
    fn default() -> Self {
        Self::new()
    }
}

// Every E gets same mock behaviour — this is intentional for generic reuse.
// Products that need event-specific mocking can put the branching inside
// `with_fn` closure inspecting `prompt` / `context`.
impl<E: Event> LlmAdapter<E> for MockLlm {
    fn call(
        &self,
        prompt: String,
        context: String,
    ) -> impl std::future::Future<Output = Result<String, FormeError>> + Send {
        let handler = Arc::clone(&self.handler);
        async move { handler(prompt, context) }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::FormeError;
    use serde::{Deserialize, Serialize};
    use std::fmt;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct Ev(String);
    impl fmt::Display for Ev {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    // Need a runtime for async trait test; tokio is optional behind feature.
    // For Wave 2 we use futures-lite blocking via std: we can poll the future
    // with a tiny executor. Instead of pulling in a crate, we use `tokio` if
    // available, otherwise we use `std::future` via `futures::executor` emulation:
    // we implement a minimal block_on for tests without extra deps.
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        use std::future::Future;
        use std::pin::Pin;
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

        fn dummy_waker() -> Waker {
            fn no_op(_: *const ()) {}
            fn clone(_: *const ()) -> RawWaker {
                RawWaker::new(std::ptr::null(), &VTABLE)
            }
            static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
            unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
        }

        let waker = dummy_waker();
        let mut cx = Context::from_waker(&waker);
        let mut fut = Box::pin(fut);
        loop {
            match Future::poll(Pin::new(&mut fut).as_mut(), &mut cx) {
                Poll::Ready(v) => break v,
                Poll::Pending => std::hint::spin_loop(),
            }
        }
    }

    #[test]
    fn happy_response() {
        let llm = MockLlm::with_response("hello mock");
        let fut = <MockLlm as LlmAdapter<Ev>>::call(&llm, "prompt".into(), "ctx".into());
        let out = block_on(fut).unwrap();
        assert_eq!(out, "hello mock");
    }

    #[test]
    fn happy_with_fn_inspects_prompt_context() {
        let llm = MockLlm::with_fn(|prompt, ctx| Ok(format!("{prompt}|{ctx}")));
        let fut = <MockLlm as LlmAdapter<Ev>>::call(&llm, "P".into(), "C".into());
        let out = block_on(fut).unwrap();
        assert_eq!(out, "P|C");

        // prompt-dependent branching
        let llm2 = MockLlm::with_fn(|prompt, _| {
            if prompt.contains("refund") {
                Ok("refund policy".into())
            } else {
                Ok("general".into())
            }
        });
        let a = block_on(<MockLlm as LlmAdapter<Ev>>::call(
            &llm2,
            "refund request".into(),
            "".into(),
        ))
        .unwrap();
        assert_eq!(a, "refund policy");
        let b = block_on(<MockLlm as LlmAdapter<Ev>>::call(
            &llm2,
            "hello".into(),
            "".into(),
        ))
        .unwrap();
        assert_eq!(b, "general");
    }

    #[test]
    fn error_propagation() {
        let llm = MockLlm::with_error("timeout");
        let fut = <MockLlm as LlmAdapter<Ev>>::call(&llm, "p".into(), "c".into());
        let err = block_on(fut).unwrap_err();
        assert!(matches!(err, FormeError::LlmFailed(_)));
        assert!(err.to_string().contains("timeout"));

        // closure returning custom LlmFailed
        let llm2 = MockLlm::with_fn(|_, _| Err(FormeError::LlmFailed("api 500".into())));
        let err2 = block_on(<MockLlm as LlmAdapter<Ev>>::call(
            &llm2,
            "p".into(),
            "c".into(),
        ))
        .unwrap_err();
        assert!(err2.to_string().contains("api 500"));
    }

    #[test]
    fn send_sync_trait_object_compile_check() {
        assert_send_sync::<MockLlm>();
        assert_send_sync::<Arc<MockLlm>>();

        // MockLlm implements LlmAdapter for any Event
        fn assert_adapter<E: crate::core::Event, A: LlmAdapter<E>>() {}
        assert_adapter::<Ev, MockLlm>();
        assert_adapter::<String, MockLlm>();

        // Boxed should still be Send+Sync (because trait is Send+Sync and MockLlm is)
        fn assert_box_send_sync() {
            assert_send_sync::<MockLlm>();
            // The trait bound itself is Send+Sync, but async trait not object-safe,
            // so we only check concrete type.
        }
        assert_box_send_sync();
    }

    #[test]
    fn clone_shares_handler() {
        let llm = MockLlm::with_response("shared");
        let cloned = llm.clone();
        let a = block_on(<MockLlm as LlmAdapter<Ev>>::call(
            &llm,
            "p".into(),
            "c".into(),
        ))
        .unwrap();
        let b = block_on(<MockLlm as LlmAdapter<Ev>>::call(
            &cloned,
            "p".into(),
            "c".into(),
        ))
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a, "shared");
    }

    #[test]
    fn closure_captures_state_safely() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_cloned = Arc::clone(&counter);
        let llm = MockLlm::with_fn(move |_, _| {
            counter_cloned.fetch_add(1, Ordering::SeqCst);
            Ok("counted".into())
        });

        let _ = block_on(<MockLlm as LlmAdapter<Ev>>::call(
            &llm,
            "a".into(),
            "b".into(),
        ))
        .unwrap();
        let _ = block_on(<MockLlm as LlmAdapter<Ev>>::call(
            &llm,
            "a".into(),
            "b".into(),
        ))
        .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }
}
