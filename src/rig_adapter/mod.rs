//! rig_adapter — generic LLM adapter, mockable for tests
//!
//! Wave 2: thin async trait over any LLM client (Rig, OpenAI, mock).
//! Real Rig integration lives behind the `rig` feature (Wave 5). For now
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
//! Real adapters in Wave 5 wrap `rig-core` / `rig-agent` and forward:
//! `agent.preamble(prompt).prompt(context).await` via the low-level
//! `CompletionModel` API.

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

// ── Real Rig adapters (behind `rig` feature) ─────────────────────────────

#[cfg(feature = "rig")]
pub mod rig {
    //! Real Rig adapters — thin wrappers over `rig-core` / `rig-agent`.
    //!
    //! Two layers:
    //! * `RigModelAdapter<M>` — low-level `CompletionModel` (from `rig-core`);
    //!   builds `completion_request(context).preamble(prompt).build()` and
    //!   joins `AssistantContent::Text` outputs.
    //! * `RigAgentAdapter<A>` — high-level `Agent` (from `rig-agent`/`rig`);
    //!   uses the classic `Prompt` trait (`agent.prompt(text).await`).
    //!
    //! Both implement `LlmAdapter<E>` for any `E: Event`, keeping the generic
    //! forme runtime unaware of the underlying provider.

    use std::sync::Arc;

    use crate::core::{Event, FormeError};

    use crate::rig_adapter::LlmAdapter;

    // ---- Low-level CompletionModel adapter (rig-core) ----

    /// Low-level wrapper around any `rig-core` `CompletionModel`.
    ///
    /// `prompt` (from registry, typed State×Event) becomes the preamble /
    /// system instruction; `context` (from builder) becomes the user message.
    /// If `context` is empty, `prompt` is used as the user message to avoid
    /// sending an empty prompt to the provider.
    pub struct RigModelAdapter<M>
    where
        M: rig_core::completion::CompletionModel + Clone + Send + Sync + 'static,
    {
        model: M,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
    }

    impl<M> RigModelAdapter<M>
    where
        M: rig_core::completion::CompletionModel + Clone + Send + Sync + 'static,
    {
        /// Create adapter from a `CompletionModel`.
        ///
        /// Example:
        /// ```no_run
        /// # #[cfg(feature = "rig")]
        /// # {
        /// use rig_core::client::{CompletionClient, ProviderClient};
        /// use rig_core::providers::openai;
        /// use forme::rig_adapter::rig::RigModelAdapter;
        ///
        /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
        /// let client = openai::Client::from_env()?;
        /// let model = client.completion_model(openai::GPT_5_2);
        /// let adapter = RigModelAdapter::new(model);
        /// # Ok(())
        /// # }
        /// # }
        /// ```
        pub fn new(model: M) -> Self {
            Self {
                model,
                temperature: None,
                max_tokens: None,
            }
        }

        /// Set temperature for subsequent calls (builder-style).
        pub fn with_temperature(mut self, t: f64) -> Self {
            self.temperature = Some(t);
            self
        }

        /// Set max_tokens for subsequent calls.
        pub fn with_max_tokens(mut self, mt: u64) -> Self {
            self.max_tokens = Some(mt);
            self
        }
    }

    impl<M> Clone for RigModelAdapter<M>
    where
        M: rig_core::completion::CompletionModel + Clone + Send + Sync + 'static,
    {
        fn clone(&self) -> Self {
            Self {
                model: self.model.clone(),
                temperature: self.temperature,
                max_tokens: self.max_tokens,
            }
        }
    }

    impl<M> std::fmt::Debug for RigModelAdapter<M>
    where
        M: rig_core::completion::CompletionModel + Clone + Send + Sync + 'static,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RigModelAdapter")
                .field("temperature", &self.temperature)
                .field("max_tokens", &self.max_tokens)
                .finish_non_exhaustive()
        }
    }

    impl<E, M> LlmAdapter<E> for RigModelAdapter<M>
    where
        E: Event,
        M: rig_core::completion::CompletionModel + Clone + Send + Sync + 'static,
        M::Response: Send + Sync + 'static,
    {
        fn call(
            &self,
            prompt: String,
            context: String,
        ) -> impl std::future::Future<Output = Result<String, FormeError>> + Send {
            let model = self.model.clone();
            let temp = self.temperature;
            let max_toks = self.max_tokens;
            async move {
                use rig_core::completion::AssistantContent;

                // prompt = system (preamble), context = user. If context empty, use prompt as user input.
                let (user_msg, preamble_opt) = if context.trim().is_empty() {
                    (prompt.clone(), None)
                } else {
                    (context.clone(), Some(prompt.clone()))
                };

                let mut builder = model.completion_request(user_msg);
                if let Some(pre) = preamble_opt {
                    builder = builder.preamble(pre);
                }
                if let Some(t) = temp {
                    builder = builder.temperature(t);
                }
                if let Some(mt) = max_toks {
                    builder = builder.max_tokens(mt);
                }

                let request = builder.build();
                let response = model
                    .completion(request)
                    .await
                    .map_err(|e| FormeError::LlmFailed(e.to_string()))?;

                // Extract all Text contents, ignore ToolCall/Reasoning for thin adapter.
                let mut texts = Vec::new();
                for item in response.choice.iter() {
                    if let AssistantContent::Text(t) = item {
                        texts.push(t.text.clone());
                    }
                    // ignore other variants for simplicity — thin adapter returns empty if no text
                }

                if texts.is_empty() {
                    Ok(String::new())
                } else {
                    Ok(texts.join("\n"))
                }
            }
        }
    }

    // ---- Closure-based generic adapter (useful for rig::Agent) ----

    /// Adapter that wraps any async closure `Fn(prompt, ctx) -> Future<Output=Result<String,FormeError>>`.
    ///
    /// This is the bridge for `rig::Agent` / `rig_agent::Agent` which already
    /// has a `.prompt(text).await` method returning `String`. Users can do:
    ///
    /// ```no_run
    /// # #[cfg(feature = "rig")]
    /// # {
    /// use std::sync::Arc;
    /// use forme::rig_adapter::rig::ClosureAdapter;
    ///
    /// # async fn example(agent: Arc<rig::Agent<rig::providers::openai::CompletionModel>>) -> forme::core::FormeError {
    /// let adapter = ClosureAdapter::from_fn(move |prompt, context| {
    ///     let agent = Arc::clone(&agent);
    ///     async move {
    ///         // Merge forme prompt (system) + context (user) into agent call:
    ///         // simplest: ignore prompt as preamble already baked, or recreate agent per-call.
    ///         // Here we just prompt with context, falling back to prompt if empty.
    ///         let user = if context.is_empty() { prompt } else { context };
    ///         agent.prompt(user).await.map_err(|e| forme::core::FormeError::LlmFailed(e.to_string()))
    ///     }
    /// });
    /// # unreachable!()
    /// # }
    /// # }
    /// ```
    #[allow(clippy::type_complexity)]
    pub struct ClosureAdapter {
        inner: Arc<
            dyn Fn(
                    String,
                    String,
                ) -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<String, FormeError>> + Send>,
                > + Send
                + Sync,
        >,
    }

    impl ClosureAdapter {
        /// Construct from an async closure.
        pub fn from_fn<F, Fut>(f: F) -> Self
        where
            F: Fn(String, String) -> Fut + Send + Sync + 'static,
            Fut: std::future::Future<Output = Result<String, FormeError>> + Send + 'static,
        {
            Self {
                inner: Arc::new(move |p, c| Box::pin(f(p, c))),
            }
        }
    }

    impl std::fmt::Debug for ClosureAdapter {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ClosureAdapter").finish_non_exhaustive()
        }
    }

    impl Clone for ClosureAdapter {
        fn clone(&self) -> Self {
            Self {
                inner: Arc::clone(&self.inner),
            }
        }
    }

    impl<E: Event> LlmAdapter<E> for ClosureAdapter {
        fn call(
            &self,
            prompt: String,
            context: String,
        ) -> impl std::future::Future<Output = Result<String, FormeError>> + Send {
            let inner = Arc::clone(&self.inner);
            async move { inner(prompt, context).await }
        }
    }

    // ---- High-level Agent adapter using rig-agent's Prompt trait ----

    /// Wrapper over a `rig_agent::Agent` (or `rig::Agent`) using its `Prompt` trait.
    ///
    /// This adapter bakes the forme `prompt` as preamble only when empty agent
    /// already has no preamble? For simplicity, if `context` is non-empty it prompts
    /// with `context`, otherwise with `prompt`. Users needing per-call preamble
    /// rebuild should use `RigModelAdapter` or `ClosureAdapter` with a factory
    /// that re-creates the agent per call: `client.agent(model).preamble(&prompt).build()`.
    pub struct RigAgentAdapter<A>
    where
        A: rig_agent::completion::Prompt + Send + Sync + 'static,
    {
        agent: Arc<A>,
    }

    impl<A> RigAgentAdapter<A>
    where
        A: rig_agent::completion::Prompt + Send + Sync + 'static,
    {
        /// Create from an existing agent (Arc-shared).
        pub fn new(agent: Arc<A>) -> Self {
            Self { agent }
        }

        /// Convenience: from owned agent (wraps in Arc internally).
        pub fn from_agent(agent: A) -> Self {
            Self {
                agent: Arc::new(agent),
            }
        }
    }

    impl<A> Clone for RigAgentAdapter<A>
    where
        A: rig_agent::completion::Prompt + Send + Sync + 'static,
    {
        fn clone(&self) -> Self {
            Self {
                agent: Arc::clone(&self.agent),
            }
        }
    }

    impl<A> std::fmt::Debug for RigAgentAdapter<A>
    where
        A: rig_agent::completion::Prompt + Send + Sync + 'static,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RigAgentAdapter").finish_non_exhaustive()
        }
    }

    impl<E, A> LlmAdapter<E> for RigAgentAdapter<A>
    where
        E: Event,
        A: rig_agent::completion::Prompt + Send + Sync + 'static,
    {
        fn call(
            &self,
            prompt: String,
            context: String,
        ) -> impl std::future::Future<Output = Result<String, FormeError>> + Send {
            let agent = Arc::clone(&self.agent);
            async move {
                let user_msg = if context.trim().is_empty() {
                    prompt
                } else {
                    context
                };
                agent
                    .prompt(user_msg)
                    .await
                    .map_err(|e| FormeError::LlmFailed(e.to_string()))
            }
        }
    }

    // ---- Ergonomic helper: build rig client.agent with preamble per call ----

    /// Factory that recreates a Rig agent per call with `preamble = prompt`.
    ///
    /// Useful when your forme `prompt` changes per State×Event and you need
    /// a fresh Rig agent with that preamble each step.
    ///
    /// ```no_run
    /// # #[cfg(feature = "rig")]
    /// # {
    /// use std::sync::Arc;
    /// use rig::client::{CompletionClient, ProviderClient};
    /// use rig::providers::openai;
    /// use forme::rig_adapter::rig::RigAgentFactory;
    ///
    /// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = openai::Client::from_env()?;
    /// let factory = RigAgentFactory::new(client.clone(), openai::GPT_5_2);
    /// // forme runtime will call `factory.call(prompt, context)` internally via LlmAdapter
    /// # Ok(())
    /// # }
    /// # }
    /// ```
    pub struct RigAgentFactory<C, M>
    where
        C: Clone + Send + Sync + 'static,
        M: Clone + Send + Sync + 'static,
    {
        client: C,
        model_name: M,
    }

    impl<C, M> RigAgentFactory<C, M>
    where
        C: Clone + Send + Sync + 'static,
        M: Clone + Send + Sync + 'static,
    {
        /// Create factory from client + model identifier (e.g. `openai::GPT_5_2`).
        pub fn new(client: C, model_name: M) -> Self {
            Self { client, model_name }
        }
    }

    // The factory itself implements a closure-like Adapter via ClosureAdapter,
    // but users can also use it directly to build agents.
    #[cfg(feature = "rig")]
    impl<C, M> RigAgentFactory<C, M>
    where
        C: rig::client::AgentClientExt + Clone + Send + Sync + 'static,
        M: Into<String> + Clone + Send + Sync + 'static,
        C::CompletionModel: rig::completion::CompletionModel + Clone + Send + Sync + 'static,
        <C::CompletionModel as rig::completion::CompletionModel>::Response: Send + Sync,
    {
        /// Convert into a `ClosureAdapter` suitable for `Runner`.
        ///
        /// Each forme step rebuilds a fresh Rig agent with `preamble = prompt`
        /// (the system) and prompts with `context` (user), falling back to
        /// `prompt` when context empty.
        pub fn into_adapter(self) -> ClosureAdapter {
            let factory = Arc::new(self);
            ClosureAdapter::from_fn(move |prompt, context| {
                let factory = Arc::clone(&factory);
                async move {
                    use rig::completion::Prompt as _;
                    let user_msg = if context.trim().is_empty() {
                        prompt.clone()
                    } else {
                        context.clone()
                    };
                    // Recreate agent per call so preamble = prompt (forme system)
                    let agent = factory
                        .client
                        .agent(factory.model_name.clone())
                        .preamble(&prompt)
                        .build();
                    agent
                        .prompt(user_msg)
                        .await
                        .map_err(|e| FormeError::LlmFailed(e.to_string()))
                }
            })
        }
    }

    #[cfg(feature = "rig")]
    impl<C, M> std::fmt::Debug for RigAgentFactory<C, M>
    where
        C: Clone + Send + Sync + 'static,
        M: Clone + Send + Sync + 'static,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RigAgentFactory").finish_non_exhaustive()
        }
    }

    #[cfg(feature = "rig")]
    impl<C, M> Clone for RigAgentFactory<C, M>
    where
        C: Clone + Send + Sync + 'static,
        M: Clone + Send + Sync + 'static,
    {
        fn clone(&self) -> Self {
            Self {
                client: self.client.clone(),
                model_name: self.model_name.clone(),
            }
        }
    }

    // Re-export a simple prelude for ergonomic use
    #[cfg(feature = "rig")]
    pub mod prelude {
        pub use super::{ClosureAdapter, RigAgentAdapter, RigAgentFactory, RigModelAdapter};
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

    #[cfg(feature = "rig")]
    mod rig_tests {
        use super::*;
        use crate::rig_adapter::rig::ClosureAdapter;

        #[test]
        fn closure_adapter_happy() {
            let adapter = ClosureAdapter::from_fn(|p, c| async move { Ok(format!("{p}|{c}")) });
            let out = block_on(<ClosureAdapter as LlmAdapter<Ev>>::call(
                &adapter,
                "prompt".into(),
                "ctx".into(),
            ))
            .unwrap();
            assert_eq!(out, "prompt|ctx");
        }

        #[test]
        fn closure_adapter_is_send_sync() {
            assert_send_sync::<ClosureAdapter>();
        }

        // RigModelAdapter compile-check (no API key needed for construction)
        // We can't instantiate a real CompletionModel without provider deps,
        // but we can ensure the type is Send+Sync when M is.
    }
}
