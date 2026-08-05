//! graph_flow_adapter — bridge `forme` runtime into `graph-flow` workflows
//!
//! Wave 5: typed `forme` `State`/`Event` → `graph-flow` `GraphBuilder`/`Task`.
//!
//! ## Design
//!
//! * `FormeGraphBuilder<S>` — thin wrapper over `graph_flow::GraphBuilder`
//!   that preserves typed-state semantics while delegating edge/task registration.
//! * `From<forme::runtime::NextAction<S>> for graph_flow::NextAction` — the
//!   push-model transition: forme decides `Next/Continue/Branch/Transition/Halt`,
//!   graph-flow executes `Continue/GoTo/End`.
//! * `FormeTask<S,E,R,B,P,L>` — `graph_flow::Task` impl that runs
//!   `Runner::prepare(state, event, history)` deterministically inside the
//!   workflow step, stores prompt/context in the shared `graph_flow::Context`,
//!   and returns `TaskResult` with mapped `NextAction`.
//!
//! No teaching logic. No wayfinder types leak. All code behind `graph-flow` feature.

#[cfg(feature = "graph-flow")]
mod inner {
    use std::fmt;
    use std::marker::PhantomData;
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use graph_flow::NextAction as GfNextAction;
    use graph_flow::{Context, Graph, GraphBuilder, GraphError, Task, TaskResult};

    use crate::context_builder::ContextBuilder;
    use crate::core::{Event, FormeError, State};
    use crate::policy::Policy;
    use crate::prompt_registry::Registry;
    use crate::rig_adapter::LlmAdapter;
    use crate::runtime::{NextAction as FormeNextAction, Runner};

    // ═══════════════════════════════════════════════════════════════════════
    // FormeGraphBuilder
    // ═══════════════════════════════════════════════════════════════════════

    /// Thin typed wrapper around `graph_flow::GraphBuilder`.
    ///
    /// Preserves `S: State` type parameter so callers can't accidentally mix
    /// graphs from different products (Ferriswheel vs Inkwell etc) without it
    /// being visible in type signatures. Underlying builder is untyped (string
    /// task ids), as is `graph-flow` itself — this wrapper keeps the forme
    /// flavor without fighting the library.
    ///
    /// Example:
    /// ```no_run
    /// # #[cfg(feature = "graph-flow")]
    /// # {
    /// use std::sync::Arc;
    /// use forme::graph_flow_adapter::FormeGraphBuilder;
    /// use forme::core::State;
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    /// enum MyState { Idle, Done }
    /// impl std::fmt::Display for MyState {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         write!(f, "{:?}", self)
    ///     }
    /// }
    ///
    /// let builder = FormeGraphBuilder::<MyState>::new("hello");
    /// // builder.add_task(my_task).build()...
    /// # }
    /// ```
    pub struct FormeGraphBuilder<S>
    where
        S: State,
    {
        inner: GraphBuilder,
        _phantom: PhantomData<S>,
    }

    impl<S> FormeGraphBuilder<S>
    where
        S: State,
    {
        /// Create builder with workflow id.
        pub fn new(id: impl Into<String>) -> Self {
            Self {
                inner: GraphBuilder::new(id),
                _phantom: PhantomData,
            }
        }

        /// Add a `graph-flow` task (any `Task` impl, including `FormeTask`).
        pub fn add_task(self, task: Arc<dyn Task>) -> Self {
            Self {
                inner: self.inner.add_task(task),
                _phantom: PhantomData,
            }
        }

        /// Convenience: add a forme task wrapping a `Runner`.
        pub fn add_state_task<E, R, B, P, L>(self, task: Arc<FormeTask<S, E, R, B, P, L>>) -> Self
        where
            E: Event,
            R: Registry + fmt::Debug + Send + Sync + 'static,
            B: ContextBuilder<E> + fmt::Debug + Send + Sync + 'static,
            P: Policy<S> + fmt::Debug + Send + Sync + 'static,
            L: LlmAdapter<E> + fmt::Debug + Send + Sync + 'static,
        {
            // `FormeTask` already impls `Task`, so coerce Arc<FormeTask> → Arc<dyn Task>
            let dyn_task: Arc<dyn Task> = task as Arc<dyn Task>;
            Self {
                inner: self.inner.add_task(dyn_task),
                _phantom: PhantomData,
            }
        }

        /// Add edge `from → to`.
        pub fn add_edge(self, from: impl Into<String>, to: impl Into<String>) -> Self {
            Self {
                inner: self.inner.add_edge(from, to),
                _phantom: PhantomData,
            }
        }

        /// Conditional edge: `from` → `yes` if `condition(ctx)`, else `no`.
        pub fn add_conditional_edge<F>(
            self,
            from: impl Into<String>,
            condition: F,
            yes: impl Into<String>,
            no: impl Into<String>,
        ) -> Self
        where
            F: Fn(&Context) -> bool + Send + Sync + 'static,
        {
            Self {
                inner: self.inner.add_conditional_edge(from, condition, yes, no),
                _phantom: PhantomData,
            }
        }

        /// Set explicit start task id.
        pub fn set_start_task(self, task_id: impl Into<String>) -> Self {
            Self {
                inner: self.inner.set_start_task(task_id),
                _phantom: PhantomData,
            }
        }

        /// Task timeout override.
        pub fn with_task_timeout(self, timeout: Duration) -> Self {
            Self {
                inner: self.inner.with_task_timeout(timeout),
                _phantom: PhantomData,
            }
        }

        /// Chain length guard for `ContinueAndExecute`.
        pub fn with_max_execution_steps(self, max: usize) -> Self {
            Self {
                inner: self.inner.with_max_execution_steps(max),
                _phantom: PhantomData,
            }
        }

        /// Build immutable `Graph`, validating edges.
        pub fn build(self) -> Result<Graph, GraphError> {
            self.inner.build()
        }
    }

    impl<S> fmt::Debug for FormeGraphBuilder<S>
    where
        S: State + fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("FormeGraphBuilder")
                .field("id", &"graph")
                .finish_non_exhaustive()
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NextAction conversion
    // ═══════════════════════════════════════════════════════════════════════

    /// Map forme's minimal `NextAction<S>` → graph-flow's richer `NextAction`.
    ///
    /// * `Next | Continue` → `Continue` (step-by-step, caller drives)
    /// * `Branch(S) | Transition(S)` → `GoTo(<state>.to_string())`
    /// * `Halt` → `End`
    ///
    /// `ContinueAndExecute` is not a forme concept — callers who want it can
    /// construct `graph_flow::NextAction::ContinueAndExecute` manually for a step.
    impl<S> From<FormeNextAction<S>> for GfNextAction
    where
        S: State,
    {
        fn from(act: FormeNextAction<S>) -> Self {
            match act {
                FormeNextAction::Next | FormeNextAction::Continue => GfNextAction::Continue,
                FormeNextAction::Branch(s) | FormeNextAction::Transition(s) => {
                    GfNextAction::GoTo(s.to_string())
                }
                FormeNextAction::Halt => GfNextAction::End,
            }
        }
    }

    /// Helper to convert with explicit handling: if caller needs `ContinueAndExecute`
    /// for `Next` they can use this.
    pub fn forme_next_to_gf<S>(act: FormeNextAction<S>, eager: bool) -> GfNextAction
    where
        S: State,
    {
        match act {
            FormeNextAction::Next if eager => GfNextAction::ContinueAndExecute,
            FormeNextAction::Continue if eager => GfNextAction::ContinueAndExecute,
            other => other.into(),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FormeTask
    // ═══════════════════════════════════════════════════════════════════════

    /// A `graph-flow` `Task` that runs `Runner::prepare` deterministically.
    ///
    /// Mirrors how Wave 4 examples used `Runner`:
    /// * `state` — current typed state (Display = task id)
    /// * `event` — current typed event driving prompt choice
    /// * `history` — sliding window of prior events (empty for simplest use)
    /// * `next_action` — how this task wishes to continue (default `Next`)
    ///
    /// The task's `run` method:
    /// 1. Calls `runner.prepare(&state, &event, &history)` (pure, no I/O)
    /// 2. Stores `prompt`, `context`, `tool_plan` into `graph_flow::Context`
    /// 3. Returns `TaskResult` with `NextAction` converted from `forme` to graph-flow.
    ///
    /// LLM calling is deferred to a separate task or `FormeLlmTask` if needed.
    pub struct FormeTask<S, E, R, B, P, L>
    where
        S: State,
        E: Event,
        R: Registry,
        B: ContextBuilder<E>,
        P: Policy<S>,
        L: LlmAdapter<E>,
    {
        runner: Arc<Runner<S, E, R, B, P, L>>,
        id: String,
        state: S,
        event: E,
        history: Vec<E>,
        next: FormeNextAction<S>,
    }

    impl<S, E, R, B, P, L> FormeTask<S, E, R, B, P, L>
    where
        S: State,
        E: Event,
        R: Registry,
        B: ContextBuilder<E>,
        P: Policy<S>,
        L: LlmAdapter<E>,
    {
        /// Create a new forme task.
        ///
        /// `id` defaults to `state.to_string()` if not supplied via `with_id`.
        pub fn new(runner: Arc<Runner<S, E, R, B, P, L>>, state: S, event: E) -> Self {
            let id = state.to_string();
            Self {
                runner,
                id,
                state,
                event,
                history: Vec::new(),
                next: FormeNextAction::Next,
            }
        }

        /// Override task id (must be unique in graph).
        pub fn with_id(mut self, id: impl Into<String>) -> Self {
            self.id = id.into();
            self
        }

        /// Set history window.
        pub fn with_history(mut self, history: Vec<E>) -> Self {
            self.history = history;
            self
        }

        /// Set how this task should continue.
        pub fn with_next(mut self, next: FormeNextAction<S>) -> Self {
            self.next = next;
            self
        }

        /// Accessors for testing / wiring.
        pub fn state(&self) -> &S {
            &self.state
        }

        pub fn event(&self) -> &E {
            &self.event
        }

        pub fn runner(&self) -> &Arc<Runner<S, E, R, B, P, L>> {
            &self.runner
        }
    }

    impl<S, E, R, B, P, L> fmt::Debug for FormeTask<S, E, R, B, P, L>
    where
        S: State + fmt::Debug,
        E: Event + fmt::Debug,
        R: Registry + fmt::Debug,
        B: ContextBuilder<E> + fmt::Debug,
        P: Policy<S> + fmt::Debug,
        L: LlmAdapter<E> + fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("FormeTask")
                .field("id", &self.id)
                .field("state", &self.state)
                .field("event", &self.event)
                .field("next", &self.next)
                .finish_non_exhaustive()
        }
    }

    // Impl Task via async_trait
    #[async_trait]
    impl<S, E, R, B, P, L> Task for FormeTask<S, E, R, B, P, L>
    where
        S: State + Send + Sync + 'static,
        E: Event + Send + Sync + 'static,
        R: Registry + Send + Sync + 'static,
        B: ContextBuilder<E> + Send + Sync + 'static,
        P: Policy<S> + Send + Sync + 'static,
        L: LlmAdapter<E> + Send + Sync + 'static,
    {
        fn id(&self) -> &str {
            &self.id
        }

        async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
            // Deterministic prepare — no LLM call
            let prepared = self
                .runner
                .prepare(&self.state, &self.event, &self.history)
                .map_err(|e: FormeError| {
                    graph_flow::GraphError::TaskExecutionFailed(format!(
                        "forme prepare failed ({}): {}",
                        self.id, e
                    ))
                })?;

            // Surface prepared data into shared Context for downstream tasks / LLM
            // Ignore errors from context.set (unlikely) — map to GraphError.
            if let Err(e) = context.set("forme.prompt", prepared.prompt.clone()) {
                return Err(graph_flow::GraphError::TaskExecutionFailed(format!(
                    "context set prompt failed: {}",
                    e
                )));
            }
            if let Err(e) = context.set("forme.context", prepared.context.clone()) {
                return Err(graph_flow::GraphError::TaskExecutionFailed(format!(
                    "context set context failed: {}",
                    e
                )));
            }
            if let Err(e) = context.set("forme.prompt_key", prepared.key.canonical()) {
                return Err(graph_flow::GraphError::TaskExecutionFailed(format!(
                    "context set prompt_key failed: {}",
                    e
                )));
            }
            // Store tool_plan as json list for debugging
            let tools_json = serde_json::to_string(
                &prepared
                    .tool_plan
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>(),
            )
            .unwrap_or_else(|_| "[]".to_string());
            let _ = context.set("forme.tools", tools_json);

            // Also record state/event for observability
            let _ = context.set("forme.state", self.state.to_string());
            let _ = context.set("forme.event", self.event.to_string());

            let gf_next: GfNextAction = self.next.clone().into();

            // TaskResult with response = prompt (deterministic placeholder)
            // Real LLM response would come from a downstream task using FormeLlmTask or rig adapter.
            Ok(TaskResult::new(
                Some(format!("prepared: {}", prepared.key)),
                gf_next,
            ))
        }
    }

    /// Variant that does full `step` including LLM call.
    ///
    /// Useful when you want a single graph-flow task to both prepare and call LLM.
    pub struct FormeLlmTask<S, E, R, B, P, L>
    where
        S: State,
        E: Event,
        R: Registry,
        B: ContextBuilder<E>,
        P: Policy<S>,
        L: LlmAdapter<E>,
    {
        runner: Arc<Runner<S, E, R, B, P, L>>,
        id: String,
        state: S,
        event: E,
        history: Vec<E>,
        next: FormeNextAction<S>,
    }

    impl<S, E, R, B, P, L> FormeLlmTask<S, E, R, B, P, L>
    where
        S: State,
        E: Event,
        R: Registry,
        B: ContextBuilder<E>,
        P: Policy<S>,
        L: LlmAdapter<E>,
    {
        pub fn new(runner: Arc<Runner<S, E, R, B, P, L>>, state: S, event: E) -> Self {
            let id = state.to_string();
            Self {
                runner,
                id,
                state,
                event,
                history: Vec::new(),
                next: FormeNextAction::Next,
            }
        }

        pub fn with_id(mut self, id: impl Into<String>) -> Self {
            self.id = id.into();
            self
        }

        pub fn with_history(mut self, history: Vec<E>) -> Self {
            self.history = history;
            self
        }

        pub fn with_next(mut self, next: FormeNextAction<S>) -> Self {
            self.next = next;
            self
        }
    }

    impl<S, E, R, B, P, L> fmt::Debug for FormeLlmTask<S, E, R, B, P, L>
    where
        S: State + fmt::Debug,
        E: Event + fmt::Debug,
        R: Registry + fmt::Debug,
        B: ContextBuilder<E> + fmt::Debug,
        P: Policy<S> + fmt::Debug,
        L: LlmAdapter<E> + fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("FormeLlmTask")
                .field("id", &self.id)
                .field("state", &self.state)
                .field("next", &self.next)
                .finish_non_exhaustive()
        }
    }

    #[async_trait]
    impl<S, E, R, B, P, L> Task for FormeLlmTask<S, E, R, B, P, L>
    where
        S: State + Send + Sync + 'static,
        E: Event + Send + Sync + 'static,
        R: Registry + Send + Sync + 'static,
        B: ContextBuilder<E> + Send + Sync + 'static,
        P: Policy<S> + Send + Sync + 'static,
        L: LlmAdapter<E> + Send + Sync + 'static,
    {
        fn id(&self) -> &str {
            &self.id
        }

        async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
            // Full step includes LLM (may be MockLlm or real Rig)
            let out = self
                .runner
                .step(&self.state, &self.event, &self.history)
                .await
                .map_err(|e| {
                    graph_flow::GraphError::TaskExecutionFailed(format!(
                        "forme step failed ({}): {}",
                        self.id, e
                    ))
                })?;

            let _ = context.set("forme.prompt", out.prompt.clone());
            let _ = context.set("forme.context", out.context.clone());
            let _ = context.set("forme.response", out.llm_response.clone());
            let _ = context.set("forme.state", out.next_state.to_string());

            let gf_next: GfNextAction = self.next.clone().into();

            Ok(TaskResult::new(Some(out.llm_response), gf_next))
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Public re-exports
    // ═══════════════════════════════════════════════════════════════════════

    pub use self::FormeGraphBuilder as GraphBuilderTyped;
}

#[cfg(feature = "graph-flow")]
pub use inner::*;

// When feature disabled, provide minimal empty module to keep lib.rs compiling
#[cfg(not(feature = "graph-flow"))]
pub mod stub {
    // Empty — real types only with feature
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "graph-flow")]
    mod gf_tests {
        use graph_flow::Task;
        use std::sync::Arc;

        use crate::context_builder::NoopBuilder;
        use crate::core::PromptKey;
        use crate::graph_flow_adapter::{FormeGraphBuilder, FormeTask};
        use crate::policy::AllowAllPolicy;
        use crate::prompt_registry::InMemoryRegistry;
        use crate::rig_adapter::MockLlm;
        use crate::runtime::{NextAction as FormeNextAction, Runner};
        use serde::{Deserialize, Serialize};
        use std::collections::HashMap;
        use std::fmt;

        #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        struct S(String);
        impl fmt::Display for S {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        #[derive(Clone, Debug, Serialize, Deserialize)]
        struct E(String);
        impl fmt::Display for E {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        fn block_on<F: std::future::Future>(fut: F) -> F::Output {
            use std::future::Future;
            use std::pin::Pin;
            use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
            fn no_op(_: *const ()) {}
            fn clone_p(_: *const ()) -> RawWaker {
                RawWaker::new(std::ptr::null(), &VTABLE)
            }
            static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_p, no_op, no_op, no_op);
            let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
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
        fn nextaction_into_gf() {
            use graph_flow::NextAction as GfNextAction;
            let cases: Vec<(FormeNextAction<S>, GfNextAction)> = vec![
                (FormeNextAction::Next, GfNextAction::Continue),
                (FormeNextAction::Continue, GfNextAction::Continue),
                (
                    FormeNextAction::Branch(S("Target".into())),
                    GfNextAction::GoTo("Target".into()),
                ),
                (
                    FormeNextAction::Transition(S("Other".into())),
                    GfNextAction::GoTo("Other".into()),
                ),
                (FormeNextAction::Halt, GfNextAction::End),
            ];

            for (forme_a, expected_gf) in cases {
                let gf: GfNextAction = forme_a.into();
                assert_eq!(gf, expected_gf);
            }
        }

        #[test]
        fn builder_add_and_build() {
            use std::sync::Arc;

            #[derive(Debug)]
            struct Dummy;
            #[async_trait::async_trait]
            impl Task for Dummy {
                fn id(&self) -> &str {
                    "idle"
                }
                async fn run(
                    &self,
                    _ctx: graph_flow::Context,
                ) -> graph_flow::Result<graph_flow::TaskResult> {
                    Ok(graph_flow::TaskResult::new(
                        Some("hi".into()),
                        graph_flow::NextAction::Continue,
                    ))
                }
            }

            let task = Arc::new(Dummy);
            let graph = FormeGraphBuilder::<S>::new("test")
                .add_task(task.clone())
                .add_edge("idle", "idle")
                .build()
                .unwrap();
            assert_eq!(graph.id, "test");
        }

        #[test]
        fn forme_task_prepare_runs() {
            // Prepare data
            let mut map = HashMap::new();
            map.insert(PromptKey::new("Idle", "Hello"), "hello prompt".into());
            let registry = Arc::new(InMemoryRegistry::new(map));
            let builder = Arc::new(NoopBuilder);
            let policy = Arc::new(AllowAllPolicy);
            let llm = MockLlm::with_response("canned");
            let runner = Arc::new(Runner::new(registry, builder, policy, llm));

            let task = Arc::new(FormeTask::new(
                Arc::clone(&runner),
                S("Idle".into()),
                E("Hello".into()),
            ));

            // GraphBuilder + run via block_on over Task::run
            let ctx = graph_flow::Context::new();
            let out = block_on(task.run(ctx.clone())).unwrap();
            // response contains prepared key
            assert!(out.response.unwrap().contains("Idle::Hello"));

            // Context got populated
            let prompt: Option<String> = ctx.get("forme.prompt");
            assert_eq!(prompt.as_deref(), Some("hello prompt"));
        }
    }
}
