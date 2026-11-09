//! runtime — generic glue between registry, context builder, policy, and LLM
//!
//! Wave 3: `Runner<S, E, R, B, P, L>` composes the four pluggable parts:
//!
//! * `R: Registry` — `PromptKey(State, Event) → prompt text`
//! * `B: ContextBuilder<E>` — current event + history → context string
//! * `P: Policy<S>` — `state → allowed tools`
//! * `L: LlmAdapter<E>` — `(prompt, context) → llm completion`
//!
//! Why `Arc` for `registry`, `builder`, `policy`?
//!
//! * Shareability — a single registry / policy can be shared across many
//!   runner clones, tasks, or graph-flow edges without cloning large state maps.
//! * `Send + Sync` — `Arc` preserves `Send + Sync` when the inner type is
//!   `Send + Sync` (all our traits require it), so `Runner` itself is
//!   `Send + Sync` and can live on a Tokio runtime or cross threads.
//! * Cheap clone — `Arc::clone` is just an atomic inc.
//!
//! `llm` is owned, not `Arc`, because `MockLlm` already contains an inner
//! `Arc<dyn Fn>` and real Rig adapters may carry non-cloneable state
//! (connections, agents). Callers who need shared LLM can wrap their adapter
//! in `Arc` and implement `LlmAdapter` for `Arc<T>`.
//!
//! Design choices
//! * Generic only — no Ferriswheel / Inkwell / Loom concrete types.
//! * Deterministic — `prepare` is pure of `(state, event, history, registry, builder)`.
//!   Calling it twice with identical inputs yields identical prompt/context/tool_plan.
//! * `step` is async and calls `LlmAdapter::call`. Errors from the LLM are
//!   normalized to `FormeError::LlmFailed`.
//! * Minimal `graph-flow` coupling — we define our own `NextAction<S>`
//!   (`Next`, `Branch(state)`). Real `graph-flow` types can be feature-gated later.
//! * No teacher / planner types leak into public API.

use std::fmt;
use std::sync::Arc;

use crate::context_builder::ContextBuilder;
use crate::core::{Event, FormeError, PromptKey, State, ToolId};
use crate::policy::Policy;
use crate::prompt_registry::Registry;
use crate::rig_adapter::LlmAdapter;
use crate::scorer::{Score, Scorer, Verdict};

/// Which tools are allowed for the current step.
///
/// Wrapper around `Vec<ToolId>` to give type safety and helper methods.
/// Empty `allowed` is valid and means "no tools" for `AllowListPolicy`,
/// or "all tools" for `AllowAllPolicy` / `DenyListPolicy` (those policies
/// return empty slice to signal infinite set). `ToolPlan` stores the slice
/// returned by `Policy::allowed` at step time — deterministic snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPlan {
    pub allowed: Vec<ToolId>,
}

impl ToolPlan {
    pub fn new(allowed: Vec<ToolId>) -> Self {
        Self { allowed }
    }

    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }

    pub fn len(&self) -> usize {
        self.allowed.len()
    }

    pub fn contains(&self, tool: &ToolId) -> bool {
        self.allowed.contains(tool)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ToolId> {
        self.allowed.iter()
    }
}

impl IntoIterator for ToolPlan {
    type Item = ToolId;
    type IntoIter = std::vec::IntoIter<ToolId>;
    fn into_iter(self) -> Self::IntoIter {
        self.allowed.into_iter()
    }
}

impl<'a> IntoIterator for &'a ToolPlan {
    type Item = &'a ToolId;
    type IntoIter = std::slice::Iter<'a, ToolId>;
    fn into_iter(self) -> Self::IntoIter {
        self.allowed.iter()
    }
}

/// Prepared inputs before LLM call — pure and replayable.
///
/// `score` is `Some` only when a `Scorer` is installed on the runner
/// (see `Runner::with_scorer`), or when a missing prompt was turned
/// into silence (see `Runner::with_missing_prompt_silent`). It is
/// `PartialEq` but not `Eq` because `Score` carries an `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct Prepared {
    pub key: PromptKey,
    pub prompt: String,
    pub context: String,
    pub tool_plan: ToolPlan,
    pub score: Option<Score>,
}

/// Full output after LLM call.
///
/// `score` is copied from `Prepared`. When the score's verdict is
/// `Verdict::Silent`, `step` skips the LLM and `llm_response` is empty.
/// `PartialEq` but not `Eq` (see `Prepared`).
#[derive(Clone, Debug, PartialEq)]
pub struct StepOutput<S> {
    /// For now next_state == current_state.clone(). Transition logic
    /// lives in graph-flow layer / product code. Keeping field makes
    /// `step(event) -> (new_state, Prompt, ToolPlan)` shape forward-compatible.
    pub next_state: S,
    pub prompt_key: PromptKey,
    pub prompt: String,
    pub context: String,
    pub llm_response: String,
    pub tool_plan: ToolPlan,
    pub score: Option<Score>,
}

/// Minimal graph-flow `NextAction` stand-in.
///
/// Real `graph-flow` will have a richer enum behind the `graph-flow` feature.
/// For Wave 3 we keep it tiny:
///
/// * `Next` — stay in current state, next prompt is `current_state × event`
/// * `Branch(S)` / `Transition(S)` — move to new state, prompt is `new_state × event`
/// * `Halt` — terminal, no further prompt (still returns a `PromptKey` for logging)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NextAction<S> {
    Next,
    Continue,
    Branch(S),
    Transition(S),
    Halt,
}

impl<S> NextAction<S> {
    pub fn is_halt(&self) -> bool {
        matches!(self, Self::Halt)
    }
}

/// Thin generic runner.
pub struct Runner<S, E, R, B, P, L>
where
    S: State,
    E: Event,
    R: Registry,
    B: ContextBuilder<E>,
    P: Policy<S>,
    L: LlmAdapter<E>,
{
    registry: Arc<R>,
    builder: Arc<B>,
    policy: Arc<P>,
    llm: L,
    /// Optional judgment gate (Wave A). `None` = pre-scorer behaviour.
    scorer: Option<Arc<dyn Scorer<S, E>>>,
    /// When true, a missing prompt means "no opinion for this
    /// (state, event)" instead of `RegistryNotFound`: prepare succeeds
    /// with an empty prompt and a forced `Silent` score.
    missing_prompt_silent: bool,
    _state: std::marker::PhantomData<S>,
    _event: std::marker::PhantomData<E>,
}

impl<S, E, R, B, P, L> Runner<S, E, R, B, P, L>
where
    S: State,
    E: Event,
    R: Registry,
    B: ContextBuilder<E>,
    P: Policy<S>,
    L: LlmAdapter<E>,
{
    /// Create a new runner.
    ///
    /// `Arc` for registry / builder / policy is intentional (see module docs).
    pub fn new(registry: Arc<R>, builder: Arc<B>, policy: Arc<P>, llm: L) -> Self {
        Self {
            registry,
            builder,
            policy,
            llm,
            scorer: None,
            missing_prompt_silent: false,
            _state: std::marker::PhantomData,
            _event: std::marker::PhantomData,
        }
    }

    /// Install a judgment scorer (builder-style; Wave A).
    ///
    /// When set, `prepare` scores the step — after the `ToolPlan`
    /// snapshot, so the scorer judges with the permitted tools visible —
    /// and the `Score` rides in `Prepared` / `StepOutput`. In `step`, a
    /// `Silent` verdict skips the LLM call entirely; every other verdict
    /// behaves exactly as an unscored step.
    pub fn with_scorer(mut self, scorer: Arc<dyn Scorer<S, E>>) -> Self {
        self.scorer = Some(scorer);
        self
    }

    /// Treat a missing prompt as silence instead of an error
    /// (builder-style; Wave A).
    ///
    /// With `yes = true`, a `RegistryNotFound` from the registry no
    /// longer fails `prepare`/`step`: the prompt is empty and the score
    /// is forced to `Silent @ 1.0`, overriding any installed scorer.
    /// This makes "no prompt file for this (state, event)" a deliberate
    /// no-opinion path — the normal case in a realtime event stream,
    /// where most (state, event) pairs should not wake the agent.
    /// Default is `false` (the original hard-error behaviour).
    pub fn with_missing_prompt_silent(mut self, yes: bool) -> Self {
        self.missing_prompt_silent = yes;
        self
    }

    /// Borrowed accessors.
    pub fn registry(&self) -> &R {
        &self.registry
    }

    pub fn builder(&self) -> &B {
        &self.builder
    }

    pub fn policy(&self) -> &P {
        &self.policy
    }

    pub fn llm(&self) -> &L {
        &self.llm
    }

    /// The installed scorer, if any.
    pub fn scorer(&self) -> Option<&Arc<dyn Scorer<S, E>>> {
        self.scorer.as_ref()
    }

    /// Whether missing prompts are treated as silence.
    pub fn missing_prompt_silent(&self) -> bool {
        self.missing_prompt_silent
    }

    /// Pure deterministic preparation — no I/O, no LLM.
    ///
    /// Steps:
    /// 1. `PromptKey::key_for(state, event)` — canonical key
    /// 2. `policy.allowed(state)` — tool allowlist snapshot (`ToolPlan`)
    /// 3. `registry.get(key)` — prompt text. On `RegistryNotFound`, if
    ///    `missing_prompt_silent` is set: empty prompt + forced
    ///    `Silent @ 1.0` score (step 5 is overridden); otherwise the
    ///    error propagates as before.
    /// 4. `builder.build(event, history)` — context string
    /// 5. `scorer.score(state, event, tool_plan)` — if a scorer is
    ///    installed, its verdict rides in `Prepared.score`.
    ///
    /// Calling `prepare` twice with identical `(state, event, history)` and
    /// unchanged registry / builder yields identical output (replay property).
    /// The replay property covers the score too when the scorer is
    /// deterministic (the built-in and mock scorers are).
    pub fn prepare(&self, state: &S, event: &E, history: &[E]) -> Result<Prepared, FormeError> {
        let key = PromptKey::key_for(state, event);
        let allowed = self.policy.allowed(state).to_vec();
        let tool_plan = ToolPlan::new(allowed);
        let (prompt, prompt_missing) = match self.registry.get(&key) {
            Ok(prompt) => (prompt, false),
            Err(e) => {
                if self.missing_prompt_silent && e.is_not_found() {
                    (String::new(), true)
                } else {
                    return Err(e);
                }
            }
        };
        let context = self.builder.build(event, history)?;
        let mut score = self
            .scorer
            .as_ref()
            .map(|scorer| scorer.score(state, event, &tool_plan));
        if prompt_missing {
            score = Some(Score::silent());
        }
        Ok(Prepared {
            key,
            prompt,
            context,
            tool_plan,
            score,
        })
    }

    /// Full step including LLM call.
    ///
    /// `history` is a caller-chosen sliding window (e.g. last 10 events).
    /// Determinism guarantee: `prompt`, `context`, and `tool_plan` are
    /// deterministic given same inputs; `llm_response` is deterministic only
    /// when the underlying `LlmAdapter` is deterministic (e.g. `MockLlm`).
    ///
    /// If the prepared score's verdict is `Silent`, the LLM is **not**
    /// called and `llm_response` is empty — silence includes not paying
    /// for (or waiting on) a completion nobody will use.
    pub async fn step(
        &self,
        state: &S,
        event: &E,
        history: &[E],
    ) -> Result<StepOutput<S>, FormeError> {
        let prepared = self.prepare(state, event, history)?;
        let score = prepared.score;
        let llm_response = if matches!(score, Some(s) if s.verdict == Verdict::Silent) {
            String::new()
        } else {
            self.llm
                .call(prepared.prompt.clone(), prepared.context.clone())
                .await
                .map_err(|e| match e {
                    FormeError::LlmFailed(msg) => FormeError::LlmFailed(msg),
                    other => FormeError::LlmFailed(other.to_string()),
                })?
        };

        Ok(StepOutput {
            next_state: state.clone(),
            prompt_key: prepared.key,
            prompt: prepared.prompt,
            context: prepared.context,
            llm_response,
            tool_plan: prepared.tool_plan,
            score,
        })
    }

    /// Map a graph-flow edge to the next `PromptKey`.
    ///
    /// Minimal policy:
    /// * `Next` / `Continue` → `current_state × event`
    /// * `Branch(new_state)` / `Transition(new_state)` → `new_state × event`
    /// * `Halt` → `current_state × event` (still a valid key for logging)
    ///
    /// No real `graph-flow` dependency — feature-gated later.
    pub fn handle_edge(&self, action: &NextAction<S>, current_state: &S, event: &E) -> PromptKey {
        match action {
            NextAction::Next | NextAction::Continue | NextAction::Halt => {
                PromptKey::key_for(current_state, event)
            }
            NextAction::Branch(ns) | NextAction::Transition(ns) => PromptKey::key_for(ns, event),
        }
    }
}

impl<S, E, R, B, P, L> fmt::Debug for Runner<S, E, R, B, P, L>
where
    S: State + fmt::Debug,
    E: Event + fmt::Debug,
    R: Registry + fmt::Debug,
    B: ContextBuilder<E> + fmt::Debug,
    P: Policy<S> + fmt::Debug,
    L: LlmAdapter<E> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Runner")
            .field("registry", &self.registry)
            .field("builder", &self.builder)
            .field("policy", &self.policy)
            .field("llm", &self.llm)
            .field("has_scorer", &self.scorer.is_some())
            .field("missing_prompt_silent", &self.missing_prompt_silent)
            .finish()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_builder::NoopBuilder;
    use crate::core::{FormeError, PromptKey, ToolId};
    use crate::policy::{AllowAllPolicy, AllowListPolicy, DenyListPolicy, Policy as PolicyTrait};
    use crate::prompt_registry::InMemoryRegistry;
    use crate::rig_adapter::{LlmAdapter as LlmAdapterTrait, MockLlm};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::fmt;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    // ── helpers ───────────────────────────────────────────────────────────

    #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct TestState(String);
    impl fmt::Display for TestState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl TestState {
        fn new(s: &str) -> Self {
            Self(s.into())
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct TestEvent(String);
    impl fmt::Display for TestEvent {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl TestEvent {
        fn new(s: &str) -> Self {
            Self(s.into())
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

    fn assert_send_sync<T: Send + Sync>() {}

    fn make_registry() -> InMemoryRegistry {
        let mut map = HashMap::new();
        map.insert(
            PromptKey::new("Idle", "UserMsg"),
            "you are idle, respond to user".into(),
        );
        map.insert(PromptKey::new("Idle", "Save"), "idle save prompt".into());
        InMemoryRegistry::new(map)
    }

    fn make_runner_with_response(
        resp: &str,
    ) -> Runner<TestState, TestEvent, InMemoryRegistry, NoopBuilder, AllowAllPolicy, MockLlm> {
        let registry = Arc::new(make_registry());
        let builder = Arc::new(NoopBuilder);
        let policy = Arc::new(AllowAllPolicy);
        let llm = MockLlm::with_response(resp);
        Runner::<TestState, TestEvent, _, _, _, _>::new(registry, builder, policy, llm)
    }

    fn unique_temp_dir(suffix: &str) -> PathBuf {
        let base = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pid = std::process::id();
        let name = format!("caulk_runtime_{}_{}_{}", pid, nanos, suffix);
        let path = base.join(name);
        let _ = fs::create_dir_all(&path);
        path
    }

    // 1. happy canned response (InMemory + Noop + AllowAll + Mock)
    #[test]
    fn step_happy_canned_returns_mock() {
        let runner = make_runner_with_response("canned answer");
        let state = TestState::new("Idle");
        let event = TestEvent::new("UserMsg");
        let out = block_on(runner.step(&state, &event, &[])).expect("step ok");

        assert_eq!(out.prompt, "you are idle, respond to user");
        assert_eq!(out.context, ""); // NoopBuilder
        assert_eq!(out.llm_response, "canned answer");
        assert_eq!(out.next_state, state);
        assert_eq!(out.prompt_key, PromptKey::new("Idle", "UserMsg"));
    }

    // 2. FsRegistry integration — prompt from filesystem
    #[test]
    fn step_fs_registry_happy() {
        use crate::prompt_registry::FsRegistry;

        let root = unique_temp_dir("fsreg");
        let state_dir = root.join("Idle");
        fs::create_dir_all(&state_dir).unwrap();
        fs::write(state_dir.join("UserMsg.md"), "fs prompt here").unwrap();

        let registry = Arc::new(FsRegistry::new(&root));
        let builder = Arc::new(NoopBuilder);
        let policy = Arc::new(AllowAllPolicy);
        let llm = MockLlm::with_response("from fs");

        let runner =
            Runner::<TestState, TestEvent, _, _, _, _>::new(registry, builder, policy, llm);
        let state = TestState::new("Idle");
        let event = TestEvent::new("UserMsg");

        // prepare only
        let prepared = runner.prepare(&state, &event, &[]).expect("prepare");
        assert_eq!(prepared.prompt, "fs prompt here");

        let out = block_on(runner.step(&state, &event, &[])).unwrap();
        assert_eq!(out.prompt, "fs prompt here");
        assert_eq!(out.llm_response, "from fs");

        let _ = fs::remove_dir_all(&root);
    }

    // 3. policy allowlist appears in tool plan
    #[test]
    fn tool_plan_respects_policy() {
        let registry = Arc::new(make_registry());
        let builder = Arc::new(NoopBuilder);
        let allowed_tools = vec![ToolId::from("read"), ToolId::from("write")];
        let policy = Arc::new(AllowListPolicy::new(allowed_tools.clone()));
        let llm = MockLlm::with_response("ok");

        let runner: Runner<TestState, TestEvent, _, _, _, _> =
            Runner::<TestState, TestEvent, _, _, _, _>::new(registry, builder, policy, llm);
        let state = TestState::new("Idle");
        let event = TestEvent::new("UserMsg");
        let out = block_on(runner.step(&state, &event, &[])).unwrap();
        assert_eq!(out.tool_plan.len(), 2);
        assert!(out.tool_plan.contains(&ToolId::from("read")));
        assert!(!out.tool_plan.contains(&ToolId::from("exec")));
    }

    // 4. deny list does not make step fail — tool_plan empty means all for that policy
    #[test]
    fn deny_list_step_still_succeeds() {
        let registry = Arc::new(make_registry());
        let builder = Arc::new(NoopBuilder);
        let deny = DenyListPolicy::from_iter(vec![ToolId::from("bad_tool")]);
        let policy = Arc::new(deny);
        let llm = MockLlm::with_response("ok");

        let runner: Runner<TestState, TestEvent, _, _, _, _> =
            Runner::<TestState, TestEvent, _, _, _, _>::new(registry, builder, policy, llm);
        let state = TestState::new("Idle");
        let event = TestEvent::new("UserMsg");

        // Runner does not auto-deny step based on tool presence; it snapshots allowed slice.
        // For DenyListPolicy allowed() returns empty meaning all, check() would deny bad_tool.
        let out = block_on(runner.step(&state, &event, &[])).unwrap();
        assert_eq!(out.tool_plan.allowed.len(), 0); // empty Meaning all

        // Direct policy check still fails for bad_tool
        let policy_ref: &DenyListPolicy = runner.policy();
        assert!(policy_ref.check(&state, &ToolId::from("bad_tool")).is_err());
        assert!(policy_ref.check(&state, &ToolId::from("good_tool")).is_ok());
    }

    // 5. replay determinism — same state/event/history => same prompt/context/tool_plan
    #[test]
    fn replay_determinism() {
        let runner = make_runner_with_response("same");
        let state = TestState::new("Idle");
        let event = TestEvent::new("UserMsg");
        let history = vec![TestEvent::new("h1"), TestEvent::new("h2")];

        let p1 = runner.prepare(&state, &event, &history).unwrap();
        let p2 = runner.prepare(&state, &event, &history).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p1.prompt, p2.prompt);
        assert_eq!(p1.context, p2.context);
        assert_eq!(p1.tool_plan, p2.tool_plan);
        assert_eq!(p1.key, p2.key);

        // step also deterministic when LLM deterministic
        let o1 = block_on(runner.step(&state, &event, &history)).unwrap();
        let o2 = block_on(runner.step(&state, &event, &history)).unwrap();
        assert_eq!(o1.prompt, o2.prompt);
        assert_eq!(o1.context, o2.context);
        assert_eq!(o1.llm_response, o2.llm_response);
    }

    // 6. handle_edge mapping
    #[test]
    fn handle_edge_next_and_branch() {
        let runner = make_runner_with_response("x");
        let cur = TestState::new("Idle");
        let event = TestEvent::new("UserMsg");

        let next_key = runner.handle_edge(&NextAction::Next, &cur, &event);
        assert_eq!(next_key, PromptKey::new("Idle", "UserMsg"));

        let cont_key = runner.handle_edge(&NextAction::Continue, &cur, &event);
        assert_eq!(cont_key, PromptKey::new("Idle", "UserMsg"));

        let new_state = TestState::new("Active");
        let branch_key = runner.handle_edge(&NextAction::Branch(new_state.clone()), &cur, &event);
        assert_eq!(branch_key, PromptKey::new("Active", "UserMsg"));

        let trans_key = runner.handle_edge(&NextAction::Transition(new_state), &cur, &event);
        assert_eq!(trans_key, PromptKey::new("Active", "UserMsg"));

        let halt_key = runner.handle_edge(&NextAction::Halt, &cur, &event);
        assert_eq!(halt_key, PromptKey::new("Idle", "UserMsg"));
        assert!(NextAction::<TestState>::Halt.is_halt());
        assert!(!NextAction::Next::<TestState>.is_halt());
    }

    // 7. error registry not found propagates
    #[test]
    fn error_registry_not_found() {
        let runner = make_runner_with_response("nope");
        let state = TestState::new("MissingState");
        let event = TestEvent::new("MissingEvent");
        let err = runner.prepare(&state, &event, &[]).unwrap_err();
        assert!(err.is_not_found());
        assert!(matches!(err, FormeError::RegistryNotFound(_)));

        let step_err = block_on(runner.step(&state, &event, &[])).unwrap_err();
        assert!(step_err.is_not_found());
    }

    // 8. llm failed maps to FormeError::LlmFailed
    #[test]
    fn llm_failed_maps() {
        let registry = Arc::new(make_registry());
        let builder = Arc::new(NoopBuilder);
        let policy = Arc::new(AllowAllPolicy);
        let llm = MockLlm::with_error("api 500");
        let runner: Runner<TestState, TestEvent, _, _, _, _> =
            Runner::<TestState, TestEvent, _, _, _, _>::new(registry, builder, policy, llm);
        let state = TestState::new("Idle");
        let event = TestEvent::new("UserMsg");
        let err = block_on(runner.step(&state, &event, &[])).unwrap_err();
        match err {
            FormeError::LlmFailed(msg) => assert!(msg.contains("api 500")),
            _ => panic!("expected LlmFailed got {:?}", err),
        }

        // closure returning other FormeError variant should be normalized to LlmFailed
        let llm2 = MockLlm::with_fn(|_, _| Err(FormeError::ContextBuildFailed("inner".into())));
        let runner2: Runner<TestState, TestEvent, _, _, _, _> =
            Runner::<TestState, TestEvent, _, _, _, _>::new(
                Arc::new(make_registry()),
                Arc::new(NoopBuilder),
                Arc::new(AllowAllPolicy),
                llm2,
            );
        let err2 = block_on(runner2.step(&state, &event, &[])).unwrap_err();
        assert!(matches!(err2, FormeError::LlmFailed(_)));
        assert!(err2.to_string().contains("inner"));
    }

    // 9. Send+Sync
    #[test]
    fn runner_is_send_sync() {
        assert_send_sync::<
            Runner<TestState, TestEvent, InMemoryRegistry, NoopBuilder, AllowAllPolicy, MockLlm>,
        >();
        fn assert_runner_bounds<S, E, R, B, P, L>()
        where
            S: State + Send + Sync,
            E: Event + Send + Sync,
            R: Registry + Send + Sync,
            B: ContextBuilder<E> + Send + Sync,
            P: Policy<S> + Send + Sync,
            L: LlmAdapterTrait<E> + Send + Sync,
            Runner<S, E, R, B, P, L>: Send + Sync,
        {
        }
        assert_runner_bounds::<
            TestState,
            TestEvent,
            InMemoryRegistry,
            NoopBuilder,
            AllowAllPolicy,
            MockLlm,
        >();
    }

    // 10. context builder failure propagates
    #[test]
    fn context_build_failure_propagates() {
        use crate::context_builder::ContextBuilder as CbTrait;
        #[derive(Clone, Debug)]
        struct FailBuilder;
        impl<E: crate::core::Event> CbTrait<E> for FailBuilder {
            fn build(&self, _event: &E, _history: &[E]) -> Result<String, FormeError> {
                Err(FormeError::ContextBuildFailed("boom".into()))
            }
        }

        let runner: Runner<TestState, TestEvent, _, FailBuilder, _, _> =
            Runner::<TestState, TestEvent, _, _, _, _>::new(
                Arc::new(make_registry()),
                Arc::new(FailBuilder),
                Arc::new(AllowAllPolicy),
                MockLlm::with_response("x"),
            );
        let err = runner
            .prepare(&TestState::new("Idle"), &TestEvent::new("UserMsg"), &[])
            .unwrap_err();
        assert!(err.to_string().contains("boom"));
    }

    // ── Wave A: scorer + prompt-miss-as-silence ─────────────────────────

    use crate::scorer::{
        AlwaysEscalateScorer, AlwaysSilentScorer, MockScorer, Score, Scorer as ScorerTrait, Verdict,
    };

    fn escalate_scorer() -> Arc<dyn ScorerTrait<TestState, TestEvent>> {
        Arc::new(AlwaysEscalateScorer)
    }

    // 12a. scorer consulted; score present in Prepared and StepOutput;
    //      non-Silent verdicts still call the LLM
    #[test]
    fn scorer_escalate_score_present_and_llm_called() {
        let runner = make_runner_with_response("canned answer").with_scorer(escalate_scorer());
        let state = TestState::new("Idle");
        let event = TestEvent::new("UserMsg");

        let prepared = runner.prepare(&state, &event, &[]).unwrap();
        assert_eq!(prepared.score, Some(Score::new(Verdict::Escalate, 1.0)));

        let out = block_on(runner.step(&state, &event, &[])).unwrap();
        assert_eq!(out.score, Some(Score::new(Verdict::Escalate, 1.0)));
        assert_eq!(out.llm_response, "canned answer");

        // replay determinism covers the score
        let p2 = runner.prepare(&state, &event, &[]).unwrap();
        assert_eq!(prepared, p2);
    }

    // 12b. Silent verdict skips the LLM — an erroring LLM must not fail the step
    #[test]
    fn scorer_silent_skips_llm() {
        let runner: Runner<TestState, TestEvent, _, _, _, _> =
            Runner::<TestState, TestEvent, _, _, _, _>::new(
                Arc::new(make_registry()),
                Arc::new(NoopBuilder),
                Arc::new(AllowAllPolicy),
                MockLlm::with_error("must not be called"),
            )
            .with_scorer(Arc::new(AlwaysSilentScorer));
        let state = TestState::new("Idle");
        let event = TestEvent::new("UserMsg");

        let out = block_on(runner.step(&state, &event, &[])).expect("silent step ok");
        assert_eq!(out.llm_response, "");
        assert_eq!(out.score, Some(Score::silent()));
        assert_eq!(out.prompt, "you are idle, respond to user");
    }

    // 12c. scorer sees the ToolPlan (MockScorer::with_fn through the runner)
    #[test]
    fn scorer_receives_tool_plan() {
        let scorer = MockScorer::<TestState, TestEvent>::with_fn(|_, _, plan| {
            if plan.contains(&ToolId::from("write")) {
                Score::new(Verdict::Act, 0.9)
            } else {
                Score::new(Verdict::Suggest, 0.5)
            }
        });
        let policy = Arc::new(AllowListPolicy::new(vec![
            ToolId::from("read"),
            ToolId::from("write"),
        ]));
        let runner: Runner<TestState, TestEvent, _, _, _, _> =
            Runner::<TestState, TestEvent, _, _, _, _>::new(
                Arc::new(make_registry()),
                Arc::new(NoopBuilder),
                policy,
                MockLlm::with_response("ok"),
            )
            .with_scorer(Arc::new(scorer));
        let prepared = runner
            .prepare(&TestState::new("Idle"), &TestEvent::new("UserMsg"), &[])
            .unwrap();
        assert_eq!(prepared.score, Some(Score::new(Verdict::Act, 0.9)));
    }

    // 13a. missing prompt + silent flag on → prepare/step succeed silently;
    //      forced Silent overrides even an escalate-everything scorer
    #[test]
    fn missing_prompt_silent_flag_on() {
        let runner: Runner<TestState, TestEvent, _, _, _, _> =
            Runner::<TestState, TestEvent, _, _, _, _>::new(
                Arc::new(make_registry()),
                Arc::new(NoopBuilder),
                Arc::new(AllowAllPolicy),
                MockLlm::with_error("must not be called"),
            )
            .with_scorer(escalate_scorer())
            .with_missing_prompt_silent(true);
        let state = TestState::new("MissingState");
        let event = TestEvent::new("MissingEvent");

        let prepared = runner.prepare(&state, &event, &[]).expect("prepare silent");
        assert_eq!(prepared.prompt, "");
        assert_eq!(prepared.score, Some(Score::silent()));
        assert_eq!(prepared.key, PromptKey::new("MissingState", "MissingEvent"));

        let out = block_on(runner.step(&state, &event, &[])).expect("step silent");
        assert_eq!(out.llm_response, "");
        assert_eq!(out.score, Some(Score::silent()));
    }

    // 13b. missing prompt + silent flag on, no scorer → agent still gets a
    //      score-free prompt absence turned into a Silent score
    #[test]
    fn missing_prompt_silent_without_scorer() {
        let runner = make_runner_with_response("unused").with_missing_prompt_silent(true);
        let prepared = runner
            .prepare(
                &TestState::new("MissingState"),
                &TestEvent::new("MissingEvent"),
                &[],
            )
            .unwrap();
        assert_eq!(prepared.score, Some(Score::silent()));
    }

    // 13c. missing prompt + silent flag off → RegistryNotFound as before,
    //      even with a scorer installed
    #[test]
    fn missing_prompt_silent_flag_off_still_errors() {
        let runner = make_runner_with_response("unused").with_scorer(escalate_scorer());
        assert!(!runner.missing_prompt_silent());
        let err = runner
            .prepare(
                &TestState::new("MissingState"),
                &TestEvent::new("MissingEvent"),
                &[],
            )
            .unwrap_err();
        assert!(err.is_not_found());
    }

    // 14. no scorer configured → score is None and behaviour is unchanged
    #[test]
    fn no_scorer_score_none() {
        let runner = make_runner_with_response("canned answer");
        assert!(runner.scorer().is_none());
        let state = TestState::new("Idle");
        let event = TestEvent::new("UserMsg");
        let prepared = runner.prepare(&state, &event, &[]).unwrap();
        assert_eq!(prepared.score, None);
        let out = block_on(runner.step(&state, &event, &[])).unwrap();
        assert_eq!(out.score, None);
        assert_eq!(out.llm_response, "canned answer");
    }

    // 11. ToolPlan helpers
    #[test]
    fn tool_plan_helpers() {
        let plan = ToolPlan::new(vec![ToolId::from("a"), ToolId::from("b")]);
        assert_eq!(plan.len(), 2);
        assert!(!plan.is_empty());
        assert!(plan.contains(&ToolId::from("a")));
        assert!(!plan.contains(&ToolId::from("c")));
        let collected: Vec<_> = plan.clone().into_iter().collect();
        assert_eq!(collected.len(), 2);
        let refs: Vec<_> = (&plan).into_iter().cloned().collect();
        assert_eq!(refs, collected);
    }
}
