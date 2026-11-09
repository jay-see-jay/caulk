//! session driver — owns one session: current state, transition table, runner, checkpointer, halted flag, history. `dispatch` resolves the transition before any LLM call, steps the current state, persists the next state, and returns the outcome. The transition never depends on LLM output.

use std::fmt;

use crate::context_builder::ContextBuilder;
use crate::core::{Event, FormeError, PromptKey, State};
use crate::persistence::Checkpointer;
use crate::policy::Policy;
use crate::prompt_registry::Registry;
use crate::rig_adapter::LlmAdapter;
use crate::runtime::{NextAction, Runner, StepOutput};
use crate::scorer::Score;
use crate::transition::{transition, Action, Transition, TransitionTable};

/// Outcome of one successful `dispatch`.
///
/// `llm_response` is `Some` exactly when `Runner::step` ran (action `Llm`);
/// it is `Some("")` when step ran but the LLM was skipped inside `step`
/// (Silent scorer verdict, or prompt miss with no scorer). It is `None`
/// for `Silent` and `Halt`, where `step` never ran.
/// `next_action` is `None` for `Silent` — it has no `NextAction`
/// equivalent — and `Some(..)` otherwise.
/// `prompt_missing` is copied verbatim from `StepOutput` for `Llm`;
/// it is `false` for `Silent`/`Halt`, where no prompt lookup happened.
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchOutput<S: State> {
    pub next_state: S,
    pub action: Action,
    pub next_action: Option<NextAction<S>>,
    pub llm_response: Option<String>,
    pub prompt_missing: bool,
    pub prompt_key: PromptKey,
    pub score: Option<Score>,
}

/// Owns one session: current state, transition table, runner, checkpointer,
/// halted flag, history.
///
/// `dispatch` resolves the transition before any LLM call, steps the
/// current state through `Runner::step` only when the action says so,
/// persists the next state through the checkpointer, and returns the
/// outcome. The transition never depends on LLM output.
pub struct SessionDriver<S, E, R, B, P, L, C>
where
    S: State,
    E: Event,
    R: Registry,
    B: ContextBuilder<E>,
    P: Policy<S>,
    L: LlmAdapter<E>,
    C: Checkpointer<S>,
{
    runner: Runner<S, E, R, B, P, L>,
    table: TransitionTable<S>,
    checkpointer: C,
    current_state: S,
    halted: bool,
    history: Vec<E>,
}

impl<S, E, R, B, P, L, C> fmt::Debug for SessionDriver<S, E, R, B, P, L, C>
where
    S: State + fmt::Debug,
    E: Event,
    R: Registry,
    B: ContextBuilder<E>,
    P: Policy<S>,
    L: LlmAdapter<E>,
    C: Checkpointer<S>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionDriver")
            .field("current_state", &self.current_state)
            .field("halted", &self.halted)
            .field("history_len", &self.history.len())
            .field("table_len", &self.table.len())
            .field("runner", &std::any::type_name::<Runner<S, E, R, B, P, L>>())
            .field("checkpointer", &std::any::type_name::<C>())
            .finish()
    }
}

impl<S, E, R, B, P, L, C> SessionDriver<S, E, R, B, P, L, C>
where
    S: State,
    E: Event,
    R: Registry,
    B: ContextBuilder<E>,
    P: Policy<S>,
    L: LlmAdapter<E>,
    C: Checkpointer<S>,
{
    /// Fresh driver: `current_state` is `initial_state`, not halted,
    /// empty history. Does not touch the checkpointer — a fresh start
    /// writes nothing until the first successful dispatch.
    pub fn new(
        runner: Runner<S, E, R, B, P, L>,
        table: TransitionTable<S>,
        checkpointer: C,
        initial_state: S,
    ) -> Self {
        Self {
            runner,
            table,
            checkpointer,
            current_state: initial_state,
            halted: false,
            history: Vec::new(),
        }
    }

    /// Resume from the checkpointer: loads the last saved state and
    /// starts there, not halted, with empty history (history is
    /// in-memory only and is never persisted). Errors with
    /// `FormeError::InvalidInput("no checkpoint to resume from")` when
    /// `load()` returns `None`; a load failure propagates as
    /// `FormeError::PersistenceFailed`.
    pub fn resume(
        runner: Runner<S, E, R, B, P, L>,
        table: TransitionTable<S>,
        checkpointer: C,
    ) -> Result<Self, FormeError> {
        match checkpointer.load()? {
            Some(state) => Ok(Self::new(runner, table, checkpointer, state)),
            None => Err(FormeError::InvalidInput(
                "no checkpoint to resume from".into(),
            )),
        }
    }

    /// Dispatch one event (caulk-001c §3.1): halt guard, resolve the
    /// transition before any LLM call, step the current state only for
    /// `Llm`, persist the next state, then commit in memory. A save failure
    /// leaves in-memory state, halted flag, and history untouched; the
    /// LLM call (if one happened) cannot be un-called.
    pub async fn dispatch(&mut self, event: &E) -> Result<DispatchOutput<S>, FormeError> {
        self.ensure_not_halted()?;
        let current = self.current_state.clone();
        let trans = transition(&self.table, &current, event)?;
        let output = self.build_output(&current, event, &trans).await?;
        let halt = trans.action == Action::Halt;
        self.commit(event, output.next_state.clone(), halt)?;
        Ok(output)
    }

    /// The state the next dispatch steps from.
    pub fn current_state(&self) -> &S {
        &self.current_state
    }

    /// True once a `Halt` transition has committed; stays true.
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Successfully dispatched events, oldest first.
    pub fn history(&self) -> &[E] {
        &self.history
    }

    /// Borrowed runner (e.g. for `driver.runner().policy().check(..)`).
    pub fn runner(&self) -> &Runner<S, E, R, B, P, L> {
        &self.runner
    }

    /// Borrowed transition table.
    pub fn table(&self) -> &TransitionTable<S> {
        &self.table
    }

    /// Borrowed checkpointer.
    pub fn checkpointer(&self) -> &C {
        &self.checkpointer
    }

    async fn build_output(
        &self,
        current: &S,
        event: &E,
        trans: &Transition<S>,
    ) -> Result<DispatchOutput<S>, FormeError> {
        match trans.action {
            Action::Llm => Ok(Self::llm_output(
                current,
                trans,
                self.run_llm(current, event).await?,
            )),
            _ => Ok(Self::skip_output_for(current, event, trans)),
        }
    }

    /// Output for the actions that never call `step`: `Silent` has no
    /// `NextAction` equivalent, `Halt` maps to `NextAction::Halt`. The
    /// prompt key is the transition key, present for logging — no prompt
    /// lookup is performed.
    fn skip_output_for(current: &S, event: &E, trans: &Transition<S>) -> DispatchOutput<S> {
        let next_action = match trans.action {
            Action::Halt => Some(NextAction::Halt),
            _ => None,
        };
        Self::skip_output(trans, next_action, PromptKey::key_for(current, event))
    }

    fn llm_output(current: &S, trans: &Transition<S>, step: StepOutput<S>) -> DispatchOutput<S> {
        DispatchOutput {
            next_state: trans.next_state.clone(),
            action: Action::Llm,
            next_action: Self::map_next_action(current, trans),
            llm_response: Some(step.llm_response),
            prompt_missing: step.prompt_missing,
            prompt_key: step.prompt_key,
            score: step.score,
        }
    }

    fn skip_output(
        trans: &Transition<S>,
        next_action: Option<NextAction<S>>,
        key: PromptKey,
    ) -> DispatchOutput<S> {
        DispatchOutput {
            next_state: trans.next_state.clone(),
            action: trans.action,
            next_action,
            llm_response: None,
            prompt_missing: false,
            prompt_key: key,
            score: None,
        }
    }

    fn ensure_not_halted(&self) -> Result<(), FormeError> {
        if self.halted {
            return Err(FormeError::SessionHalted(self.current_state.to_string()));
        }
        Ok(())
    }

    fn map_next_action(current: &S, trans: &Transition<S>) -> Option<NextAction<S>> {
        match trans.action {
            Action::Llm if trans.next_state == *current => Some(NextAction::Next),
            Action::Llm => Some(NextAction::Transition(trans.next_state.clone())),
            Action::Halt => Some(NextAction::Halt),
            Action::Silent => None,
        }
    }

    async fn run_llm(&self, current: &S, event: &E) -> Result<StepOutput<S>, FormeError> {
        self.runner.step(current, event, &self.history).await
    }

    fn commit(&mut self, event: &E, next_state: S, halt: bool) -> Result<(), FormeError> {
        self.checkpointer.save(&next_state)?;
        self.current_state = next_state;
        self.halted = halt;
        self.history.push(event.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_builder::NoopBuilder;
    use crate::core::{FormeError, PromptKey};
    use crate::persistence::{Checkpointer, InMemoryCheckpointer};
    use crate::policy::AllowAllPolicy;
    use crate::prompt_registry::InMemoryRegistry;
    use crate::rig_adapter::MockLlm;
    use crate::scorer::{AlwaysEscalateScorer, Score, Verdict};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    enum BotState {
        Idle,
        Done,
        Greeting,
        CollectingInfo,
        Resolving,
    }
    impl fmt::Display for BotState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Idle => write!(f, "Idle"),
                Self::Done => write!(f, "Done"),
                Self::Greeting => write!(f, "Greeting"),
                Self::CollectingInfo => write!(f, "CollectingInfo"),
                Self::Resolving => write!(f, "Resolving"),
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    enum BotEvent {
        UserMsg,
        Tick,
        UserAskedRefund,
        InfoProvided(String),
        RefundRequested,
        MissingEvent,
    }
    impl fmt::Display for BotEvent {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::UserMsg => write!(f, "UserMsg"),
                Self::Tick => write!(f, "Tick"),
                Self::UserAskedRefund => write!(f, "UserAskedRefund"),
                Self::InfoProvided(_) => write!(f, "InfoProvided"),
                Self::RefundRequested => write!(f, "RefundRequested"),
                Self::MissingEvent => write!(f, "MissingEvent"),
            }
        }
    }

    type Parts = (Arc<InMemoryRegistry>, Arc<NoopBuilder>, Arc<AllowAllPolicy>);
    type BotRunner =
        Runner<BotState, BotEvent, InMemoryRegistry, NoopBuilder, AllowAllPolicy, MockLlm>;
    type BotDriver = SessionDriver<
        BotState,
        BotEvent,
        InMemoryRegistry,
        NoopBuilder,
        AllowAllPolicy,
        MockLlm,
        InMemoryCheckpointer<BotState>,
    >;

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

    fn parts() -> Parts {
        let mut map = HashMap::new();
        map.insert(PromptKey::new("Idle", "UserMsg"), "idle prompt".into());
        map.insert(PromptKey::new("Done", "UserMsg"), "done prompt".into());
        map.insert(
            PromptKey::new("Greeting", "UserAskedRefund"),
            "greeting prompt".into(),
        );
        map.insert(
            PromptKey::new("CollectingInfo", "InfoProvided"),
            "collecting prompt".into(),
        );
        map.insert(
            PromptKey::new("Resolving", "RefundRequested"),
            "resolving prompt".into(),
        );
        (
            Arc::new(InMemoryRegistry::new(map)),
            Arc::new(NoopBuilder),
            Arc::new(AllowAllPolicy),
        )
    }

    fn runner_from(parts: &Parts, llm: MockLlm) -> BotRunner {
        Runner::new(
            Arc::clone(&parts.0),
            Arc::clone(&parts.1),
            Arc::clone(&parts.2),
            llm,
        )
    }

    fn support_table() -> TransitionTable<BotState> {
        TransitionTable::new()
            .with(
                &BotState::Greeting,
                &BotEvent::UserAskedRefund,
                Transition::llm(BotState::CollectingInfo),
            )
            .with(
                &BotState::CollectingInfo,
                &BotEvent::InfoProvided("order #1".into()),
                Transition::llm(BotState::Resolving),
            )
            .with(
                &BotState::Resolving,
                &BotEvent::RefundRequested,
                Transition::halt(BotState::Done),
            )
    }

    fn idle_table() -> TransitionTable<BotState> {
        TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::llm(BotState::Done),
        )
    }

    fn make_driver(
        parts: &Parts,
        table: TransitionTable<BotState>,
        llm: MockLlm,
        initial: BotState,
    ) -> (BotDriver, InMemoryCheckpointer<BotState>) {
        let cp = InMemoryCheckpointer::new();
        let shared = cp.clone();
        let driver = BotDriver::new(runner_from(parts, llm), table, cp, initial);
        (driver, shared)
    }

    fn support_log() -> Vec<BotEvent> {
        vec![
            BotEvent::UserAskedRefund,
            BotEvent::InfoProvided("order #1".into()),
            BotEvent::RefundRequested,
        ]
    }

    fn run_support_log() -> (Vec<BotState>, Vec<Option<String>>, Vec<BotEvent>, bool) {
        let parts = parts();
        let (mut driver, _) = make_driver(
            &parts,
            support_table(),
            MockLlm::with_response("canned"),
            BotState::Greeting,
        );
        let mut states = Vec::new();
        let mut replies = Vec::new();
        for event in support_log() {
            let out = block_on(driver.dispatch(&event)).expect("dispatch ok");
            states.push(out.next_state);
            replies.push(out.llm_response);
        }
        (
            states,
            replies,
            driver.history().to_vec(),
            driver.is_halted(),
        )
    }

    struct FailingCheckpointer;

    impl Checkpointer<BotState> for FailingCheckpointer {
        fn save(&self, _state: &BotState) -> Result<(), FormeError> {
            Err(FormeError::PersistenceFailed("disk full".into()))
        }

        fn load(&self) -> Result<Option<BotState>, FormeError> {
            Err(FormeError::PersistenceFailed("read failed".into()))
        }
    }

    // D1. dispatch_llm_moves_state_and_returns_reply
    #[test]
    fn dispatch_llm_moves_state_and_returns_reply() {
        let parts = parts();
        let (mut driver, cp) = make_driver(
            &parts,
            idle_table(),
            MockLlm::with_response("canned"),
            BotState::Idle,
        );
        let out = block_on(driver.dispatch(&BotEvent::UserMsg)).expect("dispatch ok");
        assert_eq!(out.next_state, BotState::Done);
        assert_eq!(out.action, Action::Llm);
        assert_eq!(
            out.next_action,
            Some(NextAction::Transition(BotState::Done))
        );
        assert_eq!(out.llm_response, Some("canned".to_string()));
        assert!(!out.prompt_missing);
        assert_eq!(*driver.current_state(), BotState::Done);
        assert_eq!(cp.load().unwrap(), Some(BotState::Done));
        assert_eq!(driver.history(), &[BotEvent::UserMsg]);
    }

    // D2. dispatch_llm_same_state_maps_to_next
    #[test]
    fn dispatch_llm_same_state_maps_to_next() {
        let parts = parts();
        let table = TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::llm(BotState::Idle),
        );
        let (mut driver, _) = make_driver(
            &parts,
            table,
            MockLlm::with_response("canned"),
            BotState::Idle,
        );
        let out = block_on(driver.dispatch(&BotEvent::UserMsg)).expect("dispatch ok");
        assert_eq!(out.next_state, BotState::Idle);
        // Explicitly Next, not the legacy Continue spelling.
        assert_eq!(out.next_action, Some(NextAction::Next));
        assert_ne!(out.next_action, Some(NextAction::Continue));
    }

    // D3. dispatch_silent_skips_step_entirely
    #[test]
    fn dispatch_silent_skips_step_entirely() {
        let parts = parts();
        let table = TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::Tick,
            Transition::silent(BotState::Idle),
        );
        let (mut driver, cp) = make_driver(
            &parts,
            table,
            MockLlm::with_error("must not be called"),
            BotState::Idle,
        );
        let out = block_on(driver.dispatch(&BotEvent::Tick)).expect("dispatch ok");
        assert_eq!(out.llm_response, None);
        assert!(!out.prompt_missing);
        assert_eq!(out.score, None);
        assert_eq!(*driver.current_state(), BotState::Idle);
        // Silent still persists.
        assert_eq!(cp.load().unwrap(), Some(BotState::Idle));
        assert_eq!(driver.history(), &[BotEvent::Tick]);
    }

    // D4. dispatch_halt_is_terminal_no_llm
    #[test]
    fn dispatch_halt_is_terminal_no_llm() {
        let parts = parts();
        let table = TransitionTable::new().with(
            &BotState::Resolving,
            &BotEvent::RefundRequested,
            Transition::halt(BotState::Done),
        );
        let (mut driver, cp) = make_driver(
            &parts,
            table,
            MockLlm::with_error("must not be called"),
            BotState::Resolving,
        );
        let out = block_on(driver.dispatch(&BotEvent::RefundRequested)).expect("dispatch ok");
        assert_eq!(out.next_state, BotState::Done);
        assert_eq!(out.action, Action::Halt);
        assert_eq!(out.next_action, Some(NextAction::Halt));
        assert_eq!(out.llm_response, None);
        assert!(driver.is_halted());
        assert_eq!(cp.load().unwrap(), Some(BotState::Done));
    }

    // D5. dispatch_after_halt_errors_session_halted
    #[test]
    fn dispatch_after_halt_errors_session_halted() {
        let parts = parts();
        let table = TransitionTable::new().with(
            &BotState::Resolving,
            &BotEvent::RefundRequested,
            Transition::halt(BotState::Done),
        );
        let (mut driver, _) = make_driver(
            &parts,
            table,
            MockLlm::with_error("must not be called"),
            BotState::Resolving,
        );
        block_on(driver.dispatch(&BotEvent::RefundRequested)).expect("halt ok");
        let err = block_on(driver.dispatch(&BotEvent::RefundRequested)).expect_err("halted");
        assert!(err.is_session_halted());
        assert!(err.to_string().contains("Done"), "err was: {err}");
        assert_eq!(*driver.current_state(), BotState::Done);
        assert_eq!(driver.history().len(), 1);
    }

    // D6. transition_miss_leaves_state_unchanged
    #[test]
    fn transition_miss_leaves_state_unchanged() {
        // (a) Populated table, unmapped pair from Done.
        let parts = parts();
        let (mut driver, cp) = make_driver(
            &parts,
            idle_table(),
            MockLlm::with_response("canned"),
            BotState::Done,
        );
        let err = block_on(driver.dispatch(&BotEvent::UserMsg)).expect_err("miss");
        assert!(err.is_transition_miss());
        assert_eq!(*driver.current_state(), BotState::Done);
        assert!(driver.history().is_empty());
        assert_eq!(cp.load().unwrap(), None);

        // (b) Empty table misses everything.
        assert_empty_table_misses(&parts);
    }

    fn assert_empty_table_misses(parts: &Parts) {
        let (mut empty, _) = make_driver(
            parts,
            TransitionTable::new(),
            MockLlm::with_response("canned"),
            BotState::Idle,
        );
        let err = block_on(empty.dispatch(&BotEvent::UserMsg)).expect_err("miss");
        assert!(err.is_transition_miss());
        assert!(!empty.is_halted());
    }

    // D7. deterministic_replay_same_log_same_states
    #[test]
    fn deterministic_replay_same_log_same_states() {
        let first = run_support_log();
        let (states, replies, history, halted) = (&first.0, &first.1, &first.2, first.3);
        assert_eq!(
            states,
            &vec![
                BotState::CollectingInfo,
                BotState::Resolving,
                BotState::Done
            ]
        );
        assert_eq!(
            replies,
            &vec![Some("canned".to_string()), Some("canned".to_string()), None,]
        );
        assert_eq!(history, &support_log());
        assert!(halted);
        // Run the whole log twice more on fresh drivers: byte-identical.
        assert_eq!(run_support_log(), first);
        assert_eq!(run_support_log(), first);
    }

    // D8. persistence_round_trip_resume
    #[test]
    fn persistence_round_trip_resume() {
        let parts = parts();
        let (mut driver, cp) = make_driver(
            &parts,
            support_table(),
            MockLlm::with_response("canned"),
            BotState::Greeting,
        );
        block_on(driver.dispatch(&BotEvent::UserAskedRefund)).expect("dispatch 1");
        block_on(driver.dispatch(&BotEvent::InfoProvided("order #1".into()))).expect("dispatch 2");
        let runner2 = runner_from(&parts, MockLlm::with_response("canned"));
        let mut resumed =
            BotDriver::resume(runner2, support_table(), cp.clone()).expect("resume ok");
        assert_eq!(*resumed.current_state(), BotState::Resolving);
        assert!(resumed.history().is_empty());
        assert!(!resumed.is_halted());
        let out = block_on(resumed.dispatch(&BotEvent::RefundRequested)).expect("dispatch 3");
        assert_eq!(out.next_state, BotState::Done);
        assert!(resumed.is_halted());
    }

    // D9. resume_with_no_checkpoint_errors
    #[test]
    fn resume_with_no_checkpoint_errors() {
        let parts = parts();
        let runner = runner_from(&parts, MockLlm::with_response("canned"));
        let err = BotDriver::resume(runner, support_table(), InMemoryCheckpointer::new())
            .expect_err("no checkpoint");
        assert!(matches!(err, FormeError::InvalidInput(_)));
        assert!(err.to_string().contains("no checkpoint"), "err was: {err}");
    }

    // D10. resume_after_halt_is_not_halted
    #[test]
    fn resume_after_halt_is_not_halted() {
        let parts = parts();
        let table = TransitionTable::new().with(
            &BotState::Resolving,
            &BotEvent::RefundRequested,
            Transition::halt(BotState::Done),
        );
        let (mut driver, cp) = make_driver(
            &parts,
            table,
            MockLlm::with_error("must not be called"),
            BotState::Resolving,
        );
        block_on(driver.dispatch(&BotEvent::RefundRequested)).expect("halt ok");
        let runner2 = runner_from(&parts, MockLlm::with_response("canned"));
        let resumed = BotDriver::resume(runner2, support_table(), cp.clone()).expect("resume ok");
        assert_eq!(*resumed.current_state(), BotState::Done);
        // Halt is not persisted (§3.6): resume returns a fresh, non-halted driver.
        assert!(!resumed.is_halted());
        assert!(resumed.history().is_empty());
    }

    // D11. prompt_miss_flag_surfaces_no_scorer
    #[test]
    fn prompt_miss_flag_surfaces_no_scorer() {
        let parts = parts();
        let runner = runner_from(&parts, MockLlm::with_error("must not be called"))
            .with_missing_prompt_silent(true);
        let table = TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::MissingEvent,
            Transition::llm(BotState::Idle),
        );
        let mut driver = BotDriver::new(runner, table, InMemoryCheckpointer::new(), BotState::Idle);
        let out = block_on(driver.dispatch(&BotEvent::MissingEvent)).expect("dispatch ok");
        assert!(out.prompt_missing);
        assert_eq!(out.llm_response, Some(String::new()));
        assert_eq!(out.score, None);
    }

    // D12. prompt_miss_flag_surfaces_with_escalate_scorer
    #[test]
    fn prompt_miss_flag_surfaces_with_escalate_scorer() {
        let parts = parts();
        let runner = runner_from(&parts, MockLlm::with_response("canned"))
            .with_scorer(Arc::new(AlwaysEscalateScorer))
            .with_missing_prompt_silent(true);
        let table = TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::MissingEvent,
            Transition::llm(BotState::Idle),
        );
        let mut driver = BotDriver::new(runner, table, InMemoryCheckpointer::new(), BotState::Idle);
        let out = block_on(driver.dispatch(&BotEvent::MissingEvent)).expect("dispatch ok");
        assert!(out.prompt_missing);
        assert_eq!(out.llm_response, Some("canned".to_string()));
        assert_eq!(out.score, Some(Score::new(Verdict::Escalate, 1.0)));
    }

    // D13. step_failure_leaves_state_unchanged
    #[test]
    fn step_failure_leaves_state_unchanged() {
        let parts = parts();
        let (mut driver, cp) = make_driver(
            &parts,
            idle_table(),
            MockLlm::with_error("api 500"),
            BotState::Idle,
        );
        let err = block_on(driver.dispatch(&BotEvent::UserMsg)).expect_err("llm fails");
        assert!(matches!(err, FormeError::LlmFailed(_)));
        assert!(err.to_string().contains("api 500"), "err was: {err}");
        assert_eq!(*driver.current_state(), BotState::Idle);
        assert!(driver.history().is_empty());
        assert_eq!(cp.load().unwrap(), None);
    }

    // D14. registry_miss_without_silent_flag_propagates
    #[test]
    fn registry_miss_without_silent_flag_propagates() {
        let parts = parts();
        let table = TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::MissingEvent,
            Transition::llm(BotState::Idle),
        );
        let (mut driver, cp) = make_driver(
            &parts,
            table,
            MockLlm::with_response("canned"),
            BotState::Idle,
        );
        let err = block_on(driver.dispatch(&BotEvent::MissingEvent)).expect_err("miss");
        assert!(err.is_not_found());
        assert_eq!(*driver.current_state(), BotState::Idle);
        assert!(driver.history().is_empty());
        assert_eq!(cp.load().unwrap(), None);
    }

    // D15. history_records_dispatched_events_only
    #[test]
    fn history_records_dispatched_events_only() {
        let parts = parts();
        let (mut driver, _) = make_driver(
            &parts,
            support_table(),
            MockLlm::with_response("canned"),
            BotState::Greeting,
        );
        block_on(driver.dispatch(&BotEvent::UserAskedRefund)).expect("dispatch 1");
        block_on(driver.dispatch(&BotEvent::InfoProvided("order #1".into()))).expect("dispatch 2");
        let err = block_on(driver.dispatch(&BotEvent::Tick)).expect_err("miss");
        assert!(err.is_transition_miss());
        block_on(driver.dispatch(&BotEvent::RefundRequested)).expect("dispatch 3");
        assert_eq!(driver.history(), &support_log());
    }

    // D16. save_failure_leaves_state_unchanged
    #[test]
    fn save_failure_leaves_state_unchanged() {
        let parts = parts();
        let runner = runner_from(&parts, MockLlm::with_response("canned"));
        let mut driver =
            SessionDriver::new(runner, idle_table(), FailingCheckpointer, BotState::Idle);
        let err = block_on(driver.dispatch(&BotEvent::UserMsg)).expect_err("save fails");
        assert!(matches!(err, FormeError::PersistenceFailed(_)));
        assert_eq!(*driver.current_state(), BotState::Idle);
        assert!(!driver.is_halted());
        assert!(driver.history().is_empty());
    }

    fn ping_pong_table() -> TransitionTable<BotState> {
        TransitionTable::new()
            .with(
                &BotState::Idle,
                &BotEvent::UserMsg,
                Transition::llm(BotState::Done),
            )
            .with(
                &BotState::Done,
                &BotEvent::UserMsg,
                Transition::llm(BotState::Idle),
            )
    }

    // D17. double_dispatch_same_event_advances_each_time
    #[test]
    fn double_dispatch_same_event_advances_each_time() {
        let parts = parts();
        let (mut driver, _) = make_driver(
            &parts,
            ping_pong_table(),
            MockLlm::with_response("canned"),
            BotState::Idle,
        );
        let first = block_on(driver.dispatch(&BotEvent::UserMsg)).expect("dispatch 1");
        let second = block_on(driver.dispatch(&BotEvent::UserMsg)).expect("dispatch 2");
        assert_eq!(
            vec![first.next_state, second.next_state],
            vec![BotState::Done, BotState::Idle]
        );
        assert_eq!(driver.history().len(), 2);
    }

    /// The `(state, event)` pair behind a canonical table key. The table
    /// only exposes keys, so the pair is recovered by matching the
    /// canonical spelling.
    fn pair_for_key(canonical: &str) -> (BotState, BotEvent) {
        match canonical {
            "CollectingInfo::InfoProvided" => (
                BotState::CollectingInfo,
                BotEvent::InfoProvided("order #1".into()),
            ),
            "Greeting::UserAskedRefund" => (BotState::Greeting, BotEvent::UserAskedRefund),
            "Resolving::RefundRequested" => (BotState::Resolving, BotEvent::RefundRequested),
            other => panic!("unexpected key {other}"),
        }
    }

    fn assert_table_keys(driver: &BotDriver) {
        let canonical: Vec<String> = driver
            .table()
            .keys()
            .iter()
            .map(|k| k.canonical())
            .collect();
        assert_eq!(
            canonical,
            vec![
                "CollectingInfo::InfoProvided",
                "Greeting::UserAskedRefund",
                "Resolving::RefundRequested",
            ]
        );
    }

    // D18. table_coverage_accepted_pairs_resolve
    #[test]
    fn table_coverage_accepted_pairs_resolve() {
        let parts = parts();
        let (driver, _) = make_driver(
            &parts,
            support_table(),
            MockLlm::with_response("canned"),
            BotState::Greeting,
        );
        assert_table_keys(&driver);
        for key in driver.table().keys() {
            let (state, event) = pair_for_key(&key.canonical());
            assert!(
                transition(driver.table(), &state, &event).is_ok(),
                "{key} should resolve"
            );
        }
        let err = transition(driver.table(), &BotState::Done, &BotEvent::Tick)
            .expect_err("Done x Tick is unmapped");
        assert_eq!(err.key(), &PromptKey::new("Done", "Tick"));
    }

    // D19. driver_is_send_sync
    #[test]
    fn driver_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BotDriver>();
    }
}
