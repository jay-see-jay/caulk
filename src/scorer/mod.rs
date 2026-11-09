//! scorer — judgment gate consulted during `Runner::prepare`
//!
//! Wave A: a `Scorer` is a cheap, non-generative judgment call made
//! *before* the LLM is invoked (the Jev/Clef pattern: score a closed
//! option set, return a verdict + confidence, generate nothing).
//! Its job is the collaboration question: given this state, this event,
//! and the tools this state is permitted (`ToolPlan`), what should the
//! agent do — stay silent, annotate, suggest, propose, act, or escalate
//! to the full LLM?
//!
//! Integration (see `runtime`):
//!
//! * `Runner::with_scorer(...)` installs a scorer. `prepare` computes the
//!   `ToolPlan` first, then scores it; the `Score` rides in `Prepared`
//!   and `StepOutput` so replay determinism covers it.
//! * A `Silent` verdict **skips the LLM call entirely** in `step` —
//!   silence is a scored, first-class outcome, not an error.
//! * All other verdicts behave exactly as an unscored step: the LLM is
//!   called and the product interprets the verdict.
//! * With no scorer installed, `score` is `None` and behaviour is
//!   identical to pre-scorer caulk.
//!
//! Like `Policy`, a scorer is a plain `Send + Sync` trait held behind
//! `Arc`. Deterministic scorers keep `prepare` replayable; a scorer that
//! wraps a learned model gives that up knowingly, same trade as the LLM.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::core::{Event, State};
use crate::runtime::ToolPlan;

/// What the agent should do with this step.
///
/// The option set is deliberately closed — a scorer picks one of these,
/// it does not invent a seventh thing to do. `Display` names are stable
/// (they match the variant names) so verdicts can be logged, keyed, and
/// compared across versions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Verdict {
    /// Do nothing. No LLM call, no output. The correct answer most of
    /// the time in a realtime collaboration setting.
    Silent,
    /// Record a private note / observation; nothing shown to the user.
    Annotate,
    /// Offer a suggestion the user can ignore at zero cost.
    Suggest,
    /// Propose a concrete, undoable action for the user to accept.
    Propose,
    /// Act directly with the tools in the `ToolPlan`.
    Act,
    /// Hand the step to the full LLM (the default, pre-scorer behaviour).
    Escalate,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Verdict::Silent => "Silent",
            Verdict::Annotate => "Annotate",
            Verdict::Suggest => "Suggest",
            Verdict::Propose => "Propose",
            Verdict::Act => "Act",
            Verdict::Escalate => "Escalate",
        };
        write!(f, "{}", name)
    }
}

/// One scored judgment: a verdict and the scorer's confidence in it.
///
/// `confidence` is in `[0.0, 1.0]` by convention (not enforced — scorers
/// are product code). Thresholds live with the caller: e.g. act only
/// above 0.9, escalate when the spread is uncertain. `f32` means
/// `Score` is `PartialEq` but not `Eq`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub verdict: Verdict,
    pub confidence: f32,
}

impl Score {
    pub fn new(verdict: Verdict, confidence: f32) -> Self {
        Self {
            verdict,
            confidence,
        }
    }

    /// Certain silence — used by the built-in silent paths.
    pub fn silent() -> Self {
        Self::new(Verdict::Silent, 1.0)
    }

    /// Certain escalation — the "always ask the LLM" default.
    pub fn escalate() -> Self {
        Self::new(Verdict::Escalate, 1.0)
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{:.2}", self.verdict, self.confidence)
    }
}

/// The judgment gate. Consulted once per step, after the `ToolPlan`
/// snapshot and before any LLM call.
///
/// Implementations must be `Send + Sync` (supertrait) so a scorer can be
/// shared across runners and tasks behind `Arc<dyn Scorer<S, E>>`.
pub trait Scorer<S: State, E: Event>: Send + Sync {
    fn score(&self, state: &S, event: &E, tool_plan: &ToolPlan) -> Score;
}

// ── Built-in scorers ────────────────────────────────────────────────────

/// Always escalate: every step goes to the LLM, exactly like pre-scorer
/// caulk, but with the decision made explicit (and logged) as a `Score`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlwaysEscalateScorer;

impl<S: State, E: Event> Scorer<S, E> for AlwaysEscalateScorer {
    fn score(&self, _state: &S, _event: &E, _tool_plan: &ToolPlan) -> Score {
        Score::escalate()
    }
}

/// Always silent: no step ever reaches the LLM. Useful for observe-only
/// postures and as a kill switch.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlwaysSilentScorer;

impl<S: State, E: Event> Scorer<S, E> for AlwaysSilentScorer {
    fn score(&self, _state: &S, _event: &E, _tool_plan: &ToolPlan) -> Score {
        Score::silent()
    }
}

// ── MockScorer ──────────────────────────────────────────────────────────

/// In-memory mock scorer for tests, mirroring `MockLlm`'s style.
///
/// * `MockScorer::with_verdict(Verdict::Propose)` → always that verdict
///   at confidence 1.0
/// * `MockScorer::with_fn(|state, event, plan| ...)` → custom closure,
///   can inspect state / event / tool plan
///
/// The `MockScorer` handler signature, aliased for readability.
type ScoreFn<S, E> = dyn Fn(&S, &E, &ToolPlan) -> Score + Send + Sync;

/// Unlike `MockLlm` (whose call signature is just two strings),
/// `MockScorer` is generic over `S`/`E` because the closure receives
/// them typed. It is `Clone` (Arc-shared handler).
pub struct MockScorer<S, E> {
    handler: Arc<ScoreFn<S, E>>,
    _marker: PhantomData<fn() -> (S, E)>,
}

impl<S, E> Clone for MockScorer<S, E> {
    fn clone(&self) -> Self {
        Self {
            handler: Arc::clone(&self.handler),
            _marker: PhantomData,
        }
    }
}

impl<S, E> fmt::Debug for MockScorer<S, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MockScorer").finish_non_exhaustive()
    }
}

impl<S, E> MockScorer<S, E> {
    /// Always returns `verdict` at confidence 1.0.
    pub fn with_verdict(verdict: Verdict) -> Self {
        Self::with_fn(move |_, _, _| Score::new(verdict, 1.0))
    }

    /// Always returns this exact `Score`.
    pub fn with_score(score: Score) -> Self {
        Self::with_fn(move |_, _, _| score)
    }

    /// Custom function `(state, event, tool_plan) -> Score`.
    pub fn with_fn<F>(func: F) -> Self
    where
        F: Fn(&S, &E, &ToolPlan) -> Score + Send + Sync + 'static,
    {
        Self {
            handler: Arc::new(func),
            _marker: PhantomData,
        }
    }
}

impl<S: State, E: Event> Scorer<S, E> for MockScorer<S, E> {
    fn score(&self, state: &S, event: &E, tool_plan: &ToolPlan) -> Score {
        (self.handler)(state, event, tool_plan)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ToolId;
    use serde::{Deserialize, Serialize};
    use std::fmt;

    #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct TestState(String);
    impl fmt::Display for TestState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct TestEvent(String);
    impl fmt::Display for TestEvent {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    fn state() -> TestState {
        TestState("Drafting".into())
    }

    fn event() -> TestEvent {
        TestEvent("ParagraphDone".into())
    }

    fn plan() -> ToolPlan {
        ToolPlan::new(vec![ToolId::from("propose_nudge")])
    }

    // 1. Display names are the stable variant names
    #[test]
    fn verdict_display_is_stable() {
        assert_eq!(Verdict::Silent.to_string(), "Silent");
        assert_eq!(Verdict::Annotate.to_string(), "Annotate");
        assert_eq!(Verdict::Suggest.to_string(), "Suggest");
        assert_eq!(Verdict::Propose.to_string(), "Propose");
        assert_eq!(Verdict::Act.to_string(), "Act");
        assert_eq!(Verdict::Escalate.to_string(), "Escalate");
    }

    // 2. Verdict serde roundtrip
    #[test]
    fn verdict_serde_roundtrip() {
        for v in [
            Verdict::Silent,
            Verdict::Annotate,
            Verdict::Suggest,
            Verdict::Propose,
            Verdict::Act,
            Verdict::Escalate,
        ] {
            let json = serde_json::to_string(&v).unwrap();
            let back: Verdict = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // 3. Always* scorers
    #[test]
    fn always_scorers() {
        let esc = AlwaysEscalateScorer.score(&state(), &event(), &plan());
        assert_eq!(esc, Score::new(Verdict::Escalate, 1.0));
        let sil = AlwaysSilentScorer.score(&state(), &event(), &plan());
        assert_eq!(sil, Score::new(Verdict::Silent, 1.0));
    }

    // 4. MockScorer::with_verdict / with_score
    #[test]
    fn mock_with_verdict_and_score() {
        let m = MockScorer::<TestState, TestEvent>::with_verdict(Verdict::Propose);
        assert_eq!(
            m.score(&state(), &event(), &plan()),
            Score::new(Verdict::Propose, 1.0)
        );
        let m2 = MockScorer::<TestState, TestEvent>::with_score(Score::new(Verdict::Act, 0.42));
        assert_eq!(
            m2.score(&state(), &event(), &plan()),
            Score::new(Verdict::Act, 0.42)
        );
    }

    // 5. MockScorer::with_fn sees state, event, and tool plan
    #[test]
    fn mock_with_fn_inspects_inputs() {
        let m = MockScorer::<TestState, TestEvent>::with_fn(|s, e, plan| {
            if s.0 == "Drafting" && e.0 == "ParagraphDone" && !plan.is_empty() {
                Score::new(Verdict::Act, 0.9)
            } else {
                Score::silent()
            }
        });
        assert_eq!(
            m.score(&state(), &event(), &plan()),
            Score::new(Verdict::Act, 0.9)
        );
        assert_eq!(
            m.score(&state(), &event(), &ToolPlan::new(vec![])),
            Score::silent()
        );
    }

    // 6. Scorer is object-safe and usable as Arc<dyn Scorer>
    #[test]
    fn scorer_object_safe() {
        let boxed: Arc<dyn Scorer<TestState, TestEvent>> = Arc::new(AlwaysSilentScorer);
        assert_eq!(
            boxed.score(&state(), &event(), &plan()).verdict,
            Verdict::Silent
        );
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<dyn Scorer<TestState, TestEvent>>>();
        assert_send_sync::<MockScorer<TestState, TestEvent>>();
    }

    // 7. Score Display
    #[test]
    fn score_display() {
        assert_eq!(
            Score::new(Verdict::Suggest, 0.5).to_string(),
            "Suggest@0.50"
        );
    }
}
