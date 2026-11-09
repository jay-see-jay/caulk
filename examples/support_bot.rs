//! support_bot — 3-state refund FSM driven by `SessionDriver`
//!
//! States: Greeting -> CollectingInfo -> Resolving -> Done
//! Events: UserAskedRefund, InfoProvided, RefundRequested
//! Demonstrates:
//! - ParagraphBuilder for prose context
//! - InMemoryRegistry with per-state prompts
//! - Custom per-state Policy (SupportPolicy)
//! - SessionDriver::dispatch (transition resolved before any LLM call)
//! - ToolDenied via FormeError
//! - Dispatch-after-halt errors with FormeError::SessionHalted

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use caulk::context_builder::ParagraphBuilder;
use caulk::core::{FormeError, PromptKey, ToolId};
use caulk::persistence::InMemoryCheckpointer;
use caulk::policy::Policy;
use caulk::prompt_registry::InMemoryRegistry;
use caulk::rig_adapter::MockLlm;
use caulk::runtime::Runner;
use caulk::transition::{Transition, TransitionTable};
use caulk::SessionDriver;
use serde::{Deserialize, Serialize};

// ── State ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum SupportState {
    Greeting,
    CollectingInfo,
    Resolving,
    Done,
}

impl fmt::Display for SupportState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Greeting => write!(f, "Greeting"),
            Self::CollectingInfo => write!(f, "CollectingInfo"),
            Self::Resolving => write!(f, "Resolving"),
            Self::Done => write!(f, "Done"),
        }
    }
}

// ── Event ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
enum SupportEvent {
    UserAskedRefund,
    InfoProvided(String),
    RefundRequested,
}

impl fmt::Display for SupportEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserAskedRefund => write!(f, "UserAskedRefund"),
            Self::InfoProvided(_) => write!(f, "InfoProvided"),
            Self::RefundRequested => write!(f, "RefundRequested"),
        }
    }
}

// ── Policy ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SupportPolicy {
    greeting_allowed: Vec<ToolId>,
    collecting_allowed: Vec<ToolId>,
    resolving_allowed: Vec<ToolId>,
}

impl SupportPolicy {
    fn new() -> Self {
        Self {
            greeting_allowed: vec![ToolId::from("read_profile"), ToolId::from("read_order")],
            collecting_allowed: vec![
                ToolId::from("read_profile"),
                ToolId::from("read_order"),
                ToolId::from("ask_clarification"),
            ],
            resolving_allowed: vec![
                ToolId::from("read_profile"),
                ToolId::from("read_order"),
                ToolId::from("process_refund"),
            ],
        }
    }
}

impl Policy<SupportState> for SupportPolicy {
    fn allowed(&self, state: &SupportState) -> &[ToolId] {
        match state {
            SupportState::Greeting => &self.greeting_allowed,
            SupportState::CollectingInfo => &self.collecting_allowed,
            SupportState::Resolving => &self.resolving_allowed,
            SupportState::Done => &[],
        }
    }
}

// ── tiny block_on (no tokio needed) ───────────────────────────────────────

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
    let mut pinned = Box::pin(fut);
    loop {
        match Future::poll(Pin::new(&mut pinned).as_mut(), &mut cx) {
            Poll::Ready(v) => break v,
            Poll::Pending => std::hint::spin_loop(),
        }
    }
}

fn denied_tool_for(state: &SupportState) -> ToolId {
    match state {
        SupportState::Greeting | SupportState::CollectingInfo => ToolId::from("process_refund"),
        SupportState::Resolving | SupportState::Done => ToolId::from("ask_clarification"),
    }
}

fn main() {
    // ── Registry (exactly 3 entries, one per transition that steps) ─────────
    let mut map: HashMap<PromptKey, String> = HashMap::new();
    map.insert(
        PromptKey::new("Greeting", "UserAskedRefund"),
        "You are a friendly support bot in GREETING. Greet the user warmly, \
         explain you can look up their profile and order (read_profile, read_order). \
         Do NOT issue refunds yet. Be concise and helpful."
            .into(),
    );
    map.insert(
        PromptKey::new("CollectingInfo", "InfoProvided"),
        "You are in COLLECTING_INFO. You have partial user info. \
         Use read_profile and read_order to verify, and ask_clarification for missing \
         order ID or reason. Do not refund yet. Keep tone supportive."
            .into(),
    );
    map.insert(
        PromptKey::new("Resolving", "RefundRequested"),
        "You are in RESOLVING. User explicitly requested refund. \
         Verify eligibility via read_profile/read_order, then use process_refund \
         if allowed. Summarize outcome clearly."
            .into(),
    );

    let registry = Arc::new(InMemoryRegistry::new(map));

    // ── Context builder (ParagraphBuilder prose, N=2) ────────────────────────
    let builder = Arc::new(ParagraphBuilder::with_n(2));

    // ── Policy ───────────────────────────────────────────────────────────────
    let policy = Arc::new(SupportPolicy::new());

    // ── LLM (MockLlm closure returns different response per prompt) ─────────
    let llm = MockLlm::with_fn(|prompt: String, context: String| {
        let lower = prompt.to_lowercase();
        let resp = if lower.contains("greeting") {
            format!(
                "Hello! I can help with refunds. I’ll look up your profile/order. \
                 [seen context: {}]",
                context.chars().take(80).collect::<String>()
            )
        } else if lower.contains("collecting") {
            "Thanks for the info. I’ve checked your profile and order. \
             Could you confirm the order ID and reason? (ask_clarification) \
             [collecting_info]"
                .to_string()
        } else if lower.contains("resolving") {
            "Refund eligible. I have processed your refund of $42.50 via \
             process_refund. You’ll see it in 3-5 days. [resolving]"
                .to_string()
        } else {
            format!(
                "Fallback response for prompt: {}",
                prompt.chars().take(60).collect::<String>()
            )
        };
        Ok(resp)
    });

    // ── Runner + table + driver ─────────────────────────────────────────────
    let runner = Runner::new(
        Arc::clone(&registry),
        Arc::clone(&builder),
        Arc::clone(&policy),
        llm,
    );
    let table = TransitionTable::new()
        .with(
            &SupportState::Greeting,
            &SupportEvent::UserAskedRefund,
            Transition::llm(SupportState::CollectingInfo),
        )
        .with(
            &SupportState::CollectingInfo,
            &SupportEvent::InfoProvided("order #12345, bought yesterday".into()),
            Transition::llm(SupportState::Resolving),
        )
        .with(
            &SupportState::Resolving,
            &SupportEvent::RefundRequested,
            Transition::halt(SupportState::Done),
        );
    let mut driver = SessionDriver::new(
        runner,
        table,
        InMemoryCheckpointer::new(),
        SupportState::Greeting,
    );

    // ── Simulate 3-turn conversation ───────────────────────────────────────
    let events: Vec<SupportEvent> = vec![
        SupportEvent::UserAskedRefund,
        SupportEvent::InfoProvided("order #12345, bought yesterday".into()),
        SupportEvent::RefundRequested,
    ];

    for (idx, event) in events.iter().enumerate() {
        let state = driver.current_state().clone();
        println!("=== Turn {}: State={} Event={} ===", idx + 1, state, event);

        // ---- policy deny demo ----------------------------------------------
        let denied_attempt = denied_tool_for(&state);
        match driver.runner().policy().check(&state, &denied_attempt) {
            Ok(()) => {
                println!(
                    "tool check: {} allowed (expected deny for demo, but allowed)",
                    denied_attempt
                );
            }
            Err(FormeError::PolicyDenied(tid)) => {
                println!(
                    "denied tool attempt: {} -> {} (FormeError::ToolDenied)",
                    tid,
                    FormeError::PolicyDenied(tid.clone())
                );
                let err = FormeError::PolicyDenied(tid.clone());
                println!("  is_denied(): {}", err.is_denied());
            }
            Err(other) => {
                println!("unexpected policy error: {other}");
            }
        }

        // ---- dispatch through the driver ------------------------------------
        let out = block_on(driver.dispatch(event)).expect("dispatch ok");
        println!("next_state: {}", out.next_state);
        println!("action: {}", out.action);
        println!("next_action: {:?}", out.next_action);
        println!(
            "llm_response: {}",
            out.llm_response.as_deref().unwrap_or("(none)")
        );
        println!("prompt_missing: {}", out.prompt_missing);
        println!();
    }

    println!("current_state: {}", driver.current_state());
    println!("is_halted: {}", driver.is_halted());
    println!("history length: {}", driver.history().len());

    // ---- dispatch after halt: must error, must not panic --------------------
    match block_on(driver.dispatch(&SupportEvent::RefundRequested)) {
        Ok(out) => println!("unexpected success after halt: {out:?}"),
        Err(err) => println!("SessionHalted error: {err}"),
    }

    println!("State sequence: Greeting → CollectingInfo → Resolving → Done");
    println!("Demo complete: 3-state FSM executed with policy enforcement and context building.");
}
