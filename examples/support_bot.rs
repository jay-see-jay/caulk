//! support_bot — 3-state refund FSM with policy deny demo
//!
//! States: Greeting -> CollectingInfo -> Resolving
//! Events: UserAskedRefund, InfoProvided, RefundRequested
//! Demonstrates:
//! - ParagraphBuilder for prose context
//! - InMemoryRegistry with per-state prompts
//! - Custom per-state Policy (SupportPolicy)
//! - Runner::prepare, Runner::step (MockLlm), Runner::handle_edge
//! - ToolDenied via FormeError

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use forme::context_builder::{ContextBuilder, ParagraphBuilder};
use forme::core::{FormeError, PromptKey, ToolId};
use forme::policy::Policy;
use forme::prompt_registry::InMemoryRegistry;
use forme::rig_adapter::MockLlm;
use forme::runtime::{NextAction, Runner};
use serde::{Deserialize, Serialize};

// ── State ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum SupportState {
    Greeting,
    CollectingInfo,
    Resolving,
}

impl fmt::Display for SupportState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Greeting => write!(f, "Greeting"),
            Self::CollectingInfo => write!(f, "CollectingInfo"),
            Self::Resolving => write!(f, "Resolving"),
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

fn main() {
    // ── Registry (3-4 entries, each state different prompt) ──────────────────
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
    // extra 4th entry to show registry size >3
    map.insert(
        PromptKey::new("Greeting", "RefundRequested"),
        "GREETING fallback for refund request: still greet and route to collection.".into(),
    );

    let registry = Arc::new(InMemoryRegistry::new(map));

    // ── Context builder (ParagraphBuilder prose, N=2) ────────────────────────
    let builder = Arc::new(ParagraphBuilder::with_n(2));

    // demonstrate builder is generic and Send+Sync
    let _builder_check: &dyn ContextBuilder<SupportEvent> = builder.as_ref();

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

    // ── Runner ───────────────────────────────────────────────────────────────
    let runner = Runner::new(
        Arc::clone(&registry),
        Arc::clone(&builder),
        Arc::clone(&policy),
        llm,
    );

    // ── Simulate 3-turn conversation ───────────────────────────────────────
    let turns: Vec<(SupportState, SupportEvent)> = vec![
        (SupportState::Greeting, SupportEvent::UserAskedRefund),
        (
            SupportState::CollectingInfo,
            SupportEvent::InfoProvided("order #12345, bought yesterday".into()),
        ),
        (SupportState::Resolving, SupportEvent::RefundRequested),
    ];

    let mut history: Vec<SupportEvent> = Vec::new();

    for (idx, (state, event)) in turns.iter().enumerate() {
        println!("=== Turn {}: State={} Event={} ===", idx + 1, state, event);

        // ---- prepare (pure, deterministic) ---------------------------------
        let prepared = runner
            .prepare(state, event, &history)
            .expect("prepare should succeed");
        println!(
            "prompt_key: {} (canonical {})",
            prepared.key,
            prepared.key.canonical()
        );
        println!("prompt: {}", prepared.prompt);
        println!("context (ParagraphBuilder N=2): {}", prepared.context);
        let allowed_strs: Vec<String> = prepared
            .tool_plan
            .allowed
            .iter()
            .map(|t| t.to_string())
            .collect();
        println!("allowed tools: {:?}", allowed_strs);

        // ---- policy deny demo ----------------------------------------------
        let denied_attempt: ToolId = match state {
            SupportState::Greeting => ToolId::from("process_refund"),
            SupportState::CollectingInfo => ToolId::from("process_refund"),
            SupportState::Resolving => ToolId::from("ask_clarification"),
        };

        match runner.policy().check(state, &denied_attempt) {
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
                // show is_denied helper
                let err = FormeError::PolicyDenied(tid.clone());
                println!("  is_denied(): {}", err.is_denied());
            }
            Err(other) => {
                println!("unexpected policy error: {other}");
            }
        }

        // ---- full step with LLM --------------------------------------------
        let step_out = block_on(runner.step(state, event, &history)).expect("step ok");
        println!("llm_response: {}", step_out.llm_response);
        println!(
            "tool_plan snapshot len {} contains refund? {}",
            step_out.tool_plan.len(),
            step_out.tool_plan.contains(&ToolId::from("process_refund"))
        );

        // ---- handle_edge with NextAction::Transition ------------------------
        if idx + 1 < turns.len() {
            let next_state = turns[idx + 1].0.clone();
            let action = NextAction::Transition(next_state.clone());
            let next_key = runner.handle_edge(&action, state, event);
            println!(
                "handle_edge NextAction::Transition({}) -> PromptKey {}",
                next_state, next_key
            );
            // also show Next stays same
            let stay_action = NextAction::Next;
            let stay_key = runner.handle_edge(&stay_action, state, event);
            println!("handle_edge NextAction::Next stays -> {}", stay_key);
        } else {
            let halt = NextAction::Halt;
            let halt_key = runner.handle_edge(&halt, state, event);
            println!("handle_edge NextAction::Halt -> {} (logging)", halt_key);
        }

        println!();
        history.push(event.clone());
    }

    println!("Demo complete: 3-state FSM executed with policy enforcement and context building.");
}
