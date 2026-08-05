//! hello_world — minimal generic example for `forme`
//!
//! * State = Idle / Done
//! * Event = UserMsg
//! * Registry = InMemoryRegistry (HashMap PromptKey -> String)
//! * Context builder = NoopBuilder
//! * Policy = AllowAllPolicy
//! * LLM = MockLlm with canned reply (no real network call)
//!
//! Demonstrates:
//! - `PromptKey::key_for` canonical construction
//! - `InMemoryRegistry` usage
//! - `Runner::prepare` (pure, deterministic) + `Runner::step` (async, via MockLlm)
//!
//! Generic outside Ferriswheel / Inkwell / Loom, deterministic output
//! suitable for snapshot to `tests/snapshots/hello-world.txt`.

use forme::context_builder::NoopBuilder;
use forme::policy::AllowAllPolicy;
use forme::prompt_registry::InMemoryRegistry;
use forme::rig_adapter::MockLlm;
use forme::{PromptKey, Runner};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

// ── State ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum AppState {
    Idle,
    Done,
}

impl fmt::Display for AppState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Done => write!(f, "Done"),
        }
    }
}

// ── Event ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
enum AppEvent {
    UserMsg,
}

impl fmt::Display for AppEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserMsg => write!(f, "UserMsg"),
        }
    }
}

// ── tiny executor (no tokio dep) ──────────────────────────────────────────

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

fn main() {
    // ── registry ──────────────────────────────────────────────────────────
    let mut map = HashMap::new();
    map.insert(
        PromptKey::new("Idle", "UserMsg"),
        "You are a helpful assistant in Idle state. Greet the user warmly.".to_string(),
    );
    map.insert(
        PromptKey::new("Done", "UserMsg"),
        "You are in Done state. Summarize the conversation and say goodbye.".to_string(),
    );
    let registry = InMemoryRegistry::new(map);

    // ── builder / policy / llm ────────────────────────────────────────────
    let builder = NoopBuilder;
    let policy = AllowAllPolicy;
    let llm = MockLlm::with_response("Hello from forme! This is canned reply.");

    // ── runner ────────────────────────────────────────────────────────────
    let runner = Runner::new(Arc::new(registry), Arc::new(builder), Arc::new(policy), llm);

    let state = AppState::Idle;
    let event = AppEvent::UserMsg;
    let history: Vec<AppEvent> = vec![];

    // ── demonstrate PromptKey::key_for ────────────────────────────────────
    let key = PromptKey::key_for(&state, &event);
    let done_key = PromptKey::key_for(&AppState::Done, &event);

    println!("--- forme hello_world ---");
    println!("State: {state}");
    println!("Event: {event}");
    println!("PromptKey via key_for: {key}");
    println!("Canonical: {}", key.canonical());
    println!("Registry: InMemoryRegistry with 2 entries (Idle::UserMsg, Done::UserMsg)");
    println!();

    // ── prepare (pure, no LLM) ────────────────────────────────────────────
    let prepared = runner
        .prepare(&state, &event, &history)
        .expect("prepare should succeed for Idle::UserMsg");

    println!("[prepare]");
    println!("prompt: {}", prepared.prompt);
    println!("context: {:?}", prepared.context);
    println!("context len: {}", prepared.context.len());
    println!("(context is empty because NoopBuilder returns empty string)");
    println!("tool_plan.allowed len: {}", prepared.tool_plan.len());
    println!("tool_plan.allowed: {:?}", prepared.tool_plan.allowed);
    println!("(AllowAllPolicy returns empty slice meaning all tools allowed)");
    println!("key used: {}", prepared.key);
    println!();

    // ── step (includes MockLlm async call) ────────────────────────────────
    let output =
        block_on(runner.step(&state, &event, &history)).expect("step should succeed with MockLlm");

    println!("[step]");
    println!("llm_response: {}", output.llm_response);
    println!("next_state: {}", output.next_state);
    println!("prompt_key: {}", output.prompt_key);
    println!("prompt: {}", output.prompt);
    println!("context: {:?}", output.context);
    println!("tool_plan.allowed: {:?}", output.tool_plan.allowed);
    println!();

    println!("Done. Key for Done + UserMsg is: {done_key}");
    println!("All done. This example used MockLlm (no network).");
}
