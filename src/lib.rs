//! caulk — thin generic runtime between rig-core and graph-flow
//! Wave 0: core — Event, State, PromptKey, FormeError
//!
//! Destination: typed prompt registry (State × Event → Prompt), pluggable
//! context builder trait, generic policy trait (allowed tools per State).
//!
//! No teaching logic. No planner types in public API.

pub mod context_builder;
pub mod core;
pub mod driver;
pub mod graph_flow_adapter;
pub mod persistence;
pub mod policy;
pub mod prompt_registry;
pub mod rig_adapter;
pub mod runtime;
pub mod scorer;
pub mod transition;

pub use core::{Event, FormeError, PromptKey, State, ToolId};
pub use driver::{DispatchOutput, SessionDriver};
pub use runtime::{NextAction, Runner, StepOutput, ToolPlan};
pub use scorer::{AlwaysEscalateScorer, AlwaysSilentScorer, MockScorer, Score, Scorer, Verdict};
pub use transition::{transition, Action, Transition, TransitionMiss, TransitionTable};

/// Re-exported version string
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
