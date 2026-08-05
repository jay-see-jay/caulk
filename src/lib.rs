//! forme — thin generic runtime between rig-core and graph-flow
//! Wave 0: core — Event, State, PromptKey, FormeError
//!
//! Destination: typed prompt registry (State × Event → Prompt), pluggable
//! context builder trait, generic policy trait (allowed tools per State).
//!
//! No teaching logic. No planner types in public API.

pub mod context_builder;
pub mod core;
pub mod graph_flow_adapter;
pub mod persistence;
pub mod policy;
pub mod prompt_registry;
pub mod rig_adapter;
pub mod runtime;

pub use core::{Event, FormeError, PromptKey, State, ToolId};
pub use runtime::{NextAction, Runner, StepOutput, ToolPlan};

/// Re-exported version string
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
