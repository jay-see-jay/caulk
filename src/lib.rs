//! forme — thin generic runtime between rig-core and graph-flow
//! Wave 0: core — Event, State, PromptKey, FormeError
//!
//! Destination: typed prompt registry (State × Event → Prompt), pluggable
//! context builder trait, generic policy trait (allowed tools per State).
//!
//! No teaching logic. No wayfinder types in public API.

pub mod context_builder;
pub mod core;
pub mod persistence;
pub mod policy;
pub mod prompt_registry;

pub use core::{Event, FormeError, PromptKey, State, ToolId};

/// Re-exported version string
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
