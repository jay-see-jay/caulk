//! transition — pure `(state, event) -> (next_state, action)` lookup.
//!
//! A [`TransitionTable`] maps canonical [`PromptKey`]s (`"State::Event"`)
//! to [`Transition`]s. The lookup [`transition()`] is pure: no I/O, no LLM,
//! no time/RNG/env. An unmapped pair is an explicit [`TransitionMiss`],
//! never a panic, a default, or a stay-put fallback.
//!
//! # Keying decision
//!
//! Keys are [`PromptKey`]s built from `Display`, not `(S, E)` pairs, because
//! `Event` has no `Eq + Hash` bound. Consequence: payload-bearing events
//! collapse by `Display` (e.g. `InfoProvided("a")` and `InfoProvided("b")`
//! are the same key) — the same behaviour as prompt lookup today. Two
//! distinct states/events sharing a `Display` string likewise collide on
//! one key. `Display` must therefore stay stable (already required by
//! `core`); this is a documented precondition, not an enforced one.
//!
//! # NextAction mapping (for the driver, 001c — not done here)
//!
//! * `{next_state == current, Llm}` ≡ `Next` / `Continue`
//! * `{next_state != current, Llm}` ≡ `Transition(next_state)`
//! * `{_, Halt}` ≡ `Halt`
//! * `{_, Silent}` has no `NextAction` equivalent — skip `Runner::step`
//!   entirely.
//!
//! # Determinism
//!
//! `keys()` and `entries()` are sorted by canonical string, so no
//! `HashMap` iteration order leaks into any output. Same table + inputs
//! gives the same output regardless of insertion order.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::core::{Event, FormeError, PromptKey, State};

/// What the driver should do after the transition is resolved.
///
/// * `Llm` — call `Runner::step` for the current state.
/// * `Silent` — no LLM, no prompt lookup, no output.
/// * `Halt` — terminal, no further dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Llm,
    Silent,
    Halt,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llm => write!(f, "Llm"),
            Self::Silent => write!(f, "Silent"),
            Self::Halt => write!(f, "Halt"),
        }
    }
}

/// The result of a resolved transition: where to go, and what to do.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "S: State")]
pub struct Transition<S: State> {
    pub next_state: S,
    pub action: Action,
}

impl<S: State> Transition<S> {
    pub fn new(next_state: S, action: Action) -> Self {
        Self { next_state, action }
    }

    /// Resolve to `next_state` and call the LLM for it.
    pub fn llm(next_state: S) -> Self {
        Self::new(next_state, Action::Llm)
    }

    /// Resolve to `next_state` with no LLM call and no output.
    pub fn silent(next_state: S) -> Self {
        Self::new(next_state, Action::Silent)
    }

    /// Resolve to `next_state` and halt: no further dispatch.
    pub fn halt(next_state: S) -> Self {
        Self::new(next_state, Action::Halt)
    }

    pub fn is_halt(&self) -> bool {
        self.action == Action::Halt
    }
}

/// Explicit miss: the table has no row for the requested `(state, event)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionMiss {
    pub key: PromptKey,
}

impl TransitionMiss {
    pub fn new(key: PromptKey) -> Self {
        Self { key }
    }

    pub fn key(&self) -> &PromptKey {
        &self.key
    }
}

impl fmt::Display for TransitionMiss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no transition for {}", self.key)
    }
}

impl std::error::Error for TransitionMiss {}

impl From<TransitionMiss> for FormeError {
    fn from(miss: TransitionMiss) -> Self {
        Self::TransitionMiss(miss.key)
    }
}

/// Concrete, enumerable transition table: `PromptKey -> Transition<S>`.
///
/// Built with [`TransitionTable::with`] (builder) or [`TransitionTable::insert`].
/// Duplicate inserts overwrite: last wins, `len()` unchanged, previous
/// transition returned.
#[derive(Clone, Debug)]
pub struct TransitionTable<S: State> {
    entries: HashMap<PromptKey, Transition<S>>,
}

impl<S: State> Default for TransitionTable<S> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

impl<S: State> TransitionTable<S> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with<E: Event>(mut self, state: &S, event: &E, transition: Transition<S>) -> Self {
        self.insert(state, event, transition);
        self
    }

    pub fn insert<E: Event>(
        &mut self,
        state: &S,
        event: &E,
        transition: Transition<S>,
    ) -> Option<Transition<S>> {
        self.entries
            .insert(PromptKey::key_for(state, event), transition)
    }

    pub fn get<E: Event>(&self, state: &S, event: &E) -> Option<&Transition<S>> {
        self.entries.get(&PromptKey::key_for(state, event))
    }

    pub fn contains<E: Event>(&self, state: &S, event: &E) -> bool {
        self.entries.contains_key(&PromptKey::key_for(state, event))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All keys, sorted by canonical string (deterministic order).
    pub fn keys(&self) -> Vec<PromptKey> {
        let mut keys: Vec<PromptKey> = self.entries.keys().cloned().collect();
        keys.sort_by_key(|a| a.canonical());
        keys
    }

    /// All `(key, transition)` pairs, sorted by canonical key.
    pub fn entries(&self) -> Vec<(PromptKey, Transition<S>)> {
        let mut entries: Vec<(PromptKey, Transition<S>)> = self
            .entries
            .iter()
            .map(|(key, transition)| (key.clone(), transition.clone()))
            .collect();
        entries.sort_by_key(|(key, _)| key.canonical());
        entries
    }
}

/// Pure lookup: resolve `(state, event)` against the table.
///
/// Returns `Err(TransitionMiss)` for an unmapped pair — never a default.
pub fn transition<S: State, E: Event>(
    table: &TransitionTable<S>,
    state: &S,
    event: &E,
) -> Result<Transition<S>, TransitionMiss> {
    match table.get(state, event) {
        Some(found) => Ok(found.clone()),
        None => Err(TransitionMiss::new(PromptKey::key_for(state, event))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FormeError, PromptKey};
    use serde::{Deserialize, Serialize};
    use std::fmt;

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

    #[derive(Clone, Debug, Serialize, Deserialize)]
    enum BotEvent {
        UserMsg,
        Tick,
        UserAskedRefund,
        InfoProvided(String),
        RefundRequested,
    }
    impl fmt::Display for BotEvent {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::UserMsg => write!(f, "UserMsg"),
                Self::Tick => write!(f, "Tick"),
                Self::UserAskedRefund => write!(f, "UserAskedRefund"),
                // Payload intentionally ignored: payload-bearing events
                // collapse by Display, same as prompt lookup today.
                Self::InfoProvided(_) => write!(f, "InfoProvided"),
                Self::RefundRequested => write!(f, "RefundRequested"),
            }
        }
    }

    fn idle_table() -> TransitionTable<BotState> {
        TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::llm(BotState::Done),
        )
    }

    fn support_bot_table() -> TransitionTable<BotState> {
        TransitionTable::new()
            .with(
                &BotState::Greeting,
                &BotEvent::UserAskedRefund,
                Transition::llm(BotState::CollectingInfo),
            )
            .with(
                &BotState::CollectingInfo,
                &BotEvent::InfoProvided("order #1".to_string()),
                Transition::llm(BotState::Resolving),
            )
            .with(
                &BotState::Resolving,
                &BotEvent::RefundRequested,
                Transition::halt(BotState::Done),
            )
    }

    // 1. lookup_defined_pair_returns_transition
    #[test]
    fn lookup_defined_pair_returns_transition() {
        let table = idle_table();
        let got = transition(&table, &BotState::Idle, &BotEvent::UserMsg)
            .expect("Idle x UserMsg is mapped");
        assert_eq!(
            got,
            Transition {
                next_state: BotState::Done,
                action: Action::Llm,
            }
        );
    }

    // 2. lookup_unmapped_returns_explicit_miss
    #[test]
    fn lookup_unmapped_returns_explicit_miss() {
        let table = idle_table();
        let err = transition(&table, &BotState::Done, &BotEvent::UserMsg)
            .expect_err("Done x UserMsg is unmapped");
        assert_eq!(
            err,
            TransitionMiss {
                key: PromptKey::new("Done", "UserMsg"),
            }
        );
        assert_eq!(err.key(), &PromptKey::new("Done", "UserMsg"));
        assert!(
            err.to_string().contains("Done::UserMsg"),
            "miss display was: {err}"
        );
        let forme_err = FormeError::from(err);
        assert!(forme_err.is_transition_miss());
        assert!(matches!(forme_err, FormeError::TransitionMiss(_)));
    }

    // 3. miss_is_not_a_default
    #[test]
    fn miss_is_not_a_default() {
        let table = idle_table();
        let result = transition(&table, &BotState::Done, &BotEvent::UserMsg);
        assert!(result.is_err(), "unmapped pair must not yield a fallback");
    }

    // 4. insert_overwrites_and_len_unchanged
    #[test]
    fn insert_overwrites_and_len_unchanged() {
        let mut table = TransitionTable::new();
        let first = table.insert(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::llm(BotState::Done),
        );
        assert_eq!(first, None);
        let previous = table.insert(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::silent(BotState::Idle),
        );
        assert_eq!(previous, Some(Transition::llm(BotState::Done)));
        assert_eq!(table.len(), 1);
        assert_eq!(
            transition(&table, &BotState::Idle, &BotEvent::UserMsg).unwrap(),
            Transition::silent(BotState::Idle)
        );
    }

    // 5. self_loop_transition
    #[test]
    fn self_loop_transition() {
        let table = TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::llm(BotState::Idle),
        );
        let got = transition(&table, &BotState::Idle, &BotEvent::UserMsg).unwrap();
        assert_eq!(got.next_state, BotState::Idle);
        assert_eq!(got.action, Action::Llm);
    }

    // 6. halt_transition
    #[test]
    fn halt_transition() {
        let table = TransitionTable::new().with(
            &BotState::Done,
            &BotEvent::UserMsg,
            Transition::halt(BotState::Done),
        );
        let got = transition(&table, &BotState::Done, &BotEvent::UserMsg).unwrap();
        assert!(got.is_halt());
        assert_eq!(got.action, Action::Halt);
    }

    // 7. silent_transition_needs_no_llm
    #[test]
    fn silent_transition_needs_no_llm() {
        let table = TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::Tick,
            Transition::silent(BotState::Idle),
        );
        let got = transition(&table, &BotState::Idle, &BotEvent::Tick).unwrap();
        assert_eq!(got, Transition::silent(BotState::Idle));
    }

    // 8. determinism_across_insertion_orders
    #[test]
    fn determinism_across_insertion_orders() {
        let a = support_bot_table();
        let b = TransitionTable::new()
            .with(
                &BotState::Resolving,
                &BotEvent::RefundRequested,
                Transition::halt(BotState::Done),
            )
            .with(
                &BotState::CollectingInfo,
                &BotEvent::InfoProvided("order #1".to_string()),
                Transition::llm(BotState::Resolving),
            )
            .with(
                &BotState::Greeting,
                &BotEvent::UserAskedRefund,
                Transition::llm(BotState::CollectingInfo),
            );
        assert_eq!(a.keys(), b.keys());
        assert_eq!(a.entries(), b.entries());
        for (state, event) in [
            (BotState::Greeting, BotEvent::UserAskedRefund),
            (
                BotState::CollectingInfo,
                BotEvent::InfoProvided("order #1".to_string()),
            ),
            (BotState::Resolving, BotEvent::RefundRequested),
        ] {
            assert_eq!(
                transition(&a, &state, &event),
                transition(&b, &state, &event)
            );
        }
    }

    // 9. keys_sorted_for_coverage
    #[test]
    fn keys_sorted_for_coverage() {
        let table = support_bot_table();
        let canonical: Vec<String> = table.keys().iter().map(|k| k.canonical()).collect();
        assert_eq!(
            canonical,
            vec![
                "CollectingInfo::InfoProvided",
                "Greeting::UserAskedRefund",
                "Resolving::RefundRequested",
            ]
        );
    }

    // 10. payload_events_collapse_by_display
    #[test]
    fn payload_events_collapse_by_display() {
        let table = support_bot_table();
        let first = transition(
            &table,
            &BotState::CollectingInfo,
            &BotEvent::InfoProvided("order #1".to_string()),
        )
        .unwrap();
        let second = transition(
            &table,
            &BotState::CollectingInfo,
            &BotEvent::InfoProvided("order #2".to_string()),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.next_state, BotState::Resolving);
    }

    // 11. serde_roundtrip_transition_and_action
    #[test]
    fn serde_roundtrip_transition_and_action() {
        let original = Transition::halt(BotState::Done);
        let json = serde_json::to_string(&original).expect("serialize transition");
        let decoded: Transition<BotState> =
            serde_json::from_str(&json).expect("deserialize transition");
        assert_eq!(original, decoded);

        let action = Action::Halt;
        let json = serde_json::to_string(&action).expect("serialize action");
        let decoded: Action = serde_json::from_str(&json).expect("deserialize action");
        assert_eq!(action, decoded);
    }

    // 12. empty_table_misses_everything
    #[test]
    fn empty_table_misses_everything() {
        let table: TransitionTable<BotState> = TransitionTable::new();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
        assert!(table.keys().is_empty());
        let err = transition(&table, &BotState::Idle, &BotEvent::UserMsg)
            .expect_err("empty table misses everything");
        assert_eq!(err.key(), &PromptKey::new("Idle", "UserMsg"));
    }

    // 13. contains_matches_get
    #[test]
    fn contains_matches_get() {
        let table = idle_table();
        for (state, event) in [
            (BotState::Idle, BotEvent::UserMsg),
            (BotState::Done, BotEvent::UserMsg),
            (BotState::Idle, BotEvent::Tick),
        ] {
            assert_eq!(
                table.contains(&state, &event),
                table.get(&state, &event).is_some(),
                "contains/get disagree for {state} x {event}"
            );
        }
        assert!(table.contains(&BotState::Idle, &BotEvent::UserMsg));
        assert!(!table.contains(&BotState::Done, &BotEvent::UserMsg));
    }

    // 14. with_builder_chains
    #[test]
    fn with_builder_chains() {
        let via_with = TransitionTable::new()
            .with(
                &BotState::Idle,
                &BotEvent::UserMsg,
                Transition::llm(BotState::Done),
            )
            .with(
                &BotState::Idle,
                &BotEvent::Tick,
                Transition::silent(BotState::Idle),
            );
        let mut via_insert = TransitionTable::new();
        via_insert.insert(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::llm(BotState::Done),
        );
        via_insert.insert(
            &BotState::Idle,
            &BotEvent::Tick,
            Transition::silent(BotState::Idle),
        );
        assert_eq!(via_with.entries(), via_insert.entries());
        assert_eq!(via_with.len(), via_insert.len());
    }

    // 15. action_display_is_stable_variant_name
    #[test]
    fn action_display_is_stable_variant_name() {
        assert_eq!(Action::Llm.to_string(), "Llm");
        assert_eq!(Action::Silent.to_string(), "Silent");
        assert_eq!(Action::Halt.to_string(), "Halt");
    }

    // 16. transition_miss_display
    #[test]
    fn transition_miss_display() {
        let miss = TransitionMiss::new(PromptKey::new("Idle", "UserMsg"));
        assert_eq!(miss.to_string(), "no transition for Idle::UserMsg");
    }
}
