//! coverage — prompt coverage report: which `Llm` rows of a transition table have a prompt in the registry. Only `Llm` rows need a prompt; `Silent` and `Halt` rows never look one up, so they are reported as skipped, never as missing.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::core::{FormeError, PromptKey, State};
use crate::prompt_registry::Registry;
use crate::transition::{Action, Transition, TransitionTable};

/// Prompt coverage for a transition table: which `Llm` rows have a prompt,
/// which are missing one, and which rows were skipped (never need one).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageReport {
    pub total_rows: usize,
    pub checked: usize,
    pub covered: usize,
    pub missing: Vec<PromptKey>,
    pub skipped: Vec<(PromptKey, Action)>,
}

impl CoverageReport {
    /// The `Llm` rows with no prompt, in canonical-key order.
    pub fn missing(&self) -> &[PromptKey] {
        &self.missing
    }

    /// The `Silent`/`Halt` rows that were not checked, as `(key, action)` pairs.
    pub fn skipped(&self) -> &[(PromptKey, Action)] {
        &self.skipped
    }

    /// True exactly when `missing` is empty — every `Llm` row has a prompt.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }

    /// `" (skipped: key [action], ...)"`, or `""` when nothing was skipped.
    fn skipped_suffix(&self) -> String {
        if self.skipped.is_empty() {
            return String::new();
        }
        let listed: Vec<String> = self
            .skipped
            .iter()
            .map(|(key, action)| format!("{key} [{action}]"))
            .collect();
        format!(" (skipped: {})", listed.join(", "))
    }

    /// Missing keys as one comma-separated string, in canonical order.
    fn missing_list(&self) -> String {
        self.missing
            .iter()
            .map(|key| key.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl fmt::Display for CoverageReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.missing.is_empty() {
            write!(
                f,
                "prompt coverage complete: {}/{} Llm rows covered{}",
                self.covered,
                self.checked,
                self.skipped_suffix()
            )
        } else {
            write!(
                f,
                "prompt coverage incomplete: {}/{} Llm rows covered; missing: {}",
                self.covered,
                self.checked,
                self.missing_list()
            )
        }
    }
}

/// Check every `Llm` row of `table` against `registry`.
///
/// `RegistryNotFound` for a key records it in `missing` and the scan
/// continues; any other registry error propagates immediately.
pub fn check_prompt_coverage<S: State, R: Registry>(
    table: &TransitionTable<S>,
    registry: &R,
) -> Result<CoverageReport, FormeError> {
    let mut report = CoverageReport {
        total_rows: table.len(),
        ..CoverageReport::default()
    };
    for (key, transition) in table.entries() {
        classify_row(&mut report, registry, key, transition)?;
    }
    Ok(report)
}

/// Classify one row: an `Llm` row is looked up, anything else is skipped.
fn classify_row<S: State, R: Registry>(
    report: &mut CoverageReport,
    registry: &R,
    key: PromptKey,
    transition: Transition<S>,
) -> Result<(), FormeError> {
    match transition.action {
        Action::Llm => record_llm_row(report, registry, key),
        action => {
            report.skipped.push((key, action));
            Ok(())
        }
    }
}

/// Count one `Llm` row as covered or missing, by registry lookup.
fn record_llm_row<R: Registry>(
    report: &mut CoverageReport,
    registry: &R,
    key: PromptKey,
) -> Result<(), FormeError> {
    report.checked += 1;
    if lookup_status(registry, &key)? {
        report.covered += 1;
        return Ok(());
    }
    report.missing.push(key);
    Ok(())
}

/// Look one key up: `Ok(true)` — prompt present; `Ok(false)` — not found;
/// `Err` — any other registry error, propagated unchanged.
fn lookup_status<R: Registry>(registry: &R, key: &PromptKey) -> Result<bool, FormeError> {
    match registry.get(key) {
        Ok(_) => Ok(true),
        Err(err) if err.is_not_found() => Ok(false),
        Err(err) => Err(err),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FormeError, PromptKey};
    use crate::prompt_registry::{InMemoryRegistry, Registry};
    use crate::transition::{Action, Transition, TransitionTable};
    use serde::{Deserialize, Serialize};
    use std::collections::{HashMap, HashSet};
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
                Self::InfoProvided(_) => write!(f, "InfoProvided"),
                Self::RefundRequested => write!(f, "RefundRequested"),
            }
        }
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
                &BotEvent::InfoProvided("order #1".to_string()),
                Transition::llm(BotState::Resolving),
            )
            .with(
                &BotState::Resolving,
                &BotEvent::RefundRequested,
                Transition::halt(BotState::Done),
            )
    }

    /// The three `Llm` rows used by the determinism and mid-scan-error
    /// tests, in canonical-key order.
    fn three_llm_rows() -> Vec<(BotState, BotEvent, BotState)> {
        vec![
            (
                BotState::CollectingInfo,
                BotEvent::InfoProvided("order #1".to_string()),
                BotState::Resolving,
            ),
            (
                BotState::Greeting,
                BotEvent::UserAskedRefund,
                BotState::CollectingInfo,
            ),
            (
                BotState::Resolving,
                BotEvent::RefundRequested,
                BotState::Done,
            ),
        ]
    }

    fn table_from_rows(
        rows: impl IntoIterator<Item = (BotState, BotEvent, BotState)>,
    ) -> TransitionTable<BotState> {
        let mut table = TransitionTable::new();
        for (state, event, next) in rows {
            table.insert(&state, &event, Transition::llm(next));
        }
        table
    }

    fn registry_with(keys: &[(&str, &str)]) -> InMemoryRegistry {
        let map: HashMap<PromptKey, String> = keys
            .iter()
            .map(|(key, prompt)| {
                let (state, event) = key.split_once("::").expect("key is State::Event");
                (PromptKey::new(state, event), prompt.to_string())
            })
            .collect();
        InMemoryRegistry::new(map)
    }

    /// Every `get` fails with a non-not-found error.
    struct FailingRegistry;
    impl Registry for FailingRegistry {
        fn get(&self, _key: &PromptKey) -> Result<String, FormeError> {
            Err(FormeError::PersistenceFailed("disk full".to_string()))
        }
    }

    /// Present prompts for some keys, `PersistenceFailed` for others.
    struct SelectiveFailRegistry {
        present: HashMap<PromptKey, String>,
        failing: HashSet<PromptKey>,
    }
    impl Registry for SelectiveFailRegistry {
        fn get(&self, key: &PromptKey) -> Result<String, FormeError> {
            if self.failing.contains(key) {
                return Err(FormeError::PersistenceFailed("disk full".to_string()));
            }
            match self.present.get(key) {
                Some(prompt) => Ok(prompt.clone()),
                None => Err(FormeError::RegistryNotFound(key.clone())),
            }
        }
    }

    // 1. full_coverage_zero_missing
    #[test]
    fn full_coverage_zero_missing() {
        let table = support_table();
        let registry = registry_with(&[
            ("Greeting::UserAskedRefund", "refund prompt"),
            ("CollectingInfo::InfoProvided", "info prompt"),
        ]);
        let report = check_prompt_coverage(&table, &registry).expect("coverage check");
        assert_eq!(report.total_rows, 3);
        assert_eq!(report.checked, 2);
        assert_eq!(report.covered, 2);
        assert!(report.missing.is_empty());
        assert_eq!(
            report.skipped,
            vec![(PromptKey::new("Resolving", "RefundRequested"), Action::Halt)]
        );
        assert!(report.is_complete());
        assert_eq!(
            report.to_string(),
            "prompt coverage complete: 2/2 Llm rows covered (skipped: Resolving::RefundRequested [Halt])"
        );
    }

    // 2. one_missing_prompt_names_exact_key
    #[test]
    fn one_missing_prompt_names_exact_key() {
        let table = support_table();
        let registry = registry_with(&[("Greeting::UserAskedRefund", "refund prompt")]);
        let report = check_prompt_coverage(&table, &registry).expect("coverage check");
        assert_eq!(report.checked, 2);
        assert_eq!(report.covered, 1);
        assert_eq!(
            report.missing,
            vec![PromptKey::new("CollectingInfo", "InfoProvided")]
        );
        assert!(!report.is_complete());
        assert_eq!(
            report.to_string(),
            "prompt coverage incomplete: 1/2 Llm rows covered; missing: CollectingInfo::InfoProvided"
        );
    }

    // 3. empty_table_empty_report_no_failure
    #[test]
    fn empty_table_empty_report_no_failure() {
        let table: TransitionTable<BotState> = TransitionTable::new();
        let registry = InMemoryRegistry::new(HashMap::new());
        let report = check_prompt_coverage(&table, &registry).expect("coverage check");
        assert_eq!(report.total_rows, 0);
        assert_eq!(report.checked, 0);
        assert_eq!(report.covered, 0);
        assert!(report.missing.is_empty());
        assert!(report.skipped.is_empty());
        assert!(report.is_complete());
        assert_eq!(
            report.to_string(),
            "prompt coverage complete: 0/0 Llm rows covered"
        );
    }

    // 4. silent_row_needs_no_prompt
    #[test]
    fn silent_row_needs_no_prompt() {
        let table = TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::Tick,
            Transition::silent(BotState::Idle),
        );
        let registry = InMemoryRegistry::new(HashMap::new());
        let report = check_prompt_coverage(&table, &registry).expect("coverage check");
        assert!(report.missing.is_empty());
        assert!(report.is_complete());
        assert_eq!(
            report.skipped,
            vec![(PromptKey::new("Idle", "Tick"), Action::Silent)]
        );
        assert_eq!(report.checked, 0);
    }

    // 5. halt_row_needs_no_prompt
    #[test]
    fn halt_row_needs_no_prompt() {
        let table = TransitionTable::new().with(
            &BotState::Resolving,
            &BotEvent::RefundRequested,
            Transition::halt(BotState::Done),
        );
        let registry = InMemoryRegistry::new(HashMap::new());
        let report = check_prompt_coverage(&table, &registry).expect("coverage check");
        assert!(report.missing.is_empty());
        assert!(report.is_complete());
        assert_eq!(
            report.skipped,
            vec![(PromptKey::new("Resolving", "RefundRequested"), Action::Halt)]
        );
    }

    // 6. mixed_table_only_llm_rows_checked
    #[test]
    fn mixed_table_only_llm_rows_checked() {
        // support_table's rows plus one Silent row.
        let table = support_table().with(
            &BotState::Idle,
            &BotEvent::Tick,
            Transition::silent(BotState::Idle),
        );
        let registry = registry_with(&[("Greeting::UserAskedRefund", "refund prompt")]);
        let report = check_prompt_coverage(&table, &registry).expect("coverage check");
        assert_eq!(report.total_rows, 4);
        assert_eq!(report.checked, 2);
        assert_eq!(report.covered, 1);
        assert_eq!(
            report.missing,
            vec![PromptKey::new("CollectingInfo", "InfoProvided")]
        );
        assert_eq!(
            report.skipped,
            vec![
                (PromptKey::new("Idle", "Tick"), Action::Silent),
                (PromptKey::new("Resolving", "RefundRequested"), Action::Halt),
            ]
        );
        assert_eq!(report.total_rows, report.checked + report.skipped.len());
    }

    // 7. registry_non_not_found_error_propagates
    #[test]
    fn registry_non_not_found_error_propagates() {
        let table = TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::llm(BotState::Done),
        );
        let err = check_prompt_coverage(&table, &FailingRegistry)
            .expect_err("a broken registry is an error, not missing prompts");
        assert!(matches!(err, FormeError::PersistenceFailed(_)));
        assert!(err.to_string().contains("disk full"), "error was: {err}");
    }

    // 8. registry_error_mid_scan_propagates_no_partial_report
    #[test]
    fn registry_error_mid_scan_propagates_no_partial_report() {
        let table = table_from_rows(three_llm_rows());
        let registry = SelectiveFailRegistry {
            present: HashMap::from([(
                PromptKey::new("CollectingInfo", "InfoProvided"),
                "info prompt".to_string(),
            )]),
            failing: HashSet::from([PromptKey::new("Resolving", "RefundRequested")]),
        };
        let err = check_prompt_coverage(&table, &registry)
            .expect_err("scan must die at the failing key, no partial report");
        assert!(matches!(err, FormeError::PersistenceFailed(_)));
    }

    // 9. missing_list_deterministically_sorted
    #[test]
    fn missing_list_deterministically_sorted() {
        let rows = three_llm_rows();
        let reverse = table_from_rows(rows.iter().rev().cloned());
        let forward = table_from_rows(rows);
        let registry = registry_with(&[("Greeting::UserAskedRefund", "refund prompt")]);
        let expected_missing = vec![
            PromptKey::new("CollectingInfo", "InfoProvided"),
            PromptKey::new("Resolving", "RefundRequested"),
        ];
        let report = check_prompt_coverage(&reverse, &registry).expect("coverage check");
        assert_eq!(report.missing, expected_missing);
        let again = check_prompt_coverage(&forward, &registry).expect("coverage check");
        assert_eq!(report, again);
    }

    // 10. payload_events_collapse_by_display
    #[test]
    fn payload_events_collapse_by_display() {
        let table = TransitionTable::new().with(
            &BotState::CollectingInfo,
            &BotEvent::InfoProvided("order #1".to_string()),
            Transition::llm(BotState::Resolving),
        );
        let registry = registry_with(&[("CollectingInfo::InfoProvided", "info prompt")]);
        let report = check_prompt_coverage(&table, &registry).expect("coverage check");
        assert_eq!(report.total_rows, 1);
        assert_eq!(report.covered, 1);
        assert!(report.missing.is_empty());
    }

    // 11. duplicate_insert_overwrites_single_row
    #[test]
    fn duplicate_insert_overwrites_single_row() {
        let mut table = TransitionTable::new();
        table.insert(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::llm(BotState::Done),
        );
        table.insert(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::silent(BotState::Idle),
        );
        assert_eq!(table.len(), 1);
        let registry = InMemoryRegistry::new(HashMap::new());
        let report = check_prompt_coverage(&table, &registry).expect("coverage check");
        assert_eq!(report.total_rows, 1);
        assert_eq!(report.checked, 0);
        assert_eq!(
            report.skipped,
            vec![(PromptKey::new("Idle", "UserMsg"), Action::Silent)]
        );
        assert!(report.missing.is_empty());
    }

    // 12. empty_prompt_string_counts_as_covered
    #[test]
    fn empty_prompt_string_counts_as_covered() {
        let table = TransitionTable::new().with(
            &BotState::Idle,
            &BotEvent::UserMsg,
            Transition::llm(BotState::Done),
        );
        let registry = registry_with(&[("Idle::UserMsg", "")]);
        let report = check_prompt_coverage(&table, &registry).expect("coverage check");
        assert_eq!(report.covered, 1);
        assert!(report.missing.is_empty());
        assert!(report.is_complete());
    }
}
