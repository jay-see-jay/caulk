//! policy — generic tool allowlisting per State
//!
//! Wave 1: `Policy<S>` trait, `AllowAllPolicy`, `DenyListPolicy`, `AllowListPolicy`.
//!
//! Design notes:
//! - `allowed(&self, state: &S) -> &[ToolId]` returns the slice of tools allowed
//!   in that state. For `AllowAll` and `DenyList`, `allowed` returns an empty
//!   slice meaning "all allowed" (the set is infinite / complement). `check` is
//!   overridden to enforce the real logic.
//! - `DenyListPolicy` and `AllowListPolicy` are newtype-ish wrappers around
//!   `HashSet<ToolId>`. `AllowList` also keeps an ordered `Vec<ToolId>` to be
//!   able to return `&[ToolId]` without allocating.
//! - `FormeError::PolicyDenied` holds only `ToolId` in Wave 1. State is not
//!   included to keep the trait generic (`S: State` may not want to be in the
//!   error). The README originally wanted state in the display, but core spec
//!   simplified to tool-only. We document the omission and keep `Display` as
//!   `"policy denied tool `{tool}`"`.
//! - Trait is `Send + Sync` via supertrait bound so it can be used in async
//!   runtimes.
//!
//! Generic only, NO teacher logic.

use std::collections::HashSet;

use crate::core::{FormeError, State, ToolId};

/// Generic policy: which tools are allowed in a given state.
///
/// `S: State` is user-defined (e.g. `Idle`, `Teaching`). The policy decides
/// per-state whether a `ToolId` may run.
///
/// Implementations must be `Send + Sync` (supertrait) to be usable across
/// async tasks.
pub trait Policy<S: State>: Send + Sync {
    /// Slice of tools allowed in `state`.
    ///
    /// - For `AllowAllPolicy` and `DenyListPolicy`, this returns an empty
    ///   slice to signal “all allowed” / “all except denied” — see their
    ///   docs. Callers should prefer `check` / `is_allowed`.
    /// - For `AllowListPolicy`, this is the actual allow-list.
    fn allowed(&self, state: &S) -> &[ToolId];

    /// Check whether `tool` may run in `state`.
    ///
    /// Default impl: allow if `allowed` contains `tool`, else deny.
    /// Override when `allowed` is empty-meaning-all (e.g. `AllowAll`,
    /// `DenyList`).
    fn check(&self, state: &S, tool: &ToolId) -> Result<(), FormeError> {
        if self.allowed(state).contains(tool) {
            Ok(())
        } else {
            Err(FormeError::PolicyDenied(tool.clone()))
        }
    }

    /// Convenience: true if `check` would succeed.
    ///
    /// Default uses `check().is_ok()` so overriding `check` automatically
    /// fixes `is_allowed`.
    fn is_allowed(&self, state: &S, tool: &ToolId) -> bool {
        self.check(state, tool).is_ok()
    }
}

// ── AllowAll ────────────────────────────────────────────────────────────

/// Allows every tool in every state.
///
/// `allowed` returns `&[]` (empty) meaning “all”. `check` always returns
/// `Ok(())`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllowAllPolicy;

impl<S: State> Policy<S> for AllowAllPolicy {
    fn allowed(&self, _state: &S) -> &[ToolId] {
        &[]
    }

    fn check(&self, _state: &S, _tool: &ToolId) -> Result<(), FormeError> {
        Ok(())
    }

    fn is_allowed(&self, _state: &S, _tool: &ToolId) -> bool {
        true
    }
}

// ── DenyList ────────────────────────────────────────────────────────────

/// Denies a fixed set, allows everything else.
///
/// Newtype wrapper around `HashSet<ToolId>`. `allowed` returns `&[]`
/// (empty-meaning-all); `check` denies only if tool is in the inner set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DenyListPolicy(pub HashSet<ToolId>);

impl DenyListPolicy {
    /// Create from a `HashSet<ToolId>`.
    pub fn new(denied: HashSet<ToolId>) -> Self {
        Self(denied)
    }

    /// Empty deny list — allows all.
    pub fn empty() -> Self {
        Self(HashSet::new())
    }

    /// Create from any iterator of `ToolId`.
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = ToolId>,
    {
        Self(iter.into_iter().collect())
    }
}

impl From<HashSet<ToolId>> for DenyListPolicy {
    fn from(set: HashSet<ToolId>) -> Self {
        Self(set)
    }
}

impl<S: State> Policy<S> for DenyListPolicy {
    fn allowed(&self, _state: &S) -> &[ToolId] {
        &[]
    }

    fn check(&self, _state: &S, tool: &ToolId) -> Result<(), FormeError> {
        if self.0.contains(tool) {
            Err(FormeError::PolicyDenied(tool.clone()))
        } else {
            Ok(())
        }
    }

    fn is_allowed(&self, _state: &S, tool: &ToolId) -> bool {
        !self.0.contains(tool)
    }
}

// ── AllowList ───────────────────────────────────────────────────────────

/// Allows only the listed tools, denies everything else.
///
/// Stores both a `HashSet<ToolId>` for O(1) `check` and a `Vec<ToolId>` to
/// return `&[ToolId]` from `allowed` without allocation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllowListPolicy {
    set: HashSet<ToolId>,
    list: Vec<ToolId>,
}

impl AllowListPolicy {
    /// Create from any iterator of `ToolId`. Preserves first-seen order for
    /// `allowed` slice, deduped by set.
    pub fn new<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = ToolId>,
    {
        let mut set: HashSet<ToolId> = HashSet::new();
        let mut list: Vec<ToolId> = Vec::new();
        for t in iter {
            if set.insert(t.clone()) {
                list.push(t);
            }
        }
        Self { set, list }
    }

    /// Create from a `HashSet<ToolId>`. Order of `allowed` is arbitrary
    /// (HashSet iteration order) but contains exactly the set elements.
    pub fn from_set(set: HashSet<ToolId>) -> Self {
        let list: Vec<ToolId> = set.iter().cloned().collect();
        Self { set, list }
    }

    /// Empty allow-list — denies all (except policies that override).
    pub fn empty() -> Self {
        Self {
            set: HashSet::new(),
            list: Vec::new(),
        }
    }

    /// Access inner set.
    pub fn as_set(&self) -> &HashSet<ToolId> {
        &self.set
    }

    /// Length of allow-list.
    pub fn len(&self) -> usize {
        self.list.len()
    }

    /// True if no tools allowed.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
}

impl From<HashSet<ToolId>> for AllowListPolicy {
    fn from(set: HashSet<ToolId>) -> Self {
        Self::from_set(set)
    }
}

impl From<Vec<ToolId>> for AllowListPolicy {
    fn from(vec: Vec<ToolId>) -> Self {
        Self::new(vec)
    }
}

impl<S: State> Policy<S> for AllowListPolicy {
    fn allowed(&self, _state: &S) -> &[ToolId] {
        &self.list
    }

    fn check(&self, _state: &S, tool: &ToolId) -> Result<(), FormeError> {
        if self.set.contains(tool) {
            Ok(())
        } else {
            Err(FormeError::PolicyDenied(tool.clone()))
        }
    }

    fn is_allowed(&self, _state: &S, tool: &ToolId) -> bool {
        self.set.contains(tool)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FormeError, ToolId};
    use serde::{Deserialize, Serialize};
    use std::collections::HashSet;
    use std::fmt;

    #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct TestState(String);

    impl fmt::Display for TestState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    fn state_idle() -> TestState {
        TestState("Idle".into())
    }

    fn tool(s: &str) -> ToolId {
        ToolId::from(s)
    }

    // 1. AllowAll allows any tool (check passes)
    #[test]
    fn allow_all_allows_any() {
        let policy = AllowAllPolicy;
        let st = state_idle();
        let any = tool("write_code");
        assert!(policy.check(&st, &any).is_ok());
        assert!(policy.is_allowed(&st, &any));
        // allowed empty meaning all
        assert_eq!(policy.allowed(&st), &[] as &[ToolId]);
    }

    // 2. DenyList denies listed tool (check Err, is_denied true)
    #[test]
    fn deny_list_denies_listed() {
        let mut set = HashSet::new();
        set.insert(tool("bad_tool"));
        let policy = DenyListPolicy::new(set);
        let st = state_idle();
        let bad = tool("bad_tool");
        let good = tool("good_tool");

        let err = policy.check(&st, &bad).unwrap_err();
        assert!(err.is_denied());
        assert!(!policy.is_allowed(&st, &bad));

        assert!(policy.check(&st, &good).is_ok());
        assert!(policy.is_allowed(&st, &good));
    }

    // 3. DenyList message contains tool name
    #[test]
    fn deny_list_message_contains_tool_name() {
        let policy = DenyListPolicy::from_iter(vec![tool("search_web")]);
        let st = state_idle();
        let t = tool("search_web");
        let err = policy.check(&st, &t).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("search_web"),
            "msg '{msg}' should contain tool name"
        );
        // Display is "policy denied tool `search_web`" per core (via FormeError)
        assert!(msg.contains("policy denied"));
        // ensure is_denied true
        match err {
            FormeError::PolicyDenied(inner) => {
                assert_eq!(inner, t);
            }
            _ => panic!("expected PolicyDenied"),
        }
    }

    // 4. AllowList allows only listed, denies rest
    #[test]
    fn allow_list_allows_only_listed() {
        let policy = AllowListPolicy::new(vec![tool("read"), tool("write")]);
        let st = state_idle();

        assert!(policy.check(&st, &tool("read")).is_ok());
        assert!(policy.is_allowed(&st, &tool("read")));
        assert!(policy.check(&st, &tool("write")).is_ok());

        assert!(policy.check(&st, &tool("exec")).is_err());
        assert!(!policy.is_allowed(&st, &tool("exec")));
        assert!(!policy.is_allowed(&st, &tool("other")));
    }

    // 5. empty DenyList allows all
    #[test]
    fn empty_deny_list_allows_all() {
        let policy = DenyListPolicy::empty();
        let st = state_idle();
        assert!(policy.check(&st, &tool("anything")).is_ok());
        assert!(policy.check(&st, &tool("x")).is_ok());
        assert!(policy.is_allowed(&st, &tool("y")));
        assert_eq!(policy.allowed(&st).len(), 0);
    }

    // 6. check includes state? Actually PolicyDenied only holds ToolId per core spec, not state — error Display = "policy denied tool `{0}`" — does not include state. Documented omission.
    #[test]
    fn policy_denied_only_tool_not_state() {
        let policy = DenyListPolicy::from_iter(vec![tool("secret")]);
        let st = state_idle();
        let err = policy.check(&st, &tool("secret")).unwrap_err();
        let msg = err.to_string();
        // Should contain tool, not necessarily state (state omitted to keep generic)
        assert!(msg.contains("secret"));
        // FormeError::PolicyDenied holds ToolId only
        match err {
            FormeError::PolicyDenied(tid) => assert_eq!(tid.to_string(), "secret"),
            _ => panic!("wrong variant"),
        }
    }

    // 7. allowed() returns correct slice
    #[test]
    fn allow_list_allowed_returns_correct_slice() {
        let policy = AllowListPolicy::new(vec![tool("a"), tool("b"), tool("c")]);
        let st = state_idle();
        let allowed = policy.allowed(&st);
        assert_eq!(allowed.len(), 3);
        // order preserved by new()
        assert_eq!(allowed[0], tool("a"));
        assert_eq!(allowed[1], tool("b"));
        assert_eq!(allowed[2], tool("c"));
        // contains checks
        assert!(allowed.contains(&tool("a")));
        assert!(!allowed.contains(&tool("z")));
    }

    // 8. check edge empty AllowList denies all
    #[test]
    fn empty_allow_list_denies_all() {
        let policy = AllowListPolicy::empty();
        let st = state_idle();
        assert!(policy.allowed(&st).is_empty());
        assert!(policy.check(&st, &tool("any")).is_err());
        assert!(!policy.is_allowed(&st, &tool("any")));
        // FormeError variant
        let err = policy.check(&st, &tool("any")).unwrap_err();
        assert!(err.is_denied());
    }

    // 9. DenyList from HashSet and AllowList from_set
    #[test]
    fn from_set_conversions() {
        let mut hs = HashSet::new();
        hs.insert(tool("x"));
        hs.insert(tool("y"));
        let deny: DenyListPolicy = hs.clone().into();
        let st = state_idle();
        assert!(!deny.is_allowed(&st, &tool("x")));
        assert!(deny.is_allowed(&st, &tool("z")));

        let allow = AllowListPolicy::from_set(hs);
        assert_eq!(allow.len(), 2);
        assert!(allow.is_allowed(&st, &tool("x")));
        assert!(allow.is_allowed(&st, &tool("y")));
        assert!(!allow.is_allowed(&st, &tool("z")));
    }

    // 10. AllowAllPolicy default check vs is_allowed consistency
    #[test]
    fn allow_all_consistency() {
        let p = AllowAllPolicy;
        let st = TestState("Teaching".into());
        for name in ["a", "b", "c", "write_code", "tool_123"] {
            let t = tool(name);
            assert_eq!(p.check(&st, &t).is_ok(), p.is_allowed(&st, &t));
        }
    }

    // 11. Policy trait Send+Sync compile check
    #[test]
    fn policy_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AllowAllPolicy>();
        assert_send_sync::<DenyListPolicy>();
        assert_send_sync::<AllowListPolicy>();

        // Trait object bound check via generic
        fn assert_policy<S: crate::core::State, P: Policy<S>>() {}
        assert_policy::<TestState, AllowAllPolicy>();
        assert_policy::<TestState, DenyListPolicy>();
        assert_policy::<TestState, AllowListPolicy>();
    }
}
