use std::fmt;
use std::hash::Hash;

use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

// ── Traits ──────────────────────────────────────────────────────────────

/// User-defined event for the forme state machine.
///
/// # Object-safety
/// This trait is **NOT** object-safe: it requires `Clone` (which returns `Self`)
/// and `Serialize` / `DeserializeOwned`. Use as a generic bound `E: Event`,
/// not as `dyn Event`. The blanket impl below makes any compatible type an `Event`
/// automatically, avoiding boilerplate.
///
/// Keep `Display` stable — it is used for canonical `PromptKey` construction.
pub trait Event:
    Clone + fmt::Debug + fmt::Display + Send + Sync + 'static + Serialize + DeserializeOwned
{
}

impl<T> Event for T where
    T: Clone + fmt::Debug + fmt::Display + Send + Sync + 'static + Serialize + DeserializeOwned
{
}

/// User-defined state for the forme state machine.
///
/// # Object-safety
/// NOT object-safe for the same reasons as `Event` (`Clone` returns `Self`,
/// plus `Eq + Hash`). Use as `S: State`. Blanket impl provides ergonomics.
///
/// `Display` should be a stable canonical form (e.g. `"Idle"`, `"Teaching"`).
pub trait State:
    Clone + fmt::Debug + fmt::Display + Eq + Hash + Send + Sync + 'static + Serialize + DeserializeOwned
{
}

impl<T> State for T where
    T: Clone
        + fmt::Debug
        + fmt::Display
        + Eq
        + Hash
        + Send
        + Sync
        + 'static
        + Serialize
        + DeserializeOwned
{
}

// ── PromptKey ───────────────────────────────────────────────────────────

/// Concrete registry key: `(state, event)`.
///
/// Canonicalized via `Display` when built from typed `State`/`Event` via
/// `key_for`, or taken verbatim when built from raw strings.
///
/// Empty strings are allowed (useful for testing or initial states) but callers
/// should prefer non-empty stable identifiers. `Display` is stable and equals
/// `canonical()` (`"{state}::{event}"`). `Debug` is the standard struct-like
/// debug representation and differs from `Display`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub struct PromptKey {
    pub state: String,
    pub event: String,
}

impl PromptKey {
    /// Create from anything convertible to `String`.
    pub fn new(state: impl Into<String>, event: impl Into<String>) -> Self {
        Self {
            state: state.into(),
            event: event.into(),
        }
    }

    /// Build a key from typed `State` and `Event` using their `Display` impls.
    ///
    /// This is the preferred constructor in generic code.
    /// Uses `Display`, so keep `Display` stable for canonical key correctness.
    pub fn key_for<S: State, E: Event>(state: &S, event: &E) -> Self {
        Self {
            state: state.to_string(),
            event: event.to_string(),
        }
    }

    /// Canonical string `state::event` useful for FS paths, logging, metrics.
    pub fn canonical(&self) -> String {
        format!("{}::{}", self.state, self.event)
    }
}

impl fmt::Display for PromptKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.canonical())
    }
}

impl From<(String, String)> for PromptKey {
    fn from((s, e): (String, String)) -> Self {
        Self::new(s, e)
    }
}

impl From<(&str, &str)> for PromptKey {
    fn from((s, e): (&str, &str)) -> Self {
        Self::new(s, e)
    }
}

// ── ToolId ──────────────────────────────────────────────────────────────

/// Newtype for tool identifiers, e.g. `"write_code"`, `"search_web"`.
///
/// Thin wrapper around `String` to give type safety in policy APIs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub struct ToolId(pub String);

impl fmt::Display for ToolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ToolId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for ToolId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl AsRef<str> for ToolId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ── Errors ──────────────────────────────────────────────────────────────

/// Core error type for forme — zero I/O, just data.
#[derive(Debug, Error)]
pub enum FormeError {
    #[error("prompt not found for {0}")]
    RegistryNotFound(PromptKey),

    #[error("context build failed: {0}")]
    ContextBuildFailed(String),

    #[error("policy denied tool {0}")]
    PolicyDenied(ToolId),

    #[error("persistence failed: {0}")]
    PersistenceFailed(String),

    #[error("llm call failed: {0}")]
    LlmFailed(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

impl FormeError {
    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::RegistryNotFound(_))
    }

    pub fn is_denied(&self) -> bool {
        matches!(self, Self::PolicyDenied(_))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};

    // Minimal State / Event conforming to blanket impls for generic tests
    #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
    struct S(String);
    impl fmt::Display for S {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    #[derive(Clone, Debug, Serialize, serde::Deserialize)]
    struct E(String);
    impl fmt::Display for E {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    // Extra state/event for key_for generic test
    #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
    enum MyState {
        Idle,
        Active,
    }
    impl fmt::Display for MyState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Idle => write!(f, "Idle"),
                Self::Active => write!(f, "Active"),
            }
        }
    }

    #[derive(Clone, Debug, Serialize, serde::Deserialize)]
    enum MyEvent {
        UserMsg,
        Save,
    }
    impl fmt::Display for MyEvent {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::UserMsg => write!(f, "UserMsg"),
                Self::Save => write!(f, "Save"),
            }
        }
    }

    // 1. prompt_key_equality
    #[test]
    fn prompt_key_equality() {
        let a = PromptKey::new("Idle", "UserMsg");
        let b = PromptKey::new("Idle", "UserMsg");
        assert_eq!(a, b);
        // From tuple
        let d: PromptKey = ("Idle".to_string(), "UserMsg".to_string()).into();
        assert_eq!(a, d);
        let e: PromptKey = ("Idle", "UserMsg").into();
        assert_eq!(a, e);
    }

    // 2. prompt_key_inequality_state
    #[test]
    fn prompt_key_inequality_state() {
        let a = PromptKey::new("Idle", "UserMsg");
        let b = PromptKey::new("Active", "UserMsg");
        assert_ne!(a, b);
        let c = PromptKey::new("Idle", "Other");
        assert_ne!(a, c);
    }

    // 3. prompt_key_display_canonical
    #[test]
    fn prompt_key_display_canonical() {
        let k = PromptKey::new("Idle", "UserMsg");
        assert_eq!(k.canonical(), "Idle::UserMsg");
        assert_eq!(k.to_string(), "Idle::UserMsg");
        assert_eq!(k.to_string(), k.canonical());
        // Display stable
        assert_eq!(format!("{}", k), "Idle::UserMsg");
        // Debug differs from Display (debug contains struct name)
        let dbg = format!("{:?}", k);
        assert!(dbg.contains("PromptKey") || dbg.contains("Idle"));
        assert_ne!(dbg, k.to_string());
    }

    // 4. prompt_key_hash_eq
    #[test]
    fn prompt_key_hash_eq() {
        let a = PromptKey::new("Idle", "A");
        let b = PromptKey::new("Idle", "A");
        assert_eq!(a, b);

        let mut hasher_a = DefaultHasher::new();
        a.hash(&mut hasher_a);
        let hash_a = hasher_a.finish();

        let mut hasher_b = DefaultHasher::new();
        b.hash(&mut hasher_b);
        let hash_b = hasher_b.finish();

        assert_eq!(hash_a, hash_b);
    }

    // 5. prompt_key_hashset_lookup
    #[test]
    fn prompt_key_hashset_lookup() {
        let mut set = HashSet::new();
        let k1 = PromptKey::new("Idle", "A");
        let k2 = PromptKey::new("Idle", "A");
        set.insert(k1.clone());
        assert!(set.contains(&k2));
        assert_eq!(set.len(), 1);
        set.insert(PromptKey::new("Idle", "B"));
        assert_eq!(set.len(), 2);
    }

    // 6. key_for_generic
    #[test]
    fn key_for_generic() {
        let state = MyState::Idle;
        let event = MyEvent::UserMsg;
        let k = PromptKey::key_for(&state, &event);
        assert_eq!(k.state, "Idle");
        assert_eq!(k.event, "UserMsg");
        assert_eq!(k.canonical(), "Idle::UserMsg");

        // Also with S(String) wrapper
        let s = S("Done".into());
        let e = E("Save".into());
        let k2 = PromptKey::key_for(&s, &e);
        assert_eq!(k2, PromptKey::new("Done", "Save"));
    }

    // 7. error_display_registry_not_found (contains key)
    #[test]
    fn error_display_registry_not_found() {
        let key = PromptKey::new("Idle", "MissingEvent");
        let err = FormeError::RegistryNotFound(key.clone());
        let msg = err.to_string();
        // Should contain canonical key per spec #[error("prompt not found for {0}")]
        assert!(msg.contains("prompt not found"), "msg was: {msg}");
        assert!(
            msg.contains(&key.to_string()),
            "msg {msg} should contain key {key}"
        );
        assert!(err.is_not_found());
        assert!(!err.is_denied());
    }

    // 8. error_display_all_variants (each variant to_string contains payload)
    #[test]
    fn error_display_all_variants() {
        let variants: Vec<(FormeError, &str)> = vec![
            (
                FormeError::RegistryNotFound(PromptKey::new("S", "E")),
                "S::E",
            ),
            (
                FormeError::ContextBuildFailed("ctx boom".into()),
                "ctx boom",
            ),
            (
                FormeError::PolicyDenied(ToolId::from("write_code")),
                "write_code",
            ),
            (
                FormeError::PersistenceFailed("disk full".into()),
                "disk full",
            ),
            (FormeError::LlmFailed("timeout".into()), "timeout"),
            (FormeError::InvalidInput("bad json".into()), "bad json"),
        ];

        for (err, payload) in variants {
            let msg = err.to_string();
            assert!(
                msg.contains(payload),
                "variant {err:?} message '{msg}' should contain '{payload}'"
            );
        }

        // PolicyDenied specifics
        let denied = FormeError::PolicyDenied(ToolId::from("search"));
        assert!(denied.is_denied());
        assert!(!denied.is_not_found());
    }

    // 9. trait_bounds_compile_check (assert generic compiles)
    #[test]
    fn trait_bounds_compile_check() {
        fn assert_state<T: State>() {}
        fn assert_event<T: Event>() {}

        assert_state::<S>();
        assert_state::<MyState>();
        assert_event::<E>();
        assert_event::<MyEvent>();

        // Ensure blanket impl works for common types that meet bounds
        assert_state::<String>(); // String is Clone+Debug+Display+Eq+Hash+Serialize+DeserializeOwned+Send+Sync
                                  // i32 also implements all Event bounds? Check: i32 is Display, Clone, Debug, Serialize, DeserializeOwned, Send+Sync
        fn assert_event_i32<T: Event>() {}
        assert_event_i32::<i32>();
    }

    // 10. serde_roundtrip_prompt_key (json to/from preserves equality)
    #[test]
    fn serde_roundtrip_prompt_key() {
        let original = PromptKey::new("Idle", "UserMsg");
        let json = serde_json::to_string(&original).expect("serialize");
        let decoded: PromptKey = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, decoded);

        // canonical preserved
        assert_eq!(original.canonical(), decoded.canonical());
    }

    // Extra ToolId tests
    #[test]
    fn tool_id_basic() {
        let a = ToolId::from("write_code");
        let b = ToolId("write_code".to_string());
        assert_eq!(a, b);
        assert_eq!(a.to_string(), "write_code");
        assert_eq!(a.as_ref() as &str, "write_code");

        let c = ToolId::from(String::from("search"));
        assert_eq!(c.to_string(), "search");
        assert_ne!(a, c);
    }

    #[test]
    fn tool_id_hashset_and_serde() {
        let mut set = HashSet::new();
        set.insert(ToolId::from("tool_a"));
        assert!(set.contains(&ToolId::from("tool_a")));
        assert!(!set.contains(&ToolId::from("tool_b")));

        let original = ToolId::from("my_tool");
        let json = serde_json::to_string(&original).unwrap();
        let decoded: ToolId = serde_json::from_str(&json).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn empty_keys_allowed() {
        // Edge case: empty strings are allowed but documented as discouraged
        let k = PromptKey::new("", "");
        assert_eq!(k.canonical(), "::");
        assert_eq!(k.to_string(), "::");
        // Still usable as map key
        let mut set = HashSet::new();
        set.insert(k.clone());
        assert!(set.contains(&k));
    }

    #[test]
    fn from_tuple_impls() {
        let k1: PromptKey = (String::from("A"), String::from("B")).into();
        assert_eq!(k1.state, "A");
        assert_eq!(k1.event, "B");

        let k2: PromptKey = ("X", "Y").into();
        assert_eq!(k2.state, "X");
        assert_eq!(k2.event, "Y");
    }
}
