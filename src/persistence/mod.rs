//! persistence — Wave 1: `Checkpointer` trait, single state vector.
//!
//! Scope: trait only. No `InMemoryCheckpointer` (Wave 2), no Postgres (Wave 3).
//! The simplest model — one slot storing the latest `S` — covers single-agent
//! execution. Future extension: keyed persistence for multi-workflow /
//! multi-tenant state (e.g. `save_for(key, state)` / `load_for(key)`).

use serde::{de::DeserializeOwned, Serialize};

use crate::core::{FormeError, State};

/// Durable checkpoint for the current machine `State`.
///
/// Implementations persist a **single** state value — conceptually `Option<S>`.
/// `save` overwrites any previous checkpoint. `load` returns `None` when no
/// checkpoint exists.
///
/// # Future extension
/// A keyed variant (`save_for`/`load_for`) for multi-workflow state may be
/// added later without breaking this trait — either as a new trait or with
/// defaulted methods. For now the single-slot model keeps Wave 1 minimal.
///
/// # Bounds
/// `S: State + Serialize + DeserializeOwned` is required so serde round-trips
/// are possible; `State` itself already implies `Serialize + DeserializeOwned`,
/// but we repeat the bound explicitly for clarity and forward-compat.
pub trait Checkpointer<S>: Send + Sync
where
    S: State + Serialize + DeserializeOwned,
{
    /// Persist `state`, overwriting any prior value.
    ///
    /// Maps I/O or serialization failures to `FormeError::PersistenceFailed`.
    fn save(&self, state: &S) -> Result<(), FormeError>;

    /// Load the last persisted state, if any.
    ///
    /// `Ok(None)` means no checkpoint yet. Errors map to `PersistenceFailed`.
    fn load(&self) -> Result<Option<S>, FormeError>;
}

// ── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;
    use std::sync::Mutex;

    use serde::{de::DeserializeOwned, Deserialize, Serialize};

    // Test state — blanket impl gives us `State`
    #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    enum TestState {
        Idle,
        Active(String),
    }

    impl fmt::Display for TestState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Idle => write!(f, "Idle"),
                Self::Active(s) => write!(f, "Active({})", s),
            }
        }
    }

    // In-memory mock using Mutex<Option<S>> — allowed per spec (inside tests)
    struct MockCheckpointer<S>
    where
        S: State + Serialize + DeserializeOwned,
    {
        inner: Mutex<Option<S>>,
    }

    impl<S> MockCheckpointer<S>
    where
        S: State + Serialize + DeserializeOwned,
    {
        fn new() -> Self {
            let inner: Mutex<Option<S>> = Mutex::new(None);
            Self { inner }
        }
    }

    impl<S> Checkpointer<S> for MockCheckpointer<S>
    where
        S: State + Serialize + DeserializeOwned + Clone,
    {
        fn save(&self, state: &S) -> Result<(), FormeError> {
            let cloned: S = state.clone();
            let mut guard: std::sync::MutexGuard<'_, Option<S>> =
                self.inner.lock().map_err(|e: std::sync::PoisonError<_>| {
                    FormeError::PersistenceFailed(format!("poisoned lock: {}", e))
                })?;
            *guard = Some(cloned);
            Ok(())
        }

        fn load(&self) -> Result<Option<S>, FormeError> {
            let guard: std::sync::MutexGuard<'_, Option<S>> =
                self.inner.lock().map_err(|e: std::sync::PoisonError<_>| {
                    FormeError::PersistenceFailed(format!("poisoned lock: {}", e))
                })?;
            let opt: Option<S> = guard.clone();
            Ok(opt)
        }
    }

    struct FailingCheckpointer;

    impl Checkpointer<TestState> for FailingCheckpointer {
        fn save(&self, _state: &TestState) -> Result<(), FormeError> {
            Err(FormeError::PersistenceFailed("disk full".into()))
        }

        fn load(&self) -> Result<Option<TestState>, FormeError> {
            Err(FormeError::PersistenceFailed("read failed".into()))
        }
    }

    // 1. trait mock roundtrip
    #[test]
    fn roundtrip_mock() {
        let cp: MockCheckpointer<TestState> = MockCheckpointer::new();
        let s: TestState = TestState::Active("hello".into());
        cp.save(&s).expect("save ok");
        let loaded: Option<TestState> = cp.load().expect("load ok");
        assert_eq!(loaded, Some(s));
    }

    // 2. load empty returns None
    #[test]
    fn load_empty_returns_none() {
        let cp: MockCheckpointer<TestState> = MockCheckpointer::new();
        let loaded: Option<TestState> = cp.load().expect("load empty ok");
        assert!(loaded.is_none(), "expected None, got {:?}", loaded);
    }

    // 3. save overwrites previous
    #[test]
    fn save_overwrites_previous() {
        let cp: MockCheckpointer<TestState> = MockCheckpointer::new();
        let first: TestState = TestState::Idle;
        let second: TestState = TestState::Active("second".into());
        cp.save(&first).expect("first save");
        let v1: Option<TestState> = cp.load().expect("load1");
        assert_eq!(v1, Some(first.clone()));
        cp.save(&second).expect("second save");
        let v2: Option<TestState> = cp.load().expect("load2");
        assert_eq!(v2, Some(second));
        assert_ne!(v1, v2);
    }

    // 4. error propagation with PersistenceFailed
    #[test]
    fn error_propagation_persistence_failed() {
        let cp: FailingCheckpointer = FailingCheckpointer;
        let state: TestState = TestState::Idle;

        let save_res: Result<(), FormeError> = cp.save(&state);
        assert!(save_res.is_err(), "save should fail");
        let err: FormeError = save_res.unwrap_err();
        assert!(
            err.to_string().contains("disk full"),
            "save err should contain payload, got: {}",
            err
        );
        match err {
            FormeError::PersistenceFailed(_) => {}
            _ => panic!("expected PersistenceFailed, got {:?}", err),
        }

        let load_res: Result<Option<TestState>, FormeError> = cp.load();
        assert!(load_res.is_err(), "load should fail");
        let err2: FormeError = load_res.unwrap_err();
        assert!(
            err2.to_string().contains("read failed"),
            "load err msg: {}",
            err2
        );
        match err2 {
            FormeError::PersistenceFailed(_) => {}
            _ => panic!("expected PersistenceFailed, got {:?}", err2),
        }
    }

    // 5. Send+Sync compile check
    #[test]
    fn send_sync_compile_check() {
        fn assert_send_sync<T: Send + Sync>() {}
        fn assert_checkpointer_send_sync<S, C>()
        where
            S: State + Serialize + DeserializeOwned,
            C: Checkpointer<S> + Send + Sync,
        {
        }

        // Concrete type is Send+Sync (Mutex is Send+Sync when T is Send)
        assert_send_sync::<MockCheckpointer<TestState>>();
        assert_send_sync::<FailingCheckpointer>();

        // Trait object is Send+Sync when implementations are
        assert_send_sync::<Box<dyn Checkpointer<TestState>>>();

        // Generic helper compiles
        assert_checkpointer_send_sync::<TestState, MockCheckpointer<TestState>>();
        assert_checkpointer_send_sync::<TestState, FailingCheckpointer>();
    }

    // 6. extra: serde bounds exercise (ensures S really is serializable)
    #[test]
    fn serde_bounds_passthrough() {
        let cp: MockCheckpointer<TestState> = MockCheckpointer::new();
        let s: TestState = TestState::Active("serde".into());
        // Ensure we can serialize the state itself — bound exercise
        let json: String = serde_json::to_string(&s).expect("serialize state");
        let de: TestState = serde_json::from_str(&json).expect("deserialize state");
        cp.save(&de).expect("save after serde roundtrip");
        let loaded: Option<TestState> = cp.load().expect("load");
        assert_eq!(loaded, Some(TestState::Active("serde".into())));
    }
}
