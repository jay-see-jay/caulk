//! persistence — `Checkpointer` trait + `InMemoryCheckpointer`
//!
//! Wave 1: trait single state vector. Wave 2: `InMemoryCheckpointer` with
//! `Arc<RwLock<Option<S>>>` shared storage, Clone sharing, round-trip.
//!
//! Why `Arc<RwLock>` not `Mutex`?
//! - `RwLock` allows many concurrent readers (load-heavy workloads) while
//!   still serializing writers. In tests and single-agent runtime the
//!   contention is tiny, but `RwLock` models the real usage better than
//!   `Mutex` and stays `Send + Sync` when `S: Send + Sync` (which `State`
//!   guarantees).
//! - `Arc` gives shared ownership: cloning the checkpointer clones the Arc,
//!   not the state. Two runner instances sharing a checkpointer see the same
//!   latest checkpoint — useful for replay tests and for swapping runners
//!   without losing state. This mirrors real persistence where the store is
//!   external.
//!
//! Future extension: keyed persistence `save_for(key, state)` for multi-tenant
//! agents. For now single-slot keeps Wave 2 minimal and deterministic.

use std::sync::{Arc, RwLock};

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

// ── InMemoryCheckpointer ────────────────────────────────────────────────

/// In-memory single-slot checkpointer.
///
/// Storage: `Arc<RwLock<Option<S>>>`. `Clone` clones the Arc (shared storage),
/// so `let b = a.clone()` sees the same latest state — intentional, mimics
/// external stores.
///
/// Thread-safe, `Send + Sync` when `S: Send + Sync` (which `State` ensures).
/// No I/O, no serialization side-effects in this impl; errors only arise from
/// poisoned locks, mapped to `FormeError::PersistenceFailed`.
///
/// Generic reuse: product states (`Idle`, `Teaching`, `MyState`) all satisfy
/// `State` via blanket impl. One `InMemoryCheckpointer<S>` works for any `S`,
/// no per-product branching needed.
///
/// Example:
/// ```
/// use caulk::persistence::{Checkpointer, InMemoryCheckpointer};
/// use serde::{Deserialize, Serialize};
/// use std::fmt;
///
/// #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// enum S { Idle, Done }
/// impl fmt::Display for S { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self) } }
///
/// let cp = InMemoryCheckpointer::<S>::new();
/// assert!(cp.load().unwrap().is_none());
/// cp.save(&S::Idle).unwrap();
/// assert_eq!(cp.load().unwrap(), Some(S::Idle));
/// ```
#[derive(Debug)]
pub struct InMemoryCheckpointer<S>
where
    S: State,
{
    inner: Arc<RwLock<Option<S>>>,
}

impl<S> InMemoryCheckpointer<S>
where
    S: State,
{
    /// Empty checkpoint (`load` → `None`).
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
        }
    }

    /// Pre-seeded checkpoint (`load` → `Some(state)`).
    pub fn with_state(state: S) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Some(state))),
        }
    }

    /// Raw inner for advanced tests (not part of public contract).
    #[cfg(test)]
    pub(crate) fn inner_ptr(&self) -> *const RwLock<Option<S>> {
        Arc::as_ptr(&self.inner) as *const _
    }
}

impl<S> Default for InMemoryCheckpointer<S>
where
    S: State,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Clone for InMemoryCheckpointer<S>
where
    S: State,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<S> Checkpointer<S> for InMemoryCheckpointer<S>
where
    S: State + Serialize + DeserializeOwned + Clone,
{
    fn save(&self, state: &S) -> Result<(), FormeError> {
        let mut guard = self
            .inner
            .write()
            .map_err(|e| FormeError::PersistenceFailed(format!("RwLock poisoned on write: {e}")))?;
        *guard = Some(state.clone());
        Ok(())
    }

    fn load(&self) -> Result<Option<S>, FormeError> {
        let guard = self
            .inner
            .read()
            .map_err(|e| FormeError::PersistenceFailed(format!("RwLock poisoned on read: {e}")))?;
        Ok(guard.clone())
    }
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

    // ── InMemoryCheckpointer (Wave 2) tests ───────────────────────────

    #[test]
    fn in_memory_roundtrip() {
        let cp = InMemoryCheckpointer::<TestState>::new();
        assert!(cp.load().unwrap().is_none());
        cp.save(&TestState::Idle).unwrap();
        assert_eq!(cp.load().unwrap(), Some(TestState::Idle));

        cp.save(&TestState::Active("a".into())).unwrap();
        assert_eq!(cp.load().unwrap(), Some(TestState::Active("a".into())));
    }

    #[test]
    fn in_memory_empty_none() {
        let cp = InMemoryCheckpointer::<TestState>::default();
        let loaded = cp.load().expect("load empty");
        assert!(loaded.is_none());

        // with_state seeds
        let seeded = InMemoryCheckpointer::with_state(TestState::Idle);
        assert_eq!(seeded.load().unwrap(), Some(TestState::Idle));
    }

    #[test]
    fn in_memory_overwrite() {
        let cp = InMemoryCheckpointer::<TestState>::new();
        cp.save(&TestState::Idle).unwrap();
        assert_eq!(cp.load().unwrap(), Some(TestState::Idle));

        cp.save(&TestState::Active("second".into())).unwrap();
        let v2 = cp.load().unwrap();
        assert_eq!(v2, Some(TestState::Active("second".into())));
    }

    #[test]
    fn in_memory_clone_shares_storage() {
        let cp = InMemoryCheckpointer::<TestState>::new();
        cp.save(&TestState::Idle).unwrap();

        let cp2 = cp.clone();
        // cp2 sees same storage
        assert_eq!(cp2.load().unwrap(), Some(TestState::Idle));

        // write via cp2 seen by cp
        cp2.save(&TestState::Active("via clone".into())).unwrap();
        assert_eq!(
            cp.load().unwrap(),
            Some(TestState::Active("via clone".into()))
        );

        // pointer equality check (Arc shared)
        assert_eq!(cp.inner_ptr(), cp2.inner_ptr());
    }

    #[test]
    fn in_memory_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InMemoryCheckpointer<TestState>>();
        // as Checkpointer trait object
        fn assert_cp<S: State + Serialize + DeserializeOwned, C: Checkpointer<S> + Send + Sync>() {}
        assert_cp::<TestState, InMemoryCheckpointer<TestState>>();

        // threading exercise
        let cp = InMemoryCheckpointer::<TestState>::new();
        cp.save(&TestState::Idle).unwrap();
        let cp_clone = cp.clone();
        let handle = std::thread::spawn(move || cp_clone.load().unwrap());
        let loaded = handle.join().unwrap();
        assert_eq!(loaded, Some(TestState::Idle));
    }

    #[test]
    fn in_memory_debug_and_default() {
        let cp = InMemoryCheckpointer::<TestState>::default();
        let dbg = format!("{:?}", cp);
        assert!(dbg.contains("InMemoryCheckpointer"));

        let cp2 = InMemoryCheckpointer::<TestState>::new();
        assert!(cp2.load().unwrap().is_none());
    }
}
