//! prompt_registry — Wave 1
//!
//! Scope: `trait Registry`, `FsRegistry`, `InMemoryRegistry`.
//!
//! Mapping: `PromptKey { state, event }` → `root.join(state).join(format!("{event}.md"))`
//! Decision: hierarchical `state/event.md` not `state::event.md` — filesystem friendly,
//! avoids `::` escaping.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::{FormeError, PromptKey};

/// Pluggable prompt source keyed by `PromptKey`.
///
/// Object-safe, `Send + Sync` so it can be shared across threads / tasks.
pub trait Registry: Send + Sync {
    /// Resolve `key` to prompt text.
    ///
    /// Returns `FormeError::RegistryNotFound(key)` when missing.
    fn get(&self, key: &PromptKey) -> Result<String, FormeError>;
}

/// Filesystem registry loading `root/<state>/<event>.md`.
///
/// Cached with mtime check. Never holds a file handle across an await
/// (this implementation is synchronous, but still drops handles promptly).
///
/// Cache stores `(content, mtime)` per key and returns a cloned `String`.
pub struct FsRegistry {
    root: PathBuf,
    cache: RwLock<HashMap<PromptKey, (String, SystemTime)>>,
}

impl FsRegistry {
    /// Create a new registry rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Resolve `PromptKey` → filesystem path.
    ///
    /// Public for testing the canonical mapping:
    /// `"Idle::UserMsg"` → `root/Idle/UserMsg.md`
    pub fn path_for_key(&self, key: &PromptKey) -> PathBuf {
        self.root.join(&key.state).join(format!("{}.md", key.event))
    }

    /// Root directory this registry reads from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Clear the entire in-memory cache.
    ///
    /// Useful for tests or manual hot-reload invalidation.
    pub fn clear_cache(&self) {
        if let Ok(mut write) = self.cache.write() {
            write.clear();
        }
    }

    fn resolve_path(&self, key: &PromptKey) -> PathBuf {
        self.path_for_key(key)
    }
}

impl Registry for FsRegistry {
    fn get(&self, key: &PromptKey) -> Result<String, FormeError> {
        let path: PathBuf = self.resolve_path(key);

        let metadata: fs::Metadata = match fs::metadata(&path) {
            Ok(m) => m,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(FormeError::RegistryNotFound(key.clone()));
            }
            Err(e) => {
                return Err(FormeError::PersistenceFailed(format!(
                    "failed to stat {}: {}",
                    path.display(),
                    e
                )));
            }
        };

        let mtime: SystemTime = metadata.modified().unwrap_or(UNIX_EPOCH);

        if let Ok(read_guard) = self.cache.read() {
            if let Some((cached_content, cached_mtime)) = read_guard.get(key) {
                if *cached_mtime == mtime {
                    return Ok(cached_content.clone());
                }
            }
        }

        let content: String = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(FormeError::RegistryNotFound(key.clone()));
            }
            Err(e) => {
                return Err(FormeError::PersistenceFailed(format!(
                    "failed to read {}: {}",
                    path.display(),
                    e
                )));
            }
        };

        if let Ok(mut write_guard) = self.cache.write() {
            write_guard.insert(key.clone(), (content.clone(), mtime));
        }

        Ok(content)
    }
}

/// In-memory registry backed by a `HashMap`.
///
/// Ideal for tests, examples, and hot pre-loaded prompts.
pub struct InMemoryRegistry(pub HashMap<PromptKey, String>);

impl InMemoryRegistry {
    /// Create from an existing map.
    pub fn new(map: HashMap<PromptKey, String>) -> Self {
        Self(map)
    }

    /// Alias of `new`.
    pub fn from_map(map: HashMap<PromptKey, String>) -> Self {
        Self::new(map)
    }

    /// Convenience: build from an iterator of `(key, prompt)`.
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (PromptKey, String)>,
    {
        Self(iter.into_iter().collect())
    }
}

impl From<HashMap<PromptKey, String>> for InMemoryRegistry {
    fn from(map: HashMap<PromptKey, String>) -> Self {
        Self::new(map)
    }
}

impl Registry for InMemoryRegistry {
    fn get(&self, key: &PromptKey) -> Result<String, FormeError> {
        match self.0.get(key) {
            Some(s) => Ok(s.clone()),
            None => Err(FormeError::RegistryNotFound(key.clone())),
        }
    }
}

// ── Tests (Senior IC plan, 6+) ──────────────────────────────────────────
#[allow(clippy::should_implement_trait)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(suffix: &str) -> PathBuf {
        let base: PathBuf = std::env::temp_dir();
        let nanos: u128 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pid: u32 = std::process::id();
        let name: String = format!("caulk_registry_{}_{}_{}", pid, nanos, suffix);
        let path: PathBuf = base.join(name);
        let _ = fs::create_dir_all(&path);
        path
    }

    fn cleanup_dir(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    // 1. FsRegistry happy reads file
    #[test]
    fn fs_registry_happy_reads_file() {
        let root: PathBuf = unique_temp_dir("happy");
        let state_dir: PathBuf = root.join("Idle");
        fs::create_dir_all(&state_dir).unwrap();
        let file_path: PathBuf = state_dir.join("UserMsg.md");
        let expected: &str = "You are in Idle, responding to UserMsg.";
        fs::write(&file_path, expected).unwrap();

        let registry: FsRegistry = FsRegistry::new(&root);
        let key: PromptKey = PromptKey::new("Idle", "UserMsg");
        let got: String = registry.get(&key).expect("should read");
        assert_eq!(got, expected);

        cleanup_dir(&root);
    }

    // 2. InMemoryRegistry happy returns
    #[test]
    fn in_memory_happy_returns() {
        let mut map: HashMap<PromptKey, String> = HashMap::new();
        let key: PromptKey = PromptKey::new("Active", "Save");
        map.insert(key.clone(), "active-save-prompt".into());
        let registry: InMemoryRegistry = InMemoryRegistry::new(map);

        let got: String = registry.get(&key).expect("in-memory should hit");
        assert_eq!(got, "active-save-prompt");

        // Also via Registry trait object: Send+Sync check
        let boxed: Box<dyn Registry> = Box::new(InMemoryRegistry::from_map(HashMap::from([(
            PromptKey::new("Idle", "UserMsg"),
            "hello".into(),
        )])));
        let k2: PromptKey = PromptKey::new("Idle", "UserMsg");
        assert_eq!(boxed.get(&k2).unwrap(), "hello");
    }

    // 3. missing -> Err NotFound with helpful path containing key
    #[test]
    fn missing_returns_not_found_with_key_substring() {
        let root: PathBuf = unique_temp_dir("missing");
        fs::create_dir_all(&root).unwrap();
        let registry: FsRegistry = FsRegistry::new(&root);
        let key: PromptKey = PromptKey::new("Idle", "MissingEvent");
        let err: FormeError = registry.get(&key).unwrap_err();
        assert!(err.is_not_found(), "should be not_found");
        let msg: String = err.to_string();
        // Display is "prompt not found for {key}" where key = "Idle::MissingEvent"
        assert!(
            msg.contains("Idle::MissingEvent") || msg.contains("Idle"),
            "msg '{}' should contain canonical key",
            msg
        );
        assert!(
            msg.contains("MissingEvent"),
            "msg '{}' should contain event part",
            msg
        );
        // also contains "prompt not found"
        assert!(msg.contains("prompt not found"), "msg was: {}", msg);

        // InMemory missing also helpful
        let empty: InMemoryRegistry = InMemoryRegistry::new(HashMap::new());
        let err2: FormeError = empty.get(&key).unwrap_err();
        assert!(err2.is_not_found());
        assert!(err2.to_string().contains("Idle::MissingEvent"));

        cleanup_dir(&root);
    }

    // 4. mtime cache invalidation after touch
    #[test]
    fn mtime_cache_invalidation_after_touch() {
        let root: PathBuf = unique_temp_dir("mtime");
        let state_dir: PathBuf = root.join("Idle");
        fs::create_dir_all(&state_dir).unwrap();
        let file_path: PathBuf = state_dir.join("UserMsg.md");
        fs::write(&file_path, "v1").unwrap();

        let registry: FsRegistry = FsRegistry::new(&root);
        let key: PromptKey = PromptKey::new("Idle", "UserMsg");

        let first: String = registry.get(&key).expect("first read");
        assert_eq!(first, "v1");

        // Second read should hit cache, still v1
        let cached: String = registry.get(&key).expect("cached read");
        assert_eq!(cached, "v1");

        // Touch file with new content, ensuring mtime changes (sleep >1s for sec-granularity FS)
        std::thread::sleep(std::time::Duration::from_millis(1100));
        fs::write(&file_path, "v2").unwrap();

        let second: String = registry.get(&key).expect("after touch");
        assert_eq!(
            second, "v2",
            "cache should have invalidated via mtime, got '{}' expected 'v2'",
            second
        );

        cleanup_dir(&root);
    }

    // 5. key canonical "Idle::UserMsg" maps to path "Idle/UserMsg.md"
    #[test]
    fn key_canonical_maps_to_path() {
        let root: PathBuf = PathBuf::from("/tmp/root_example");
        let registry: FsRegistry = FsRegistry::new(&root);
        let key: PromptKey = PromptKey::new("Idle", "UserMsg");
        assert_eq!(key.canonical(), "Idle::UserMsg");
        assert_eq!(key.to_string(), "Idle::UserMsg");

        let path: PathBuf = registry.path_for_key(&key);
        // Expect root/Idle/UserMsg.md
        let expected_suffix: PathBuf = PathBuf::from("Idle").join("UserMsg.md");
        assert!(
            path.ends_with(expected_suffix),
            "path {:?} should end with Idle/UserMsg.md",
            path
        );
        // Also ensure it starts with root
        assert!(path.starts_with(&root));

        // Direct mapping check without relying on FS existence
        let path2: PathBuf = root.join("Idle").join("UserMsg.md");
        assert_eq!(path, path2);

        // Ensure mapping matches spec: root.join(state).join(format!("{event}.md"))
        let manual: PathBuf = root
            .join(key.state.clone())
            .join(format!("{}.md", key.event));
        assert_eq!(path, manual);
    }

    // 6. whitespace raw returned as-is (no trim)
    #[test]
    fn whitespace_raw_returned_as_is() {
        let root: PathBuf = unique_temp_dir("whitespace");
        let state_dir: PathBuf = root.join("Idle");
        fs::create_dir_all(&state_dir).unwrap();
        let content: &str = "  hello world  \n\n\t\n";
        fs::write(state_dir.join("UserMsg.md"), content).unwrap();

        let registry: FsRegistry = FsRegistry::new(&root);
        let key: PromptKey = PromptKey::new("Idle", "UserMsg");
        let got: String = registry.get(&key).unwrap();
        assert_eq!(
            got, content,
            "FsRegistry should return raw content without trimming"
        );
        assert!(got.starts_with("  "), "leading spaces preserved");
        assert!(got.ends_with("\n"), "trailing newline preserved");

        // InMemory also preserves whitespace
        let mut map: HashMap<PromptKey, String> = HashMap::new();
        map.insert(key.clone(), content.to_string());
        let mem: InMemoryRegistry = InMemoryRegistry::from(map);
        let got2: String = mem.get(&key).unwrap();
        assert_eq!(got2, content);

        cleanup_dir(&root);
    }

    // 7. extra: manual clear_cache invalidates (alternative to mtime)
    #[test]
    fn manual_clear_cache_invalidates() {
        let root: PathBuf = unique_temp_dir("clear_cache");
        let state_dir: PathBuf = root.join("StateA");
        fs::create_dir_all(&state_dir).unwrap();
        let file_path: PathBuf = state_dir.join("Ev.md");
        fs::write(&file_path, "first").unwrap();

        let registry: FsRegistry = FsRegistry::new(&root);
        let key: PromptKey = PromptKey::new("StateA", "Ev");
        let v1: String = registry.get(&key).unwrap();
        assert_eq!(v1, "first");

        // Overwrite file but keep mtime artificially same-ish by not sleeping,
        // then clear cache manually — should read new content.
        // Even if mtime didn't change (if FS has 1s granularity and we didn't sleep),
        // clear_cache forces re-read. We simulate by using clear_cache after write.
        // First, overwrite file quickly:
        fs::write(&file_path, "second").unwrap();
        // Clear cache:
        registry.clear_cache();

        let v2: String = registry.get(&key).unwrap();
        // Could be "first" if mtime didn't change and cache wasn't cleared, but we cleared.
        // After clear, it must be "second" because file content changed.
        // If FS mtime granularity caused second == first file mtime? No, we cleared anyway,
        // so we still read file anew regardless of mtime.
        assert_eq!(v2, "second", "after clear_cache, should re-read file");

        cleanup_dir(&root);
    }

    // 8. extra: returns cloned String, not ref
    #[test]
    fn returns_cloned_string() {
        let mut map: HashMap<PromptKey, String> = HashMap::new();
        let key: PromptKey = PromptKey::new("S", "E");
        map.insert(key.clone(), "original".into());
        let reg: InMemoryRegistry = InMemoryRegistry::new(map);
        let mut got: String = reg.get(&key).unwrap();
        got.push_str(" mutated");
        // Original in registry unchanged
        let got2: String = reg.get(&key).unwrap();
        assert_eq!(got2, "original");
    }
}
