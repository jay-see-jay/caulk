# prompt_registry

Wave 1 module, depends only on `core` (`PromptKey`, `FormeError`).

Scope:
- `trait Registry { fn get(&self, key: &PromptKey) -> Result<String, FormeError> }` Send+Sync
- `FsRegistry { root: PathBuf, cache: RwLock<HashMap<PromptKey,(String,SystemTime)>> }` loads `root/<state>/<event>.md`
- decision: hierarchical `state/event.md` not `state::event.md` — filesystem friendly, avoids `::` escaping.
- Mapping: PromptKey {state, event} -> Path = root.join(state).join(format!("{event}.md"))
- `InMemoryRegistry(HashMap<PromptKey,String>)`
- cached lookup with mtime check, returns cloned String, never holds file handle across await.
- Error: `RegistryNotFound(key)` includes attempted path in Display via FormeError.

Tests (per Senior IC):
- FsRegistry happy reads file
- InMemory happy returns
- missing -> Err NotFound with helpful path containing key
- mtime cache invalidation after touch
- key canon "Idle::UserMsg" maps to path "Idle/UserMsg.md"
- whitespace trimming? decision: raw content returned as-is, no trim, caller trims.

Generic only, no teaching prompts.

