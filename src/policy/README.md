# policy

Wave 1 module. Depends on `core` (`State`, `ToolId`, `FormeError`).

Generic only, NO teacher logic.

Exports:
- `trait Policy<S: State> { fn allowed(&self, state: &S) -> &[ToolId]; fn check(&self, state: &S, tool: &ToolId) -> Result<(), FormeError> }` default check = if allowed contains tool -> Ok else Err PolicyDenied
- `struct AllowAllPolicy` — check always Ok, allowed empty (means “all”)
- `struct DenyListPolicy(HashSet<ToolId>)` — denies set, allows rest
- `struct AllowListPolicy(HashSet<ToolId>)` — allows only set

Why: FormeError::PolicyDenied includes tool + state display.

Tests (6):
- AllowAll allows any
- DenyList denies listed
- DenyList message contains tool name
- AllowList allows only listed, denies rest
- empty DenyList allows all
- check includes state in error Display

Size S 15-30m, integration-heavy minimal mocks.

