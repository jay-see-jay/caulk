# transition

Pure `(state, event) -> (next_state, action)` lookup for the caulk state machine.

## Scope

A `TransitionTable<S>` maps canonical `PromptKey`s (`"State::Event"`) to
`Transition<S>` values (`next_state` + `Action`). The free function
`transition()` resolves one pair. An unmapped pair is an explicit
`TransitionMiss`, never a panic, a default, or a stay-put fallback.

This module is pure: no I/O, no LLM, no time/RNG/env. It never touches the
prompt registry, the runner, or persistence. The driver (caulk-001c) calls
`transition()` *before* `Runner::step` and steps the **current** state;
`next_state` is persisted after.

## Keying decision

Keys are `PromptKey`s built from `Display`, not `(S, E)` pairs, because
`Event` has no `Eq + Hash` bound. Consequences, documented not enforced:

- Payload-bearing events collapse by `Display`: `InfoProvided("a")` and
  `InfoProvided("b")` are the same key — the same behaviour as prompt
  lookup today.
- Two distinct states/events sharing a `Display` string collide on one key
  (pre-existing `PromptKey` behaviour).
- `Display` must stay stable (already required by `core`).

The concrete table struct (not a trait) is deliberate: the table is explicit
and enumerable, which the coverage report (caulk-001d) needs. A
`match`-based trait cannot enumerate its rows.

## NextAction mapping

For the driver / graph-flow, a resolved `Transition` maps onto the existing
`NextAction` enum (unchanged in this leaf):

| Transition | NextAction equivalent |
|---|---|
| `{next_state == current, Llm}` | `Next` / `Continue` |
| `{next_state != current, Llm}` | `Transition(next_state)` |
| `{_, Halt}` | `Halt` |
| `{_, Silent}` | none — skip `Runner::step` entirely |

## Determinism rules

- `keys()` and `entries()` are sorted by canonical string; no `HashMap`
  iteration order leaks into any output.
- Same table + inputs gives the same output regardless of insertion order.
- Duplicate inserts overwrite: last wins, `len()` unchanged, previous
  transition returned.
- Self-loops (`next_state == current`, `Llm`) and `Halt` from any state are
  legal rows in the table.
