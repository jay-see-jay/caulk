//! context_builder — trait + implementations (Wave 1+2)
//!
//! Pluggable context from current event + sliding window history,
//! feeding Rig preamble (`&str`). `Context = String` for simplicity.
//!
//! Why generic over `E: Event`?
//!
//! Products need different context shapes from the same core:
//!
//! - Ferriswheel cares about **file snippets** (50 lines around a cursor)
//! - Inkwell cares about **paragraph history** (last 3 paragraphs)
//! - Generic users may want noop or concat.
//!
//! By keeping `ContextBuilder<E>: Send + Sync` generic, the `forme`
//! runtime stays product-agnostic. Products implement or compose builders
//! without branching core. This also mirrors `Policy<S>` and `LlmAdapter<E>`
//! design.

use std::collections::HashMap;

use crate::core::{Event, FormeError};

/// Builds context string from current event and history.
///
/// `E: Event` generic, not object-safe. Users needing dynamic dispatch
/// should use an enum wrapper for `E`.
///
/// Implementors must be `Send + Sync`.
pub trait ContextBuilder<E: Event>: Send + Sync {
    /// Build context for `event` given `history`.
    ///
    /// `history` is a caller-chosen sliding window (last N events).
    /// Returns `FormeError::ContextBuildFailed` on failure.
    fn build(&self, event: &E, history: &[E]) -> Result<String, FormeError>;
}

// ── Noop ────────────────────────────────────────────────────────────────

/// No-op builder — returns empty string.
///
/// Useful as default when no external context is needed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NoopBuilder;

impl<E: Event> ContextBuilder<E> for NoopBuilder {
    fn build(&self, _event: &E, _history: &[E]) -> Result<String, FormeError> {
        Ok(String::new())
    }
}

// ── FileSnippetBuilder (Ferriswheel pattern) ──────────────────────────

/// File-snippet builder for cursor-oriented products (e.g. Ferriswheel).
///
/// Holds an in-memory map `file path → full content`. When an event
/// encodes a cursor as `file:line` (e.g. `src/main.rs:42`), it returns a
/// 50-line window around that line, with line numbers. Deterministic.
///
/// Parsing rule: event `Display` string is tokenised by whitespace.
/// For each token (reverse order, last wins), strip surrounding
/// `()[]<>"',;` and trailing `:` then look for last `:` separating file and
/// leading digits. File part must contain `.` or `/` to avoid matching
/// `Error:42`. If no match or file missing from map, falls back to
/// `event.to_string()` — never errors for missing files, only for internal
/// map corruption (which cannot happen with `HashMap`).
///
/// Why `HashMap<String, String>` not `PathBuf`? `Event::Display` is a free-form
/// string; using `String` avoids `Path` canonicalization mismatches and keeps
/// the builder portable (WASI, tests). Callers provide keys exactly as they
/// appear in events.
///
/// Example: event = `"edit src/main.rs:42"` with file content 200 lines →
/// returns lines 17..66 (25 before, 24 after + target) formatted as:
/// `src/main.rs:42\n  17 | fn foo() {`
///
/// Edge handling:
/// - line 0 → clamped to 1
/// - line > file len → clamped to len, window is last 50 lines
/// - file empty → `"file:line (empty file)"`
/// - window < 50 lines (small files) → all lines
/// - deterministic: same map + event + history ⇒ same output
#[derive(Clone, Debug, Default)]
pub struct FileSnippetBuilder {
    files: HashMap<String, String>,
}

impl FileSnippetBuilder {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    /// Build from an existing map.
    pub fn from_map(map: HashMap<String, String>) -> Self {
        Self { files: map }
    }

    /// Build from any iterator of `(path, content)`.
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (String, String)>,
    {
        Self {
            files: iter.into_iter().collect(),
        }
    }

    /// Convenience single-file constructor.
    pub fn with_file(path: impl Into<String>, content: impl Into<String>) -> Self {
        let mut map = HashMap::new();
        map.insert(path.into(), content.into());
        Self { files: map }
    }

    /// Mutable insert, builder pattern.
    pub fn insert(&mut self, path: impl Into<String>, content: impl Into<String>) -> &mut Self {
        self.files.insert(path.into(), content.into());
        self
    }

    /// Consuming insert, chainable.
    pub fn with_insert(mut self, path: impl Into<String>, content: impl Into<String>) -> Self {
        self.files.insert(path.into(), content.into());
        self
    }

    fn extract_file_line(event_str: &str) -> Option<(String, usize)> {
        // try tokens reverse for last occurrence
        for token in event_str.split_whitespace().rev() {
            if let Some((f, n)) = Self::parse_token(token) {
                return Some((f, n));
            }
        }
        // fallback: whole string as single token (no whitespace)
        if !event_str.contains(' ') {
            if let Some(pair) = Self::parse_token(event_str) {
                return Some(pair);
            }
        }
        None
    }

    fn parse_token(raw: &str) -> Option<(String, usize)> {
        // strip surrounding punctuation but keep interior ':' '/' '.'
        let trimmed = raw.trim_matches(|c: char| {
            matches!(
                c,
                '(' | ')' | '[' | ']' | '<' | '>' | '"' | '\'' | ',' | ';'
            )
        });
        // also trim trailing ':' that may follow number, and leading punctuation
        let trimmed = trimmed.trim_end_matches(':').trim();
        if trimmed.is_empty() {
            return None;
        }
        let pos = trimmed.rfind(':')?;
        let file_part = trimmed[..pos].trim();
        let line_part = trimmed[pos + 1..].trim();
        if file_part.is_empty() {
            return None;
        }
        // file must look like a file: contain '.' or '/'
        if !file_part.contains('.') && !file_part.contains('/') {
            return None;
        }
        // line_part: leading digits
        let mut digits = String::new();
        for ch in line_part.chars() {
            if ch.is_ascii_digit() {
                digits.push(ch);
            } else {
                break;
            }
        }
        if digits.is_empty() {
            return None;
        }
        let line_num: usize = digits.parse().ok()?;
        if line_num == 0 {
            // allow 0 but clamp later; still valid
        }
        Some((file_part.to_string(), line_num))
    }

    fn window_snippet(&self, file: &str, line: usize, content: &str) -> String {
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        if total == 0 {
            return format!("{file}:{line} (empty file)");
        }
        let target = line.clamp(1, total);
        const WINDOW: usize = 50;
        const HALF: usize = WINDOW / 2;
        let mut start = if target > HALF { target - HALF } else { 1 };
        let mut end = (start + WINDOW - 1).min(total);
        if total >= WINDOW && end - start + 1 < WINDOW {
            start = total - WINDOW + 1;
            end = total;
        }
        // header + numbered lines
        let mut out = String::new();
        out.push_str(&format!("{file}:{line}\n"));
        for lineno in start..=end {
            // lines vec 0-indexed
            out.push_str(&format!("{:>4} | {}\n", lineno, lines[lineno - 1]));
        }
        out.trim_end().to_string()
    }
}

impl<E: Event> ContextBuilder<E> for FileSnippetBuilder {
    fn build(&self, event: &E, _history: &[E]) -> Result<String, FormeError> {
        let ev_str = event.to_string();
        if let Some((file, line)) = Self::extract_file_line(&ev_str) {
            if let Some(content) = self.files.get(&file) {
                return Ok(self.window_snippet(&file, line, content));
            }
        }
        // fallback deterministic: event itself
        Ok(ev_str)
    }
}

// ── ParagraphBuilder (Inkwell pattern) ─────────────────────────────────

/// Paragraph-history builder for prose-oriented products (e.g. Inkwell).
///
/// Groups history into paragraphs split by blank line, returns last N
/// paragraphs plus current event. Deterministic, snapshot-friendly.
///
/// Definition of paragraph:
/// - any contiguous non-blank lines inside a single event form one paragraph
/// - blank line (`line.trim().is_empty()`) ends a paragraph
/// - history events are processed in order; their internal paragraphs are
///   preserved in order
///
/// `N=3` by default, configurable via `with_n`.
///
/// Example:
/// history = `["para1\n\npara2", "para3 line1\npara3 line2"]`,
/// current = `"para4"` → returns `"para2\n\npara3 line1\npara3 line2\n\npara4"`
/// (last 2 history paras + current; if history had 5 paras, we'd take last 3).
///
/// Why not just `concat`? Prose products care about recent thematic blocks,
/// not full history. Keeping 3 paragraphs caps prompt size while retaining
/// narrative coherence. Returning `current` always ensures the model sees the
/// immediate edit.
#[derive(Clone, Debug)]
pub struct ParagraphBuilder {
    n: usize,
}

impl Default for ParagraphBuilder {
    fn default() -> Self {
        Self { n: 3 }
    }
}

impl ParagraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_n(n: usize) -> Self {
        Self { n }
    }

    fn split_into_paragraphs(text: &str) -> Vec<String> {
        let mut paras = Vec::new();
        let mut cur_lines: Vec<String> = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                if !cur_lines.is_empty() {
                    paras.push(cur_lines.join("\n"));
                    cur_lines.clear();
                }
            } else {
                cur_lines.push(line.to_string());
            }
        }
        if !cur_lines.is_empty() {
            paras.push(cur_lines.join("\n"));
        }
        // Filter empty trimmed paras
        paras
            .into_iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect()
    }
}

impl<E: Event> ContextBuilder<E> for ParagraphBuilder {
    fn build(&self, event: &E, history: &[E]) -> Result<String, FormeError> {
        let mut paras: Vec<String> = Vec::new();
        for h in history {
            let h_str = h.to_string();
            paras.extend(Self::split_into_paragraphs(&h_str));
        }

        let take = self.n;
        let selected: Vec<String> = if paras.len() <= take {
            paras
        } else {
            paras
                .into_iter()
                .rev()
                .take(take)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect()
        };

        let cur = event.to_string();
        if selected.is_empty() {
            Ok(cur)
        } else {
            Ok(format!("{}\n\n{}", selected.join("\n\n"), cur))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    use serde::{Deserialize, Serialize};

    // ── helpers ─────────────────────────────────────────────────────

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct SimpleEvent(String);

    impl fmt::Display for SimpleEvent {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct CustomEvent {
        id: u32,
        text: String,
    }

    impl fmt::Display for CustomEvent {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}:{}", self.id, self.text)
        }
    }

    /// Concatenates history with "\n" and appends current event.
    #[derive(Clone, Debug, Default)]
    struct ConcatBuilder;

    impl<E: Event> ContextBuilder<E> for ConcatBuilder {
        fn build(&self, event: &E, history: &[E]) -> Result<String, FormeError> {
            let mut parts: Vec<String> = Vec::with_capacity(history.len() + 1);
            for h in history {
                let s: String = h.to_string();
                parts.push(s);
            }
            let cur: String = event.to_string();
            parts.push(cur);
            let joined: String = parts.join("\n");
            Ok(joined)
        }
    }

    #[derive(Clone, Debug, Default)]
    struct FailingBuilder;

    impl<E: Event> ContextBuilder<E> for FailingBuilder {
        fn build(&self, _event: &E, _history: &[E]) -> Result<String, FormeError> {
            Err(FormeError::ContextBuildFailed("forced failure".into()))
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    // ── original tests ─────────────────────────────────────────────

    #[test]
    fn noop_returns_empty() {
        let builder: NoopBuilder = NoopBuilder;
        let event: SimpleEvent = SimpleEvent("hello".into());
        let history: Vec<SimpleEvent> = vec![SimpleEvent("one".into()), SimpleEvent("two".into())];
        let result: Result<String, FormeError> = builder.build(&event, &history);
        assert!(result.is_ok());
        let ctx: String = result.unwrap();
        assert_eq!(ctx, "");
    }

    #[test]
    fn noop_returns_empty_no_history() {
        let builder: NoopBuilder = NoopBuilder::default();
        let event: SimpleEvent = SimpleEvent("ev".into());
        let history: Vec<SimpleEvent> = Vec::new();
        let ctx: String = builder.build(&event, &history).unwrap();
        assert_eq!(ctx.len(), 0);
        assert!(ctx.is_empty());
    }

    #[test]
    fn custom_builder_concatenates_history_deterministic() {
        let builder: ConcatBuilder = ConcatBuilder;
        let history: Vec<SimpleEvent> = vec![
            SimpleEvent("a".into()),
            SimpleEvent("b".into()),
            SimpleEvent("c".into()),
        ];
        let event: SimpleEvent = SimpleEvent("d".into());
        let ctx: String = builder.build(&event, &history).unwrap();
        assert_eq!(ctx, "a\nb\nc\nd");

        // deterministic: same input -> same join
        let ctx2: String = builder.build(&event, &history).unwrap();
        assert_eq!(ctx, ctx2);
    }

    #[test]
    fn snapshot_test_deterministic() {
        let builder: ConcatBuilder = ConcatBuilder::default();
        let history: Vec<SimpleEvent> =
            vec![SimpleEvent("first".into()), SimpleEvent("second".into())];
        let event: SimpleEvent = SimpleEvent("current".into());

        let out1: String = builder.build(&event, &history).unwrap();
        let out2: String = builder.build(&event, &history).unwrap();

        // snapshot: fully deterministic string
        let expected: String = "first\nsecond\ncurrent".to_string();
        assert_eq!(out1, expected);
        assert_eq!(out2, expected);
        assert_eq!(out1, out2);
    }

    #[test]
    fn error_propagation_returns_context_build_failed() {
        let builder: FailingBuilder = FailingBuilder;
        let event: SimpleEvent = SimpleEvent("x".into());
        let history: Vec<SimpleEvent> = vec![];
        let result: Result<String, FormeError> = builder.build(&event, &history);
        assert!(result.is_err());
        let err: FormeError = result.unwrap_err();
        let msg: String = err.to_string();
        assert!(msg.contains("context build failed"), "msg was: {msg}");
        assert!(msg.contains("forced failure"), "msg was: {msg}");
        match err {
            FormeError::ContextBuildFailed(s) => {
                assert_eq!(s, "forced failure");
            }
            _ => panic!("expected ContextBuildFailed, got {:?}", err),
        }
    }

    #[test]
    fn send_sync_compile_check() {
        // compile-time bound check: if this compiles, NoopBuilder is Send+Sync
        assert_send_sync::<NoopBuilder>();
        assert_send_sync::<ConcatBuilder>();
        assert_send_sync::<FailingBuilder>();
        assert_send_sync::<FileSnippetBuilder>();
        assert_send_sync::<ParagraphBuilder>();

        fn assert_builder_send_sync<E: Event, B: ContextBuilder<E> + Send + Sync>() {}
        assert_builder_send_sync::<SimpleEvent, NoopBuilder>();
        assert_builder_send_sync::<CustomEvent, ConcatBuilder>();
        assert_builder_send_sync::<SimpleEvent, FileSnippetBuilder>();
        assert_builder_send_sync::<SimpleEvent, ParagraphBuilder>();
    }

    #[test]
    fn generic_over_custom_event() {
        let builder: ConcatBuilder = ConcatBuilder;
        let history: Vec<CustomEvent> = vec![
            CustomEvent {
                id: 1,
                text: "hello".into(),
            },
            CustomEvent {
                id: 2,
                text: "world".into(),
            },
        ];
        let event: CustomEvent = CustomEvent {
            id: 3,
            text: "now".into(),
        };
        let ctx: String = builder.build(&event, &history).unwrap();
        // Display is "id:text"
        assert_eq!(ctx, "1:hello\n2:world\n3:now");

        // also Noop works over custom event
        let noop: NoopBuilder = NoopBuilder;
        let empty: String = noop.build(&event, &history).unwrap();
        assert_eq!(empty, "");
    }

    #[test]
    fn noop_generic_over_string_event() {
        // String itself satisfies Event via blanket impl
        let builder: NoopBuilder = NoopBuilder;
        let event: String = "ev".to_string();
        let history: Vec<String> = vec!["a".to_string(), "b".to_string()];
        let ctx: String = builder.build(&event, &history).unwrap();
        assert_eq!(ctx, "");
    }

    // ── FileSnippetBuilder tests ───────────────────────────────────

    fn make_large_file(lines: usize) -> String {
        (1..=lines)
            .map(|i| format!("line {i} content"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn file_snippet_50_lines_middle() {
        let content = make_large_file(200);
        let builder = FileSnippetBuilder::with_file("src/main.rs", content);
        let ev = SimpleEvent("edit src/main.rs:42".into());
        let ctx = builder.build(&ev, &[]).unwrap();
        // should start with header src/main.rs:42
        assert!(ctx.starts_with("src/main.rs:42\n"));
        // contains line 42
        assert!(ctx.contains("  42 | line 42 content"));
        // 50 lines window: 17..66 (25 before 42, 24 after inclusive makes 50)
        assert!(ctx.contains("  17 | line 17 content"));
        assert!(ctx.contains("  66 | line 66 content"));
        // line 16 and 67 should not be present
        assert!(!ctx.contains("  16 |"));
        assert!(!ctx.contains("  67 |"));
        // count numbered lines = 50
        let numbered = ctx.lines().filter(|l| l.contains('|')).count();
        assert_eq!(numbered, 50);
    }

    #[test]
    fn file_snippet_edges_start_and_end() {
        let content = make_large_file(100);
        let builder = FileSnippetBuilder::with_file("src/lib.rs", content.clone());

        // cursor at line 1 → window 1..50
        let ev1 = SimpleEvent("src/lib.rs:1".into());
        let ctx1 = builder.build(&ev1, &[]).unwrap();
        assert!(ctx1.contains("   1 | line 1 content"));
        assert!(ctx1.contains("  50 | line 50 content"));
        assert!(!ctx1.contains("  51 |"));
        let c1 = ctx1.lines().filter(|l| l.contains('|')).count();
        assert_eq!(c1, 50);

        // cursor at line 100 (end) → last 50 lines 51..100
        let ev2 = SimpleEvent("open src/lib.rs:100".into());
        let ctx2 = builder.build(&ev2, &[]).unwrap();
        assert!(ctx2.contains("  51 | line 51 content"));
        assert!(ctx2.contains(" 100 | line 100 content"));
        assert!(!ctx2.contains("  50 | line 50 content"));
        let c2 = ctx2.lines().filter(|l| l.contains('|')).count();
        assert_eq!(c2, 50);

        // small file <50 lines: cursor 5 in 10-line file → all 10 lines
        let small = make_large_file(10);
        let builder_small = FileSnippetBuilder::with_file("a.rs", small);
        let ev_small = SimpleEvent("a.rs:5".into());
        let ctx_small = builder_small.build(&ev_small, &[]).unwrap();
        let counted = ctx_small.lines().filter(|l| l.contains('|')).count();
        assert_eq!(counted, 10);
        assert!(ctx_small.contains("   1 |"));
        assert!(ctx_small.contains("  10 |"));
    }

    #[test]
    fn file_snippet_fallback_missing_file() {
        let builder = FileSnippetBuilder::with_file("exists.rs", "one\ntwo");
        let ev = SimpleEvent("edit missing.rs:10".into());
        let ctx = builder.build(&ev, &[]).unwrap();
        // fallback returns event itself
        assert_eq!(ctx, "edit missing.rs:10");

        // no file:line in event → fallback
        let ev2 = SimpleEvent("just a message".into());
        let ctx2 = builder.build(&ev2, &[]).unwrap();
        assert_eq!(ctx2, "just a message");
    }

    #[test]
    fn file_snippet_deterministic_snapshot() {
        let content = make_large_file(60);
        let builder =
            FileSnippetBuilder::from_iter(vec![("src/app.rs".to_string(), content.clone())]);
        let ev = SimpleEvent("src/app.rs:30".into());
        let out1 = builder.build(&ev, &[]).unwrap();
        let out2 = builder.build(&ev, &[]).unwrap();
        assert_eq!(out1, out2);

        // snapshot: first 2 lines known
        let lines: Vec<&str> = out1.lines().collect();
        assert_eq!(lines[0], "src/app.rs:30");
        // window 5..54? actually 30-25=5 → 5..54 inclusive 50 lines
        assert!(lines[1].contains("5 | line 5 content"));
        assert!(lines.last().unwrap().contains("54 | line 54 content"));
    }

    #[test]
    fn file_snippet_empty_file() {
        let builder = FileSnippetBuilder::with_file("empty.rs", "");
        let ev = SimpleEvent("empty.rs:1".into());
        let ctx = builder.build(&ev, &[]).unwrap();
        assert!(ctx.contains("(empty file)"));
        assert!(ctx.starts_with("empty.rs:1"));
    }

    #[test]
    fn file_snippet_parses_with_trailing_punct() {
        let content = make_large_file(20);
        let builder = FileSnippetBuilder::with_file("src/main.rs", content);
        // token with comma
        let ev = SimpleEvent("see src/main.rs:10, please".into());
        let ctx = builder.build(&ev, &[]).unwrap();
        assert!(ctx.starts_with("src/main.rs:10"));
        // token with parens
        let ev2 = SimpleEvent("at (src/main.rs:12)".into());
        let ctx2 = builder.build(&ev2, &[]).unwrap();
        assert!(ctx2.starts_with("src/main.rs:12"));
    }

    // ── ParagraphBuilder tests ─────────────────────────────────────

    #[test]
    fn paragraph_builder_last_n_plus_current() {
        let builder = ParagraphBuilder::new(); // n=3
        let history = vec![
            SimpleEvent("para1".into()),
            SimpleEvent("para2".into()),
            SimpleEvent("para3".into()),
            SimpleEvent("para4".into()),
        ];
        let ev = SimpleEvent("current".into());
        let ctx = builder.build(&ev, &history).unwrap();
        // last 3 paras from history = para2,para3,para4 + current
        assert_eq!(ctx, "para2\n\npara3\n\npara4\n\ncurrent");

        // deterministic
        let ctx2 = builder.build(&ev, &history).unwrap();
        assert_eq!(ctx, ctx2);
    }

    #[test]
    fn paragraph_builder_split_by_blank_line() {
        let builder = ParagraphBuilder::with_n(2);
        let history = vec![
            SimpleEvent("first para line1\nfirst para line2\n\nsecond para".into()),
            SimpleEvent("third para".into()),
        ];
        let ev = SimpleEvent("current".into());
        let ctx = builder.build(&ev, &history).unwrap();
        // history paras: ["first para line1\nfirst para line2", "second para", "third para"]
        // last 2 = second para + third para
        assert_eq!(ctx, "second para\n\nthird para\n\ncurrent");
    }

    #[test]
    fn paragraph_builder_snapshot_deterministic() {
        let builder = ParagraphBuilder::default();
        let history = vec![
            SimpleEvent("a\n\nb".into()),
            SimpleEvent("c\n\nd\n\ne".into()),
        ];
        let ev = SimpleEvent("f".into());
        let ctx = builder.build(&ev, &history).unwrap();
        // paras = ["a","b","c","d","e"]; last 3 = c,d,e
        assert_eq!(ctx, "c\n\nd\n\ne\n\nf");

        // with empty history → only current
        let builder2 = ParagraphBuilder::new();
        let ctx_empty = builder2.build(&ev, &[]).unwrap();
        assert_eq!(ctx_empty, "f");
    }

    #[test]
    fn paragraph_builder_trims_and_filters() {
        let builder = ParagraphBuilder::with_n(10);
        let history = vec![SimpleEvent("  \n\npara\n\n  \n".into())];
        let ev = SimpleEvent("cur".into());
        let ctx = builder.build(&ev, &history).unwrap();
        assert_eq!(ctx, "para\n\ncur");
    }
}
