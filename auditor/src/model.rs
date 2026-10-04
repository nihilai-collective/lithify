//! Two models behind one trait: a deterministic mock (so the demos and tests run
//! offline and reproducibly) and any OpenAI-compatible chat endpoint.
//!
//! Each file is handled ReAct-style in two model turns with tool calls between them:
//! `plan` decides which patterns to check, tools count them, `analyze` writes the note.
//! That is what makes the crash comparison meaningful: a step has several side effects,
//! and the question is how many of them a crash makes you redo.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};

use crate::audit::{approx_tokens, Finding, CATEGORIES};

pub trait Model {
    fn name(&self) -> &str;
    /// Which categories to check in this file.
    fn plan(&self, path: &str, contents: &str) -> Result<Vec<String>>;
    /// Notes on one file given the tool counts. Must contain `F: path | category | count`
    /// lines for every non-zero count.
    fn analyze(&self, path: &str, contents: &str, counts: &[(String, usize)]) -> Result<String>;
    /// Rewrite `notes` into one note of at most roughly `budget_tokens` tokens.
    fn summarize(&self, notes: &[String], budget_tokens: usize) -> Result<String>;
    /// Final report from the notes. Must repeat every `F:` line the model still has.
    fn report(&self, notes: &[String]) -> Result<String>;
}

/// The "outside world": one line per model or tool invocation, appended at the moment
/// the call is *made* (that is when an API call is billed), not when it returns.
#[derive(Clone, Default)]
pub struct Trace(pub Option<Arc<PathBuf>>);

impl Trace {
    pub fn new(p: Option<PathBuf>) -> Self {
        Trace(p.map(Arc::new))
    }
    pub fn log(&self, line: &str) {
        if let Some(p) = &self.0 {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(p.as_ref())
            {
                let _ = writeln!(f, "{line}");
            }
        }
    }
}

/// Wraps a model so every call is traced before it is made.
pub struct Traced<M> {
    pub inner: M,
    pub trace: Trace,
}

impl<M: Model> Model for Traced<M> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn plan(&self, path: &str, contents: &str) -> Result<Vec<String>> {
        self.trace.log(&format!("model plan {path}"));
        self.inner.plan(path, contents)
    }
    fn analyze(&self, path: &str, contents: &str, counts: &[(String, usize)]) -> Result<String> {
        self.trace.log(&format!("model analyze {path}"));
        self.inner.analyze(path, contents, counts)
    }
    fn summarize(&self, notes: &[String], budget_tokens: usize) -> Result<String> {
        self.trace.log(&format!(
            "model summarize {} notes to {budget_tokens}",
            notes.len()
        ));
        self.inner.summarize(notes, budget_tokens)
    }
    fn report(&self, notes: &[String]) -> Result<String> {
        self.trace
            .log(&format!("model report from {} notes", notes.len()));
        self.inner.report(notes)
    }
}

/// Adds latency to every model call, so a crash demo has something to interrupt.
/// Sits *inside* `Traced`, so the trace line precedes the latency, as a real API call's
/// billing precedes its response.
pub struct Slow<M> {
    pub inner: M,
    pub ms: u64,
}

impl<M: Model> Model for Slow<M> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn plan(&self, path: &str, contents: &str) -> Result<Vec<String>> {
        self.nap();
        self.inner.plan(path, contents)
    }
    fn analyze(&self, path: &str, contents: &str, counts: &[(String, usize)]) -> Result<String> {
        self.nap();
        self.inner.analyze(path, contents, counts)
    }
    fn summarize(&self, notes: &[String], budget_tokens: usize) -> Result<String> {
        self.nap();
        self.inner.summarize(notes, budget_tokens)
    }
    fn report(&self, notes: &[String]) -> Result<String> {
        self.nap();
        self.inner.report(notes)
    }
}

impl<M> Slow<M> {
    fn nap(&self) {
        if self.ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.ms));
        }
    }
}

/// The one tool: count a pattern in a file. Trivial on purpose; it stands in for any
/// tool call whose re-execution you would rather avoid.
pub fn tool_count(contents: &str, category: &str) -> usize {
    CATEGORIES
        .iter()
        .find(|(c, _)| *c == category)
        .map(|(_, needle)| contents.matches(needle).count())
        .unwrap_or(0)
}

// ---- mock ------------------------------------------------------------------------------

/// Checks every category, writes notes from the tool counts, and compacts by dropping
/// the least severe findings until the budget fits. Deterministic, so recall is a pure
/// function of the budget and the file order: that is what makes the sweep a
/// falsifiable experiment.
pub struct Mock;

impl Model for Mock {
    fn name(&self) -> &str {
        "mock"
    }

    fn plan(&self, _path: &str, _contents: &str) -> Result<Vec<String>> {
        Ok(CATEGORIES.iter().map(|(c, _)| c.to_string()).collect())
    }

    fn analyze(&self, path: &str, contents: &str, counts: &[(String, usize)]) -> Result<String> {
        let mut note = format!(
            "## {path}\n{} lines, {} bytes.\n",
            contents.lines().count(),
            contents.len()
        );
        let findings: Vec<Finding> = counts
            .iter()
            .filter(|(_, n)| *n > 0)
            .map(|(c, n)| Finding {
                path: path.to_string(),
                category: c.clone(),
                count: *n,
            })
            .collect();
        if findings.is_empty() {
            note.push_str("No findings.\n");
        }
        for f in findings {
            note.push_str(&f.line());
            note.push('\n');
            // A couple of evidence lines per finding, like a real reviewer would quote.
            // This is also what makes notes big enough to need compacting.
            let needle = CATEGORIES
                .iter()
                .find(|(c, _)| *c == f.category)
                .map(|(_, n)| *n)
                .unwrap_or("");
            for (i, l) in contents
                .lines()
                .enumerate()
                .filter(|(_, l)| l.contains(needle))
                .take(2)
            {
                let snippet: String = l.trim().chars().take(90).collect();
                note.push_str(&format!("    L{}: {snippet}\n", i + 1));
            }
        }
        Ok(note)
    }

    fn summarize(&self, notes: &[String], budget_tokens: usize) -> Result<String> {
        let mut all: Vec<Finding> = notes
            .iter()
            .flat_map(|n| n.lines())
            .filter_map(Finding::parse)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let total = all.len();
        // Most severe, then most numerous, survive.
        all.sort_by(|a, b| {
            b.severity()
                .cmp(&a.severity())
                .then(b.count.cmp(&a.count))
                .then(a.path.cmp(&b.path))
        });
        // A summary is allowed about half the budget, so there is room to keep working
        // after compaction. Everything that does not fit is forgotten.
        let allowance = budget_tokens / 2;
        let mut out = String::new();
        let mut kept = 0usize;
        for f in &all {
            let line = f.line();
            if approx_tokens(&out) + approx_tokens(&line) + 1 > allowance {
                break;
            }
            out.push_str(&line);
            out.push('\n');
            kept += 1;
        }
        Ok(format!(
            "## compacted ({kept} of {total} findings retained; {} notes squashed)\n{out}",
            notes.len()
        ))
    }

    fn report(&self, notes: &[String]) -> Result<String> {
        let findings: BTreeSet<Finding> = notes
            .iter()
            .flat_map(|n| n.lines())
            .filter_map(Finding::parse)
            .collect();
        let mut s = format!("# Audit report\n\n{} findings.\n\n", findings.len());
        for f in findings {
            s.push_str(&f.line());
            s.push('\n');
        }
        Ok(s)
    }
}

// ---- OpenAI-compatible -----------------------------------------------------------------

/// Any `/v1/chat/completions` endpoint: OpenAI, vLLM, llama.cpp, your own engine.
/// Configured from `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`.
pub struct OpenAi {
    base: String,
    key: String,
    model: String,
    http: reqwest::blocking::Client,
}

impl OpenAi {
    pub fn from_env() -> Result<Self> {
        let base =
            std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());
        let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        Ok(OpenAi {
            base: base.trim_end_matches('/').to_string(),
            key,
            model,
            http: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()?,
        })
    }

    fn chat(&self, system: &str, user: &str) -> Result<String> {
        let body = serde_json::json!({
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        });
        let mut req = self
            .http
            .post(format!("{}/chat/completions", self.base))
            .json(&body);
        if !self.key.is_empty() {
            req = req.bearer_auth(&self.key);
        }
        let resp: serde_json::Value = req
            .send()
            .context("model request failed")?
            .error_for_status()
            .context("model returned an error status")?
            .json()
            .context("model returned non-JSON")?;
        resp["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow!("no content in model response: {resp}"))
    }
}

const SYSTEM: &str = "You are a meticulous code auditor. You report findings ONLY as lines of the exact form\n\
`F: <path> | <category> | <count>` where category is one of: unsafe, fixme, panic, todo, unwrap, expect,\n\
and count is the number given to you by the counting tool. After the F: lines you may add at most three\n\
short sentences of commentary. Never invent paths or counts.";

impl Model for OpenAi {
    fn name(&self) -> &str {
        "openai"
    }

    fn plan(&self, path: &str, contents: &str) -> Result<Vec<String>> {
        let head: String = contents.chars().take(1500).collect();
        let user = format!(
            "File: {path}\n\n```\n{head}\n```\n\nWhich of these categories are worth counting in this file: \
             unsafe, fixme, panic, todo, unwrap, expect? Answer with a JSON array of category names only."
        );
        let out = self.chat(SYSTEM, &user)?;
        let start = out.find('[').unwrap_or(0);
        let end = out.rfind(']').map(|i| i + 1).unwrap_or(out.len());
        let cats: Vec<String> = serde_json::from_str(&out[start..end]).unwrap_or_default();
        let valid: Vec<String> = cats
            .into_iter()
            .filter(|c| CATEGORIES.iter().any(|(k, _)| k == c))
            .collect();
        Ok(if valid.is_empty() {
            CATEGORIES.iter().map(|(c, _)| c.to_string()).collect()
        } else {
            valid
        })
    }

    fn analyze(&self, path: &str, contents: &str, counts: &[(String, usize)]) -> Result<String> {
        let tool: Vec<String> = counts.iter().map(|(c, n)| format!("{c}: {n}")).collect();
        let user = format!(
            "File: {path}\n\nTool counts:\n{}\n\n```\n{contents}\n```\n\nReport findings for this file.",
            tool.join("\n")
        );
        let out = self.chat(SYSTEM, &user)?;
        Ok(format!("## {path}\n{out}\n"))
    }

    fn summarize(&self, notes: &[String], budget_tokens: usize) -> Result<String> {
        let user = format!(
            "Your working notes are over budget. Rewrite them into ONE note of at most {} tokens \
             (about {} characters). Keep as many `F:` lines as fit, preferring the most severe \
             (unsafe > fixme > panic > todo > unwrap > expect) and the largest counts. Drop everything else.\n\n{}",
            budget_tokens / 2,
            budget_tokens * 2,
            notes.join("\n")
        );
        let out = self.chat(SYSTEM, &user)?;
        Ok(format!(
            "## compacted ({} notes squashed)\n{out}\n",
            notes.len()
        ))
    }

    fn report(&self, notes: &[String]) -> Result<String> {
        let user = format!(
            "Write the final audit report. Repeat EVERY `F:` line present in your notes, deduplicated, \
             one per line, then a short summary paragraph.\n\n{}",
            notes.join("\n")
        );
        self.chat(SYSTEM, &user)
    }
}
