//! The consumer-owned part: the context type, its deltas, and the domain heuristics.
//! lithify knows nothing about any of this.

use std::collections::BTreeSet;
use std::path::Path;

use lithify::State;
use serde::{Deserialize, Serialize};

/// One finding: a (file, category) pair with a count. The unit of recall.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Finding {
    pub path: String,
    pub category: String,
    pub count: usize,
}

impl Finding {
    pub fn line(&self) -> String {
        format!("F: {} | {} | {}", self.path, self.category, self.count)
    }
    pub fn parse(line: &str) -> Option<Finding> {
        let rest = line.trim().strip_prefix("F:")?;
        let mut parts = rest.split('|').map(str::trim);
        let path = parts.next()?.to_string();
        let category = parts.next()?.to_string();
        let count = parts.next()?.parse().ok()?;
        Some(Finding {
            path,
            category,
            count,
        })
    }
    /// Severity rank for compaction priority; higher survives longer.
    pub fn severity(&self) -> u8 {
        match self.category.as_str() {
            "unsafe" => 6,
            "fixme" => 5,
            "panic" => 4,
            "todo" => 3,
            "unwrap" => 2,
            "expect" => 1,
            _ => 0,
        }
    }
}

/// The agent's context. Everything the model "knows" is in `notes`; everything else is
/// bookkeeping. Compaction rewrites `notes` and nothing else.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub root: String,
    pub budget: usize,
    pub pending: Vec<String>,
    pub done: Vec<String>,
    /// Free-text working memory, one entry per analysed file (or one compacted entry).
    pub notes: Vec<String>,
    pub report: Option<String>,
    pub compactions: u32,
    pub model_calls: u32,
    /// Approximate tokens of model input consumed so far.
    pub tokens_in: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Delta {
    Init {
        root: String,
        files: Vec<String>,
        budget: usize,
    },
    SetBudget(usize),
    Analyzed {
        path: String,
        note: String,
        tokens_in: usize,
    },
    Reported {
        report: String,
        tokens_in: usize,
    },
}

impl State for Audit {
    type Delta = Delta;

    fn apply(&mut self, d: &Delta) {
        match d {
            Delta::Init {
                root,
                files,
                budget,
            } => {
                self.root = root.clone();
                self.pending = files.clone();
                self.budget = *budget;
            }
            Delta::SetBudget(b) => self.budget = *b,
            Delta::Analyzed {
                path,
                note,
                tokens_in,
            } => {
                self.pending.retain(|p| p != path);
                self.done.push(path.clone());
                self.notes.push(note.clone());
                self.model_calls += 2; // plan + analyze
                self.tokens_in += tokens_in;
            }
            Delta::Reported { report, tokens_in } => {
                self.report = Some(report.clone());
                self.model_calls += 1;
                self.tokens_in += tokens_in;
            }
        }
    }

    /// Context size in approximate tokens: what we would send to the model.
    fn size(&self) -> usize {
        self.notes.iter().map(|n| approx_tokens(n)).sum()
    }
}

impl Audit {
    pub fn findings_in_notes(&self) -> BTreeSet<Finding> {
        self.notes
            .iter()
            .flat_map(|n| n.lines())
            .filter_map(Finding::parse)
            .collect()
    }
    pub fn findings_in_report(&self) -> BTreeSet<Finding> {
        self.report
            .as_deref()
            .unwrap_or("")
            .lines()
            .filter_map(Finding::parse)
            .collect()
    }
}

/// ~4 chars per token. Good enough for budgeting; consistent between mock and real.
pub fn approx_tokens(s: &str) -> usize {
    s.len().div_ceil(4)
}

// ---- domain heuristics ---------------------------------------------------------------

pub const CATEGORIES: &[(&str, &str)] = &[
    ("unsafe", "unsafe "),
    ("fixme", "FIXME"),
    ("panic", "panic!("),
    ("todo", "TODO"),
    ("unwrap", ".unwrap()"),
    ("expect", ".expect("),
];

/// Deterministic findings for a file: the ground truth the mock model reproduces
/// exactly and the real model approximates.
pub fn scan(path: &str, contents: &str) -> Vec<Finding> {
    CATEGORIES
        .iter()
        .filter_map(|(cat, needle)| {
            let count = contents.matches(needle).count();
            (count > 0).then(|| Finding {
                path: path.to_string(),
                category: cat.to_string(),
                count,
            })
        })
        .collect()
}

const EXTS: &[&str] = &[
    "rs", "py", "go", "js", "ts", "c", "h", "cpp", "java", "toml", "md", "sh",
];

/// Files to audit, relative to `root`, sorted, capped.
pub fn list_files(root: &Path, max: usize) -> Vec<String> {
    let mut out: Vec<String> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let n = e.file_name().to_string_lossy();
            !(n == ".git"
                || n == "target"
                || n == "node_modules"
                || n == ".venv"
                || n == "__pycache__")
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .map(|x| EXTS.contains(&x))
                .unwrap_or(false)
        })
        .filter(|e| e.metadata().map(|m| m.len() < 200_000).unwrap_or(false))
        .filter_map(|e| {
            e.path()
                .strip_prefix(root)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
        })
        .collect();
    out.sort();
    out.truncate(max);
    out
}

/// Ground truth for a run: every finding in every file the run was asked to audit.
pub fn ground_truth(root: &Path, files: &[String]) -> BTreeSet<Finding> {
    files
        .iter()
        .flat_map(|f| {
            let contents = std::fs::read_to_string(root.join(f)).unwrap_or_default();
            scan(f, &contents)
        })
        .collect()
}
