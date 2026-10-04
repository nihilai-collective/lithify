//! The agent loop. Notice how little of it is about durability: every tool call and
//! model call goes through `branch.effect`, every state change through
//! `branch.apply`, and that is the entire crash-safety story.

use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Result};
use lithify::{Branch, Compactor, State};

use crate::audit::{approx_tokens, ground_truth, Audit, Delta};
use crate::model::{tool_count, Model, Trace};

/// Compaction = ask the model to summarise the notes, journaled as an effect so that a
/// crash mid-compaction resumes with the same summary instead of a fresh one.
pub struct ModelCompactor<'m> {
    pub model: &'m dyn Model,
}

impl Compactor<Audit> for ModelCompactor<'_> {
    type Error = lithify::Error;

    fn compact(&self, branch: &Branch<Audit>, s: &Audit) -> Result<Audit, Self::Error> {
        let n = s.compactions + 1;
        let notes = s.notes.clone();
        let budget = s.budget;
        let summary: String = branch.effect(format!("summarize:{n}:{}", s.done.len()), || {
            self.model.summarize(&notes, budget)
        })?;
        let mut out = s.clone();
        out.notes = vec![summary];
        out.compactions = n;
        out.model_calls += 1;
        out.tokens_in += s.notes.iter().map(|x| approx_tokens(x)).sum::<usize>();
        Ok(out)
    }
}

pub struct RunReport {
    pub steps_this_run: usize,
    pub wall_ms: u128,
}

/// Drive `branch` until it has a report. Idempotent: call it on a fresh branch, a
/// half-finished one, or a finished one, and it does only the remaining work.
pub fn run(
    branch: &Branch<Audit>,
    model: &dyn Model,
    trace: &Trace,
    verbose: bool,
) -> Result<RunReport> {
    let started = Instant::now();
    let mut steps = 0usize;
    let compactor = ModelCompactor { model };

    loop {
        let st = branch.state()?;
        if st.report.is_some() {
            break;
        }
        if st.root.is_empty() {
            return Err(anyhow!(
                "branch {} is not initialised; run `init` first",
                branch.id()
            ));
        }
        steps += 1;

        // Finished reading: write the report, measure, stop.
        if st.pending.is_empty() {
            let notes = st.notes.clone();
            let report: String = branch.effect("report", || model.report(&notes))?;
            let tokens_in = notes.iter().map(|n| approx_tokens(n)).sum();
            branch.apply(Delta::Reported { report, tokens_in })?;
            measure(branch, started)?;
            if verbose {
                eprintln!("[{}] reported", branch.id());
            }
            break;
        }

        // Over budget: compact before reading more.
        if st.size() > st.budget {
            let seq = branch.compact(&compactor)?;
            if verbose {
                let after = branch.state()?;
                eprintln!(
                    "[{}] seq {seq}: compacted {} notes ({} tok) -> {} tok",
                    branch.id(),
                    st.notes.len(),
                    st.size(),
                    after.size()
                );
            }
            continue;
        }

        // One ReAct-style step for the next file: read it (tool), ask the model what to
        // check (model), count each pattern (tools), write the note (model). Every call
        // is an effect with its own key, so a crash anywhere in the step costs at most
        // the one call that was in flight.
        let path = st.pending[0].clone();
        let root = st.root.clone();
        let contents: String = branch.effect(format!("read:{path}"), || {
            trace.log(&format!("tool read {path}"));
            std::fs::read_to_string(Path::new(&root).join(&path))
        })?;
        let plan: Vec<String> =
            branch.effect(format!("plan:{path}"), || model.plan(&path, &contents))?;
        let mut counts: Vec<(String, usize)> = Vec::with_capacity(plan.len());
        for cat in &plan {
            let n: usize = branch.effect(format!("count:{path}:{cat}"), || {
                trace.log(&format!("tool count {path} {cat}"));
                Ok::<_, std::io::Error>(tool_count(&contents, cat))
            })?;
            counts.push((cat.clone(), n));
        }
        let note: String = branch.effect(format!("analyze:{path}"), || {
            model.analyze(&path, &contents, &counts)
        })?;
        let tokens_in = 2 * approx_tokens(&contents) + st.size();
        if verbose {
            eprintln!(
                "[{}] seq {}: analysed {path} ({} pending, ctx {} tok / {})",
                branch.id(),
                branch.head()?,
                st.pending.len() - 1,
                st.size() + approx_tokens(&note),
                st.budget
            );
        }
        branch.apply(Delta::Analyzed {
            path,
            note,
            tokens_in,
        })?;
    }

    Ok(RunReport {
        steps_this_run: steps,
        wall_ms: started.elapsed().as_millis(),
    })
}

/// Record the results-matrix observations for a finished branch.
fn measure(branch: &Branch<Audit>, started: Instant) -> Result<()> {
    let st = branch.state()?;
    let files: Vec<String> = st.done.clone();
    let truth = ground_truth(Path::new(&st.root), &files);
    let found = st.findings_in_report();
    let hit = truth.intersection(&found).count();
    let recall = if truth.is_empty() {
        1.0
    } else {
        hit as f64 / truth.len() as f64
    };
    let precision = if found.is_empty() {
        1.0
    } else {
        hit as f64 / found.len() as f64
    };
    branch.observe("recall", recall)?;
    branch.observe("precision", precision)?;
    branch.observe("compactions", st.compactions as f64)?;
    branch.observe("model_calls", st.model_calls as f64)?;
    branch.observe("tokens_in", st.tokens_in as f64)?;
    branch.observe("files", st.done.len() as f64)?;
    branch.observe("budget", st.budget as f64)?;
    branch.observe("wall_ms_last_run", started.elapsed().as_millis() as f64)?;
    Ok(())
}

/// Human view of a state, for `at` and `status`.
pub fn describe(st: &Audit) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "root={} budget={} done={} pending={} notes={} ctx_tokens={} compactions={} model_calls={} tokens_in={}\n",
        st.root,
        st.budget,
        st.done.len(),
        st.pending.len(),
        st.notes.len(),
        st.size(),
        st.compactions,
        st.model_calls,
        st.tokens_in
    ));
    s.push_str(&format!(
        "findings in notes: {}\n",
        st.findings_in_notes().len()
    ));
    if let Some(r) = &st.report {
        s.push_str(&format!(
            "report: {} findings, {} bytes\n",
            st.findings_in_report().len(),
            r.len()
        ));
    }
    s
}
