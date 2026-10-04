//! Crash test: a child process runs an agent-like loop (effect, then delta, repeat),
//! gets SIGKILLed at a random moment, and the parent resumes the same loop in-process
//! against the same journal. Afterwards every effect must have executed exactly once
//! and the state must equal the uninterrupted run.
//!
//! The child is this same test binary, re-executed with `LITHIFY_KILL9_CHILD` set.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use lithify::{Journal, State};
use serde::{Deserialize, Serialize};

const N: i64 = 400;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct Acc {
    sum: i64,
    steps: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Delta {
    Took(i64),
}

impl State for Acc {
    type Delta = Delta;
    fn apply(&mut self, d: &Delta) {
        let Delta::Took(v) = d;
        self.sum += v;
        self.steps += 1;
    }
}

/// The "agent loop". Each step: an effect that appends to a side file (the observable
/// side effect), then a delta recording it. Resumable because state().steps tells us
/// where we are and the effect key is derived from the step number.
fn run(journal: &Path, side_file: &Path, upto: i64, slow: bool) -> lithify::Result<Acc> {
    let j = Journal::<Acc>::open(journal)?;
    let b = j.root();
    loop {
        let st = b.state()?;
        if st.steps >= upto {
            return Ok(st);
        }
        let step = st.steps + 1;
        let v: i64 = b.effect(format!("step:{step}"), || -> std::io::Result<i64> {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(side_file)?;
            writeln!(f, "{step}")?;
            f.sync_all()?;
            Ok(step * 3)
        })?;
        if slow {
            std::thread::sleep(Duration::from_micros(300));
        }
        b.apply(Delta::Took(v))?;
    }
}

#[test]
fn child_entry() {
    // Only acts when invoked as the child; otherwise this is a no-op test.
    let Ok(dir) = std::env::var("LITHIFY_KILL9_CHILD") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let _ = run(&dir.join("j.redb"), &dir.join("side.txt"), N, true);
}

#[test]
fn resume_after_sigkill_executes_each_effect_once() {
    if std::env::var("LITHIFY_KILL9_CHILD").is_ok() {
        return;
    }
    let exe = std::env::current_exe().unwrap();
    for round in 0..6 {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("j.redb");
        let side = dir.path().join("side.txt");

        let mut child = Command::new(&exe)
            .arg("child_entry")
            .arg("--exact")
            .arg("--nocapture")
            .env("LITHIFY_KILL9_CHILD", dir.path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        // Let it get somewhere, then kill it mid-flight. Vary the delay per round so
        // the kill lands at different phases of the step.
        std::thread::sleep(Duration::from_millis(40 + round * 23));
        child.kill().unwrap(); // SIGKILL on unix
        child.wait().unwrap();

        let progressed = Journal::<Acc>::open(&journal)
            .unwrap()
            .root()
            .head()
            .unwrap();

        // Resume in-process.
        let final_state = run(&journal, &side, N, false).unwrap();
        assert_eq!(final_state.steps, N);
        assert_eq!(final_state.sum, (1..=N).map(|s| s * 3).sum::<i64>());

        // Side effects. Every step whose result was journaled before the kill must
        // appear exactly once. The one honest exception is the step that was *in
        // flight*: if the kill landed after the file write but before the journal
        // commit, that step re-executes on resume (at-least-once; the journal and the
        // outside world are not one transaction). Each step is two events (effect at
        // odd seq, delta at even), so the in-flight step is progressed/2 + 1 and only
        // when progressed is even (its effect was not yet committed).
        let side_txt = std::fs::read_to_string(&side).unwrap();
        let mut lines: Vec<i64> = side_txt.lines().map(|l| l.parse().unwrap()).collect();
        let in_flight = progressed
            .is_multiple_of(2)
            .then_some((progressed / 2) as i64 + 1);
        let mut dups = Vec::new();
        let mut i = 1;
        while i < lines.len() {
            if lines[i] == lines[i - 1] {
                dups.push(lines[i]);
                lines.remove(i);
            } else {
                i += 1;
            }
        }
        assert_eq!(
            lines,
            (1..=N).collect::<Vec<_>>(),
            "round {round}: child reached seq {progressed}; side effects missing or out of order"
        );
        assert!(
            dups.is_empty() || (dups.len() == 1 && Some(dups[0]) == in_flight),
            "round {round}: child reached seq {progressed}; unexpected re-executions {dups:?} (in-flight step was {in_flight:?})"
        );

        // Journal: head is exactly 2N (effect + delta per step), replay agrees.
        let j = Journal::<Acc>::open(&journal).unwrap();
        assert_eq!(j.root().head().unwrap(), (2 * N) as u64);
        assert_eq!(j.root().state().unwrap(), final_state);
        eprintln!(
            "round {round}: killed at seq {progressed}, resumed cleanly to {}",
            2 * N
        );
    }
}
