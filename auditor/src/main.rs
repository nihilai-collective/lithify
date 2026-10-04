mod agent;
mod audit;
mod model;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use lithify::{EventKind, Journal};

use audit::{Audit, Delta};
use model::{Mock, Model, OpenAi, Slow, Trace, Traced};

#[derive(Parser)]
#[command(
    name = "auditor",
    about = "Example lithify agent: a crash-safe, forkable, budgeted code auditor."
)]
struct Cli {
    /// Journal file. Created if missing, resumed if present.
    #[arg(long, global = true, default_value = "auditor.redb")]
    db: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Mock,
    Openai,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run (or resume) a branch until it has produced a report.
    Run {
        /// Repository to audit. Required to initialise a fresh journal.
        #[arg(long)]
        repo: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        branch: u64,
        /// Context budget in approximate tokens (fresh journals only).
        #[arg(long, default_value_t = 3000)]
        budget: usize,
        #[arg(long, default_value_t = 80)]
        max_files: usize,
        #[arg(long, value_enum, default_value_t = Backend::Mock)]
        model: Backend,
        /// Append one line per model invocation here (the "outside world").
        #[arg(long)]
        trace: Option<PathBuf>,
        #[arg(short, long)]
        verbose: bool,
    },
    /// Show branches and their current state.
    Status,
    /// Time travel: the state of a branch as of a sequence number.
    At {
        #[arg(long)]
        seq: u64,
        #[arg(long, default_value_t = 0)]
        branch: u64,
        /// Also print the notes (the model's working memory) at that point.
        #[arg(long)]
        notes: bool,
    },
    /// List journal records for a branch (resolved through its ancestors).
    History {
        #[arg(long, default_value_t = 0)]
        branch: u64,
        #[arg(long, default_value_t = 1)]
        from: u64,
        #[arg(long, default_value_t = u64::MAX)]
        to: u64,
    },
    /// Fork a branch at a sequence number with a different budget.
    Fork {
        #[arg(long)]
        at: u64,
        #[arg(long)]
        budget: usize,
        #[arg(long, default_value_t = 0)]
        branch: u64,
    },
    /// Fork at `at` once per budget, run every fork to completion, print the matrix.
    Sweep {
        #[arg(long)]
        at: u64,
        /// Comma-separated budgets, e.g. 1000,2000,4000,8000
        #[arg(long, value_delimiter = ',')]
        budgets: Vec<usize>,
        #[arg(long, default_value_t = 0)]
        branch: u64,
        #[arg(long, value_enum, default_value_t = Backend::Mock)]
        model: Backend,
        #[arg(long)]
        trace: Option<PathBuf>,
    },
    /// Print every observation joined to its branch's parameters, as CSV.
    Matrix,
    /// Print a branch's report.
    Report {
        #[arg(long, default_value_t = 0)]
        branch: u64,
    },
    /// Start a run as a child process, SIGKILL it at a random moment, repeat, then
    /// finish the run and prove that no model call was repeated.
    CrashDemo {
        #[arg(long)]
        repo: PathBuf,
        #[arg(long, default_value_t = 3)]
        kills: u32,
        #[arg(long, default_value_t = 3000)]
        budget: usize,
        #[arg(long, default_value_t = 80)]
        max_files: usize,
        /// Milliseconds between kills; slow the child so kills land mid-run.
        #[arg(long, default_value_t = 150)]
        interval_ms: u64,
    },
}

fn make_model(b: Backend, trace: &Trace) -> Result<Box<dyn Model>> {
    // AUDITOR_SLOW_MS adds latency inside each model call (after the trace line is
    // written), so crash demos interrupt calls that are "in flight".
    let ms: u64 = std::env::var("AUDITOR_SLOW_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    Ok(match b {
        Backend::Mock => Box::new(Traced {
            inner: Slow { inner: Mock, ms },
            trace: trace.clone(),
        }),
        Backend::Openai => Box::new(Traced {
            inner: Slow {
                inner: OpenAi::from_env()?,
                ms,
            },
            trace: trace.clone(),
        }),
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let j =
        Journal::<Audit>::open(&cli.db).with_context(|| format!("opening {}", cli.db.display()))?;

    match cli.cmd {
        Cmd::Run {
            repo,
            branch,
            budget,
            max_files,
            model,
            trace,
            verbose,
        } => {
            let b = j.branch(branch)?;
            if b.head()? == 0 {
                let repo = repo.ok_or_else(|| anyhow!("fresh journal: --repo is required"))?;
                let repo = repo.canonicalize()?;
                let files = audit::list_files(&repo, max_files);
                eprintln!(
                    "init: {} files under {}, budget {budget}",
                    files.len(),
                    repo.display()
                );
                b.apply(Delta::Init {
                    root: repo.to_string_lossy().into_owned(),
                    files,
                    budget,
                })?;
            }
            let trace = Trace::new(trace);
            let model = make_model(model, &trace)?;
            let r = agent::run(&b, model.as_ref(), &trace, verbose)?;
            let st = b.state()?;
            println!("{}", agent::describe(&st));
            println!(
                "model: {}  steps this run: {}  wall: {} ms  head seq: {}",
                model.name(),
                r.steps_this_run,
                r.wall_ms,
                b.head()?
            );
            print_metrics(&j, branch)?;
        }

        Cmd::Status => {
            for m in j.branches()? {
                let b = j.branch(m.id)?;
                let st = b.state()?;
                println!(
                    "branch {} {:<10} parent={:<4} fork_seq={:<5} head={:<5} params={}",
                    m.id,
                    m.name.clone().unwrap_or_default(),
                    m.parent.map(|p| p.to_string()).unwrap_or("-".into()),
                    m.fork_seq,
                    m.head,
                    m.params
                );
                print!("  {}", agent::describe(&st));
            }
        }

        Cmd::At { seq, branch, notes } => {
            let b = j.branch(branch)?;
            let st = b.at(seq)?;
            print!("{}", agent::describe(&st));
            if notes {
                println!("--- notes at seq {seq} ---");
                for n in &st.notes {
                    println!("{n}");
                }
            }
        }

        Cmd::History { branch, from, to } => {
            let b = j.branch(branch)?;
            for r in b.history(from, to)? {
                let what = match &r.event {
                    EventKind::Delta { delta } => {
                        let d: Delta = serde_json::from_value(delta.clone())?;
                        match d {
                            Delta::Init { files, budget, .. } => {
                                format!("init {} files budget {budget}", files.len())
                            }
                            Delta::SetBudget(b) => format!("set budget {b}"),
                            Delta::Analyzed { path, .. } => format!("analyzed {path}"),
                            Delta::Reported { report, .. } => {
                                format!("reported ({} bytes)", report.len())
                            }
                        }
                    }
                    EventKind::Effect {
                        key_hint, result, ..
                    } => {
                        format!(
                            "effect {} -> {} bytes",
                            key_hint.clone().unwrap_or_default(),
                            result.to_string().len()
                        )
                    }
                    EventKind::Compact { .. } => "compact".to_string(),
                    EventKind::Observe { name, value } => format!("observe {name} = {value}"),
                };
                println!(
                    "{:>5} b{:<3} {:<8} {} {}",
                    r.seq,
                    r.branch,
                    r.event.name(),
                    &lithify::hex(&r.hash)[..12],
                    what
                );
            }
        }

        Cmd::Fork { at, budget, branch } => {
            let base = j.branch(branch)?;
            let f = base.fork_at(at, None, serde_json::json!({"budget": budget}))?;
            f.apply(Delta::SetBudget(budget))?;
            println!(
                "branch {} forked from {} at seq {at} with budget {budget}",
                f.id(),
                branch
            );
            println!(
                "run it with: auditor --db {} run --branch {}",
                cli.db.display(),
                f.id()
            );
        }

        Cmd::Sweep {
            at,
            budgets,
            branch,
            model,
            trace,
        } => {
            let trace = Trace::new(trace);
            let model = make_model(model, &trace)?;
            let forks = j.sweep(
                branch,
                at,
                budgets.iter().map(|b| serde_json::json!({"budget": b})),
            )?;
            for (f, budget) in forks.iter().zip(&budgets) {
                f.apply(Delta::SetBudget(*budget))?;
                let r = agent::run(f, model.as_ref(), &trace, false)?;
                let st = f.state()?;
                eprintln!(
                    "branch {} budget {budget:>6}: {} steps, {} compactions, {} findings in report, {} ms",
                    f.id(),
                    r.steps_this_run,
                    st.compactions,
                    st.findings_in_report().len(),
                    r.wall_ms
                );
            }
            print!("{}", Journal::<Audit>::matrix_csv(&j.matrix()?));
        }

        Cmd::Matrix => print!("{}", Journal::<Audit>::matrix_csv(&j.matrix()?)),

        Cmd::Report { branch } => {
            let st = j.branch(branch)?.state()?;
            println!("{}", st.report.unwrap_or_else(|| "(no report yet)".into()));
        }

        Cmd::CrashDemo {
            repo,
            kills,
            budget,
            max_files,
            interval_ms,
        } => crash_demo(&cli.db, &repo, kills, budget, max_files, interval_ms)?,
    }
    Ok(())
}

fn print_metrics(j: &Journal<Audit>, branch: u64) -> Result<()> {
    let rows = j.matrix()?;
    let mine: Vec<_> = rows.iter().filter(|r| r.branch == branch).collect();
    if !mine.is_empty() {
        let s: Vec<String> = mine
            .iter()
            .map(|r| format!("{}={}", r.metric, r.value))
            .collect();
        println!("metrics: {}", s.join("  "));
    }
    Ok(())
}

fn crash_demo(
    db: &Path,
    repo: &Path,
    kills: u32,
    budget: usize,
    max_files: usize,
    interval_ms: u64,
) -> Result<()> {
    let exe = std::env::current_exe()?;
    let trace = db.with_extension("trace");
    let _ = std::fs::remove_file(&trace);
    let _ = std::fs::remove_file(db);

    let spawn = || -> Result<std::process::Child> {
        Ok(Command::new(&exe)
            .arg("--db")
            .arg(db)
            .arg("run")
            .arg("--repo")
            .arg(repo)
            .arg("--budget")
            .arg(budget.to_string())
            .arg("--max-files")
            .arg(max_files.to_string())
            .arg("--trace")
            .arg(&trace)
            .env("AUDITOR_SLOW_MS", "20")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?)
    };

    let mut rng_state: u64 = 0x9E3779B97F4A7C15;
    let mut rand_ms = |max: u64| {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state % max
    };

    for k in 1..=kills {
        let mut child = spawn()?;
        let wait = interval_ms + rand_ms(interval_ms);
        std::thread::sleep(Duration::from_millis(wait));
        child.kill()?; // SIGKILL
        child.wait()?;
        let j = Journal::<Audit>::open(db)?;
        let b = j.root();
        let st = b.state()?;
        println!(
            "kill {k}: SIGKILL after {wait} ms -> journal head {} ({} files analysed, {} compactions)",
            b.head()?,
            st.done.len(),
            st.compactions
        );
    }

    println!("resuming to completion in-process...");
    let j = Journal::<Audit>::open(db)?;
    let b = j.root();
    let tr = Trace::new(Some(trace.clone()));
    let model = Traced {
        inner: Slow { inner: Mock, ms: 0 },
        trace: tr.clone(),
    };
    let r = agent::run(&b, &model, &tr, false)?;
    let st = b.state()?;
    println!(
        "done: {} files, {} compactions, {} model calls recorded in state, {} steps in the final run, head seq {}",
        st.done.len(),
        st.compactions,
        st.model_calls,
        r.steps_this_run,
        b.head()?
    );

    // The outside world: how many times was each call actually made?
    let txt = std::fs::read_to_string(&trace).unwrap_or_default();
    report_trace(&txt, kills + 1);
    print_metrics(&j, 0)?;
    Ok(())
}

/// Summarise a trace file: distinct calls, repeats, split by model vs tool.
pub fn report_trace(txt: &str, lifetimes: u32) {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for l in txt.lines() {
        *counts.entry(l).or_default() += 1;
    }
    let total: usize = counts.values().sum();
    let model_dups: Vec<_> = counts
        .iter()
        .filter(|(k, c)| k.starts_with("model") && **c > 1)
        .map(|(k, c)| (*k, *c))
        .collect();
    let tool_dups: Vec<_> = counts
        .iter()
        .filter(|(k, c)| k.starts_with("tool") && **c > 1)
        .map(|(k, c)| (*k, *c))
        .collect();
    let extra = |v: &[(&str, usize)]| v.iter().map(|(_, c)| c - 1).sum::<usize>();
    println!(
        "trace: {} invocations for {} distinct calls across {} process lifetimes",
        total,
        counts.len(),
        lifetimes
    );
    println!(
        "re-executed: {} model call(s), {} tool call(s)",
        extra(&model_dups),
        extra(&tool_dups)
    );
    let mut names: Vec<&str> = model_dups.iter().map(|(k, _)| *k).collect();
    names.sort();
    if !names.is_empty() {
        println!("  model calls run more than once: {names:?}");
    }
}
