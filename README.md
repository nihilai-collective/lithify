# lithify

*Sediment in, rock out.*

A journaled, forkable, compactable context store for long-running agents, in pure
Rust on [`redb`](https://github.com/cberner/redb).

You write a plain `State` type and a `Delta` enum. lithify gives you:

- **durability** — every delta and every side-effect result is journaled before it
  is used; `kill -9` the process and resume at the exact step, without re-executing
  anything whose result was already committed;
- **time travel** — `branch.at(seq)` is the state exactly as it was;
- **forks** — `branch.fork_at(seq)` is one row, O(1), and shares the prefix
  *including its effects*, so a sweep from step 37 pays for steps 1–37 once;
- **compaction** — a `Compactor` replaces the state with a smaller one, journaled and
  snapshotted; what the agent knew before it forgot is still addressable;
- **a results matrix** — `observe(metric, value)` across branches, joined to each
  branch's parameters, exported as CSV.

The model is git's: an append-only log of content-addressed events, branch refs that
point at a sequence number, snapshots as trees, compaction as squash, replay as
checkout.

```rust
use lithify::{Journal, State};
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Serialize, Deserialize)]
struct Ctx { messages: Vec<String> }

#[derive(Serialize, Deserialize)]
enum Delta { Say(String) }

impl State for Ctx {
    type Delta = Delta;
    fn apply(&mut self, d: &Delta) { match d { Delta::Say(s) => self.messages.push(s.clone()) } }
    fn size(&self) -> usize { self.messages.iter().map(|m| m.len()).sum() }
}

fn main() -> lithify::Result<()> {
    let journal = Journal::<Ctx>::open("run.redb")?;        // resumes if it exists
    let ctx = journal.root();
    let reply: String = ctx.effect("llm:step-1", || call_model("hello"))?;  // journaled; never re-run
    ctx.apply(Delta::Say(reply))?;
    let then = ctx.at(1)?;                                    // time travel
    let variant = ctx.fork_at(1, None, serde_json::json!({"temperature": 0.2}))?;
    Ok(())
}
```

## What's here

| path | what |
|---|---|
| `lithify/` | the library (six redb tables, ~800 lines) and its tests, including a SIGKILL test |
| `auditor/` | an example agent: audits a source tree with a budgeted, compacting context; mock model and any OpenAI-compatible endpoint |
| `compare/langgraph/` | the same agent on LangGraph + SqliteSaver, for a fair comparison |
| `docs/whitepaper.md` | motivation, theory, design, prior art, falsifiable hypotheses |
| `docs/comparison.md` | measured results: lithify vs LangGraph on identical runs |
| `scripts/demo.sh` | the pitch in one minute |
| `scripts/compare.sh` | reproduce the comparison |

## Try it

```sh
cargo test                       # includes a kill -9 / resume test
scripts/demo.sh /path/to/some/source/tree
```

The demo SIGKILLs the agent four times and finishes the run, shows the state as of
event 40, lists the first dozen events with their hashes, and forks a five-budget
sweep from nineteen files in, printing recall per budget.

To use a real model: `--model openai` with `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
`OPENAI_MODEL` (any `/v1/chat/completions` endpoint).

## Status

Proof of concept. The API will change. See the roadmap at the end of the white paper.

## License

MIT or Apache-2.0, at your option.
