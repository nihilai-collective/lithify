# The same agent, twice: lithify vs LangGraph + SqliteSaver

Two implementations of one agent, with one deterministic mock model, run against one
corpus (the `redb` 4.3.0 source tree, 74 files), so that the only thing that differs is
the persistence layer underneath.

- `auditor/` — Rust, on lithify.
- `compare/langgraph/auditor.py` — Python, on LangGraph 1.2 with `SqliteSaver`
  (`langgraph-checkpoint-sqlite` 3.1).

The agent walks a source tree. For each file it performs one ReAct-style step: read
the file (tool), ask the model which patterns to count (model), count each pattern
(tools), write a note (model). Notes accumulate in a token-budgeted context; when the
budget is exceeded the model compacts the notes into one summary, dropping the least
severe findings. At the end the model writes a report from whatever notes survived.
*Recall* is the fraction of ground-truth findings that made it into the report.

Everything below was produced by running the two binaries as described; the commands
are in `scripts/demo.sh` and `scripts/compare.sh`.

## 1. Both do the job, identically

Same corpus, budget 2000, 74 files:

|                       | lithify | LangGraph |
|-----------------------|--------:|----------:|
| files analysed        | 74      | 74        |
| model calls           | 151     | 151       |
| compactions           | 2       | 2         |
| findings in report    | 83      | 83        |
| recall / precision    | 1.0 / 1.0 | 1.0 / 1.0 |
| wall time (mock model)| 405 ms  | 172 ms    |

The wall-time gap is real and worth understanding: lithify commits every event with an
fsync (755 commits here, roughly 0.5 ms each), LangGraph writes one checkpoint per node
(79 here). Per-event durability costs about 10 fsyncs per file in this agent. Against
real model latency it vanishes; against a mock it does not. `Options::lazy_deltas`
drops the fsync for deltas and observations only; effects stay immediate.

## 2. Sweep: recall as a function of budget

Fork at the point where 19 files had been analysed (lithify seq 191, LangGraph
checkpoint 19), one fork per budget, each run to completion:

| budget | compactions | recall (lithify) | recall (LangGraph) |
|-------:|------------:|-----------------:|-------------------:|
| 300    | 19–20       | 0.169            | 0.169              |
| 600    | 10          | 0.349            | 0.349              |
| 1200   | 4           | 0.723            | 0.723              |
| 2400   | 1           | 1.000            | 1.000              |
| 4800   | 0           | 1.000            | 1.000              |

Identical curves, as they must be with a deterministic model. This is the
falsifiable-experiment shape the white paper argues for: budget is the independent
variable, recall is the measured one, and the prefix (19 files of reads, plans, counts
and notes) was paid for once and shared by all five forks in both systems.

Where they differ is in how the sweep is expressed and read back.

lithify: `journal.sweep(0, 191, budgets)` creates five branches as five rows; each fork
inherits the parent's *effects*, not only its state, so if a fork re-requests
`read:src/lib.rs` it gets the journaled bytes without touching the disk;
`journal.matrix()` returns every observation across every branch joined to the
branch's merged parameters, and `matrix_csv` is the export.

LangGraph: forking is `update_state` on a past checkpoint, which creates a new
checkpoint *in the same thread*; to keep the sweep legible the baseline copies the
checkpoint's state into a fresh thread per budget. The checkpointer API has no
cross-thread query, so `matrix` reads the sqlite file directly to enumerate threads.
Both work. One is a feature, the other is a workaround.

## 3. Crash: what gets redone

Each system runs its agent as a child process with 20 ms of latency injected into
every model call (after the call is logged, as a real API call is billed when sent).
The parent SIGKILLs the child at a random moment, eight times, then finishes the run
in-process. A trace file outside both systems records every model and tool invocation.

Three trials each (8 kills; LangGraph's first kill always landed during Python
startup, so it saw 7 effective kills):

| trial | system    | model calls re-executed | tool calls re-executed |
|------:|-----------|------------------------:|-----------------------:|
| 1     | lithify   | 7                       | 0                      |
| 2     | lithify   | 4                       | 0                      |
| 3     | lithify   | 8                       | 0                      |
| 1     | LangGraph | 8                       | 28                     |
| 2     | LangGraph | 7                       | 22                     |
| 3     | LangGraph | 5                       | 15                     |

Read this carefully, because the headline is not "lithify never re-executes".

Both systems re-execute roughly one model call per kill: the call that was in flight.
That is a floor no client-side journal can get under. A model call that has been sent
but whose response has not been committed is indistinguishable, after a crash, from
one that was never sent. Only an idempotency key honoured by the provider, or a
transactional outbox on their side, removes it.

What differs is the blast radius. lithify's unit of durability is the effect: the
in-flight call is redone and nothing else is, because the read, the plan and the
counts for that file were each committed the moment they returned. LangGraph's unit
of durability (in the graph API) is the node: the checkpoint is written when the node
returns, so a kill during the second model call of a step redoes the read, the plan,
and every count in that step, and sometimes the first model call too. With one model
call and one tool per step the two systems would look the same; with six tool calls
per step, as here, the difference is twenty-odd tool calls per eight kills, and it
grows linearly with the number of effects per step.

To be fair to LangGraph: its *functional* API (`@entrypoint` / `@task`) persists task
results individually and does not re-run completed tasks on resume, which is
effect-level granularity. The comparison above is against the graph API, which is what
the documentation leads with and what most LangGraph agents are written in. An
equivalent result against the functional API would be expected to narrow the tool
column to zero; the model-call column would not move.

## 4. Time travel

lithify: `auditor at --seq 40 --notes` materialises the state as of event 40 (nearest
snapshot plus replay) and prints the notes the model had at that moment; `auditor
history` lists every event, with its content hash, resolved through the fork chain.
Granularity is one event: you can see the state between the `read` and the `plan` of
a file.

LangGraph: `graph.get_state_history(config)` returns one snapshot per checkpoint, and
`auditor.py at --index 13 --notes` prints the thirteenth. Granularity is one node.
There is no record of the individual tool and model calls inside a node, only the
state before and after.

## 5. Shape, not size

Lines of code, non-blank non-comment, for the agent itself (not the library):

| | lines |
|-|------:|
| Rust agent (`auditor/src/*.rs`) | 948 |
| of which `agent.rs`, the loop that touches lithify | 148 |
| Python agent (`compare/langgraph/auditor.py`) | 374 |
| lithify library (`lithify/src/*.rs`) | 818 |

Python is shorter; it usually is, and Rust pays for the CLI, two model backends and
the type definitions in full. The number that matters is 148: the agent loop that
knows about durability is a hundred and fifty lines, and three of its calls
(`effect`, `apply`, `compact`) are the entire crash-safety story.

## 6. Summary table

| capability | lithify | LangGraph graph API + SqliteSaver |
|---|---|---|
| resume after `kill -9` | yes | yes |
| unit of durability | one effect / delta | one node |
| in-flight model call redone on crash | yes (floor) | yes (floor) |
| other work in the step redone on crash | no | yes |
| time travel granularity | event | node |
| fork cost | one row (≈60 µs in memory) | one checkpoint write |
| forks share prefix | state **and** effects | state |
| cross-branch results query | `matrix()` | read sqlite by hand |
| content-addressed history | blake3 chain, `lookup_hash` | checkpoint ids |
| budget / compaction primitive | `Compactor`, `Compact` event, snapshot | none (user code) |
| schema of state | consumer's own types, JSON | TypedDict, pickle/JSON |
| runtime | library, pure Rust, one file | Python, library |
| per-event fsync cost | yes (optional for deltas) | per node |

## Reproducing

```sh
cargo build --release
pip install langgraph langgraph-checkpoint-sqlite
REPO=$(ls -d ~/.cargo/registry/src/*/redb-4.3.0)   # or any source tree
scripts/compare.sh "$REPO"
```
