# lithify: a journaled, forkable, compactable context store for long-running agents

*Lothrop, with Claude. October 2026. Draft 0.1.*

## Abstract

An LLM agent's context is state. It is appended to on every step, it must survive the
process that holds it, it must be inspectable after the fact, it must be forkable so
that alternatives can be tried from a shared prefix, and it must be compacted when it
outgrows its budget. Today each of those is solved separately and badly, by hand, in
every agent framework, and the solutions do not compose. We argue that they are one
problem with one well-understood shape, the append-only log of content-addressed
deltas with snapshots and branch references (git's data model), and that a small Rust
library built around that shape makes durability, time travel, forking and
compaction fall out of a single mechanism. We describe lithify, a proof of concept on
`redb`, give the theoretical reasons to expect the design to hold up (event sourcing,
structural sharing, journaled effects, and, for the multi-agent extension, patch
theory and the CALM theorem), compare it to the systems that solve adjacent problems,
and state the hypotheses it makes falsifiable. Two of those hypotheses are already
tested against a LangGraph baseline on an identical agent; the results are reported.

## 1. Motivation

### 1.1 The hairy problems

Anyone who has run an agent for longer than a coffee break has hit four problems.

The process dies. An agent forty tool calls into a task is killed by an OOM, a deploy,
a laptop lid. Resuming means having serialised enough state to reconstruct where it
was, which in practice means a hand-written checkpoint format, written once, wrong at
the edges, and silently re-executing whatever happened between the last checkpoint
and the crash: an email sent twice, a file written twice, a model call billed twice.

Nobody knows what the model saw. When an agent does something strange at step 37, the
question "what exactly was in its context at that moment?" is answered by grepping
logs and reconstructing by hand, if the logs are even complete. The rendered prompt is
almost never recorded; the state that produced it is recorded, if at all, as a
pickle.

Experiments are expensive. The interesting question is almost always counterfactual:
*had the instruction at step 37 been different, what would have happened?* Answering
it means re-running steps 1 through 36, paying for every model call again, and
copying mutable state carefully enough that the two runs do not interfere. So the
experiment does not get run, or it gets run once, with n = 1.

The context fills up. Every agent has an ad hoc truncation strategy: drop the oldest
messages, summarise when a threshold is crossed, keep the system prompt. The strategy
is lossy by construction, and what was lost is gone. There is no way to ask what the
agent knew before it forgot.

### 1.2 The inspiring idea

The idea that started this was not about agents. It was about Rust library design:
build the *seams* first. Rust's most successful crates are seams rather than
implementations (`serde::Serialize`, `tower::Service`, `std::io::Read`,
`embedded-hal`), narrow trait boundaries that let independently written pieces
compose. The original list of seams worth building was: how a context structure
persists to a data layer; how shared mutable state is read and written; how a
structure is distributed to workers that each need their own copy; how an endpoint
gets read-only, read-write or write-only access.

Two observations turned that list into this project. First, the seams compose only
after one picks a *unit of state ownership*. Orleans, Akka, Erlang/OTP and
Cloudflare's Durable Objects all made the same choice (a single-threaded actor with
colocated state) and then persistence became journal-plus-snapshot per actor, sharing
disappeared into message passing, and distribution became placement. Without that
choice the seams are a grab bag. Second, if persistence is built as a *log of deltas*
rather than as `save()` or as a repository, the other seams fall out of it: the same
delta stream that persists also replicates; readers hold immutable snapshots while a
single writer appends; a fork is a pointer into the log. Datomic is the clearest
existing demonstration of the whole package, and Jon Gjengset's `left-right` is the
in-process miniature.

An LLM agent's context turns out to be the ideal first consumer. It is append-mostly,
it has a natural notion of compaction (summarisation) that is exactly a snapshot, and
its consumers already suffer the four problems above acutely enough to care. The
multi-agent extension (agents compacting into a shared corpus and pulling relevant
knowledge back out) is where the project wants to go; durability is where it has to
start, because everything else is built on it.

The name is geological. Sediment is deposited layer on layer; under pressure it
compacts and lithifies into rock, which keeps the layers. That is the whole design.

## 2. The idea

### 2.1 Context as a journal

The consumer defines a plain `State` type and a `Delta` enum, and one function:
`apply(&mut State, &Delta)`. lithify never mutates the state except through `apply`,
and never stores it except as JSON. Deltas are the unit of truth; the state is a cache.

Every change to the context is an *event* appended to a *branch*. There are four kinds:

- `Delta(d)`: a consumer delta, applied on replay.
- `Effect { key, result }`: the recorded result of a side effect (a model call, a
  tool call, a file read). On replay the result is read, not re-executed.
- `Compact(state)`: the state was replaced wholesale by a compactor; a snapshot is
  written at the same sequence number.
- `Observe { name, value }`: a measurement for the results matrix; no effect on state.

Events are content-addressed: each carries `blake3(parent_hash ‖ canonical(event))`,
so a branch head's hash identifies everything that has ever happened on it, and two
branches that did the same things have the same hash.

### 2.2 The git model

A branch is a row: `(id, parent, fork_seq, head, head_hash, params)`. Its *resolved
history* is its own events from `fork_seq + 1` to `head`, preceded by its parent's
resolved history up to `fork_seq`, recursively. Forking at sequence `s` is inserting
one row with `fork_seq = s`; it is O(1) and copies nothing. Materialising the state
at `s` is finding the nearest snapshot at or before `s` in the resolved history and
replaying the events after it. Compaction is a squash with the pre-squash history
retained. Time travel is checkout.

The correspondence is close enough that the vocabulary is borrowed deliberately:
commits, trees, refs, squash, checkout. Jujutsu's operation log, which applies the
same model to the repository's own history, was a confirming example.

### 2.3 Effects and the one honest caveat

`branch.effect(key, || do_the_thing())` looks up `blake3(key)` in the branch's
resolved history. If an effect with that key exists, its stored result is returned
and the closure is not called. Otherwise the closure runs and its result is committed,
with immediate durability, before being returned. A fork therefore inherits every
effect of its prefix: a parameter sweep forked at step 37 shares steps 1 through 37's
model calls, tool calls and file reads, and pays only for its own tail.

The caveat: the journal and the outside world are not one transaction. A process
killed after a model call was *sent* but before its response was *committed* will,
on resume, send the call again. This at-least-once window is the duration of the
call itself, and no client-side journal can close it; only an idempotency key
honoured by the provider, or a transactional outbox on their side, can. What the
journal does guarantee is that a result once committed is never re-executed, and
that the window is one effect wide, not one step wide. Section 8 measures this.

### 2.4 Compaction

A `Compactor` is a consumer-supplied function from a state to a smaller state. It is
handed the branch so that anything expensive or non-deterministic it does (asking the
model to summarise) can itself be journaled as an effect, which makes compaction
replayable and crash-safe. The result is appended as a `Compact` event with a
snapshot, so replay from that point is O(1) and the pre-compaction history remains
addressable. "What did the agent know before it forgot?" is `branch.at(seq - 1)`.

### 2.5 The matrix

`branch.observe("recall", 0.72)` appends an observation. `journal.matrix()` returns
every observation in the journal joined, by hand, to the merged parameters of its
branch (children override ancestors), and `matrix_csv` exports it. This is the
analysis off-ramp: redb is the system of record, DuckDB or Polars is where the pivot
tables live.

### 2.6 What it deliberately is not

lithify is not a database (it is a schema on top of one), not an actor runtime, not an
orchestrator, not a workflow engine, and not yet a multi-agent memory. It has no
opinion about models, prompts, tools or message formats. It is one seam, with two
backends (redb on disk and redb in memory), and a few hundred lines.

## 3. Theoretical background: why we expect it to work

### 3.1 The log is the truth

Event sourcing's central claim is that a system's state is a left fold over its
history: $S_n = \mathrm{apply}(S_{n-1}, d_n)$, $S_0 = \bot$. Everything else, the
materialised state, indexes, caches, summaries, is derived and can be rebuilt. The
claim is old (database write-ahead logs are the same idea) and its consequences are
well charted: audit for free, time travel for free, replication by shipping the log,
and the known tax of event schema evolution and replay cost. lithify pays the tax in
the usual ways: self-describing JSON payloads with `serde` defaults for evolution,
snapshots for replay cost.

The fold must be *deterministic* for replay to agree with the live run. That is why
effects exist as a separate event kind: the non-deterministic parts of a step are
captured as data, so that the function from history to state is pure. This is the
"sans-IO" discipline applied to agent state: the data structure never does IO; it
consumes the results of IO that happened elsewhere.

### 3.2 Structural sharing and the fork

A persistent (in the functional sense) data structure makes a new version by sharing
unchanged structure with the old one, so that versions are cheap and old versions
remain valid. The journal is such a structure at the granularity of events: a fork
shares its entire prefix by reference, and materialising any version is a function of
(nearest snapshot, replay distance), independent of how many other versions exist.
The fork cost measured in the test suite is about sixty microseconds in memory, one
row, and does not scale with history length.

This is the property that makes the parameter sweep an everyday operation instead of
a project. Twenty variants forked at step 37 cost twenty rows plus twenty tails.

### 3.3 Durable execution and the at-least-once floor

Restate, Temporal and Azure Durable Functions share a technique: journal the result
of every side effect under a stable key, and on recovery replay the function with the
journal supplying the results, so that completed effects are never re-done. lithify's
`effect` is that technique with the function removed: there is no re-execution of
code, only materialisation of state, so the journaled results are consulted by key
rather than by position.

The floor is the same in all of these systems and follows from a simple argument. Let
an effect have three phases: request sent, response received, result committed. A
crash between the first and the third leaves no committed evidence that the request
was sent. On recovery the system must either skip the effect (losing it if it was in
fact not sent) or redo it (duplicating it if it was). With no information to
distinguish the cases, redo is the only safe choice. Exactly-once therefore requires
cooperation from the other side of the effect. What a client-side journal controls is
the *width* of the window: lithify's is one effect; a node-granular checkpointer's is
one node.

### 3.4 Compaction as a lossy channel

Compaction maps a context of size $n$ tokens to one of size at most $b$. For $b < n$
it is lossy, and the natural measure of the loss is task-relative: what fraction of
the facts the agent will later need are still recoverable from the compacted context?
In the proof-of-concept agent that fraction is *recall* of audit findings in the
final report.

Rate–distortion theory says that for a fixed source and a fixed fidelity measure, the
minimum achievable distortion is a non-increasing function of the rate. Compaction
with a budget is a rate constraint, and recall is one minus a distortion. So the first
prediction the framework makes is monotonicity: recall as a function of budget should
be non-decreasing, and should reach one at the budget where no compaction is needed.
This is trivially true for the mock compactor, which is a priority truncation; whether
it holds for an LLM compactor is an empirical question with a known failure mode
(summaries that drop structured facts while keeping prose). Section 7 states it as a
hypothesis.

The generational garbage collector is the right operational analogy, and it names
the parts: the agent's live context is the nursery, compaction promotes survivors,
the number of compactions an item has survived is its generation, and the interesting
engineering is in the write barrier, that is, what is allowed to cross.

### 3.5 Patch theory and the lattice question

The multi-agent extension needs several agents to merge their compacted knowledge
into a shared corpus. Whether that can be done without coordination is a property of
the delta type, and there is a precise theory of it.

A set of deltas with a composition operation forms a monoid acting on states. If
additionally merge is commutative, associative and idempotent (a join-semilattice)
then the state space is partially ordered, every finite set of concurrent updates has
a least upper bound, and replicas that have seen the same set of updates in any order
agree. That is the definition of a state-based CRDT (Shapiro et al. 2011); the
delta-state refinement (Almeida et al. 2018) is exactly the "ship deltas, merge
joins" shape of a compacting journal. If merge is only a monoid, order matters and a
single writer or a total order is required: a log.

The CALM theorem (Hellerstein and Alvaro 2020) is the general statement: a program
has a coordination-free, eventually consistent distributed implementation if and
only if it is *monotone*, meaning that growing the input never retracts an output.
Appending facts is monotone. Deduplication-with-supersession, forgetting, and "that
was wrong" are not. CALM therefore tells the multi-agent design exactly where its
barriers go: monotone accumulation flows between agents freely, and the non-monotone
operations happen only at compaction boundaries, which are the coordination points.
The question "when does communication happen?" is answered by the shape of the
operation rather than by a policy.

Patch theory makes the same structure visible from the other direction. Mimram and
Di Giusto (2013) give a categorical theory of patches in which repository states are
objects and patches are morphisms, and derive the merge of independent patches as a
pushout. Angiuli, Morehouse, Licata and Harper (2014) redo this in homotopy type
theory, where states are points of a higher inductive type, patches are paths, and
the laws patches must satisfy (that two independent patches commute, say) are 2-paths
between paths. In that setting a journal is a path, a fork is two paths from a common
point, and a merge is a filler. The lattice question of this section is whether the
filler always exists.

### 3.6 Fixed points and saturation

If the corpus is a bounded lattice and compaction is monotone on it, then by
Knaster–Tarski the compaction operator has a least fixed point, and by Kleene's
theorem iterating it from the bottom reaches that fixed point. A converged corpus is
a fixed point of its own compaction. The practical obstacle is that an LLM
summariser is neither monotone nor idempotent: summarising a summary drifts, and
merging A's view of B's view of A's facts produces an echo rather than knowledge. The
design response, planned but not built, is to separate the authoritative corpus (a
set of claims with provenance, append-only, with retraction as a kind of claim: the
Datomic model) from the rendered context the model sees, so that nondeterminism
lives only in the view and the lattice stays a lattice.

For a long-running system "converged" is the wrong target anyway; "stationary" is
the right one. The useful signal is the survivor rate: of the deltas arriving from
agents, what fraction is novel after deduplication? Estimating how far a corpus is
from saturation is the species-estimation problem of ecology, and its standard tools
transfer directly. Good–Turing (1953) estimates the probability that the next claim
is one never seen before as $f_1 / N$, the fraction of observations that are
singletons. Chao1 (1984) estimates the number of unseen classes as
$\hat{S} = S_{\mathrm{obs}} + f_1^2 / (2 f_2)$ from the singleton and doubleton
counts. These are cheap to compute on a provenance-tagged corpus and give the
multi-agent system a principled stopping criterion for exploration, which is the
closest thing to a goal state a long-running system can have.

## 4. Design

### 4.1 API

```rust
pub trait State: Serialize + DeserializeOwned + Default + Clone + Send + Sync + 'static {
    type Delta: Serialize + DeserializeOwned + Send + Sync + 'static;
    fn apply(&mut self, delta: &Self::Delta);
    fn size(&self) -> usize { 0 }
}

pub trait Compactor<S: State> {
    type Error;
    fn compact(&self, branch: &Branch<S>, state: &S) -> Result<S, Self::Error>;
}

impl<S: State> Journal<S> {
    fn open(path) / open_with(path, Options) / in_memory();
    fn root() -> Branch<S>;  fn branch(id);  fn branches();  fn children(id);
    fn chain(branch, upto) -> Vec<Segment>;        // the resolved history
    fn sweep(base, at, params) -> Vec<Branch<S>>;   // one fork per parameter set
    fn matrix() -> Vec<ObservationRow>;  fn matrix_csv(&rows);
    fn lookup_hash(&Hash) -> Option<(BranchId, Seq)>;
}

impl<S: State> Branch<S> {
    fn apply(delta) -> Seq;
    fn effect<T>(key, || -> Result<T>) -> Result<T>;   // + effect_async
    fn observe(name, value) -> Seq;
    fn compact(&impl Compactor<S>) -> Seq;
    fn snapshot() -> Seq;   fn over_budget(budget) -> bool;
    fn state() -> S;  fn with_state(|&S| ..);  fn at(seq) -> S;
    fn history(lo, hi) -> Vec<Record>;  fn full_history();
    fn fork(name, params);  fn fork_at(seq, name, params);
    fn head() -> Seq;  fn head_hash() -> Hash;  fn meta() -> BranchMeta;
}
```

### 4.2 Storage

Six redb tables. Every access is a point lookup or a prefix range; the only indexes
are the two that make effects and content addressing O(1).

```
branches   u64            -> BranchMeta (json)
children   u64            ->> u64                 (multimap: parent -> children)
events     (u64, u64)     -> Record (json)        (branch, seq)
effects    (u64, [u8;32]) -> u64                  (branch, blake3(key)) -> seq
snapshots  (u64, u64)     -> State (json)
by_hash    [u8;32]        -> (u64, u64)
```

`chain(branch, upto)` walks parents and returns the list of `(branch, lo, hi)`
segments covering `1..=upto`, root first. Materialisation searches the segments
newest-first for a snapshot, then replays forward. Effect lookup searches the
segments newest-first for the key and accepts a hit only if its seq lies within the
segment, so an effect recorded on the parent *after* the fork point is correctly
invisible to the child. This walk is the one piece of real logic in the library; it
is what a SQL `WITH RECURSIVE` would hide, and the design keeps it visible on purpose.

### 4.3 Durability

redb offers `Durability::Immediate` (fsync on commit) and `None` (no fsync; lost on
crash unless a later immediate commit lands). lithify commits effects, compactions,
snapshots and forks immediately, always, because the invariant "a result observed is
a result journaled" is the point. Deltas and observations are immediate by default
and may be made lazy with `Options::lazy_deltas`, which trades a possible loss of
trailing deltas on power failure (not on `kill -9`) for fewer fsyncs. The measured
cost of the default is about half a millisecond per event; the proof-of-concept
agent spends ten events per file. Against model latency this is noise; against a
mock it is the dominant term, and the comparison reports it.

### 4.4 Why redb

A research library should be pure Rust: no C toolchain, no system SQLite, one crate
to vendor, and a file that `cargo run` can create anywhere. redb is stable, ACID,
single-writer with MVCC readers, and ships an in-memory backend, so the test backend
and the disk backend are one code path.

The deeper reason is discipline. redb has no query language and no secondary indexes
you did not build. Every join in this design is done by hand, map-side, in Rust. The
first time a "query" would need a shuffle, you have to write the shuffle, and that
friction is the signal that something in the data model is wrong. The analysis layer
is an export, not a query, and lives in DuckDB or Polars where shuffles belong.

### 4.5 Concurrency

redb's transaction model is the read/write seam from the original list, for free: one
exclusive writer, unlimited concurrent readers, each read transaction a consistent
snapshot. Rendering the context as of step 37 while the writer appends step 41 is a
read transaction and no code on lithify's side. Branch handles cache the head state
and validate the cache against the journal head on every read, so stale handles are
safe and merely slower.

## 5. Existing systems

**LangGraph checkpointers** (SqliteSaver, PostgresSaver). The closest thing in
widespread use. A checkpoint is written per node (per superstep); `get_state_history`
gives time travel at node granularity; `update_state` on a past checkpoint forks
within a thread. The functional API's `@task` persists individual task results and
does not re-run them on resume, which is effect-level granularity; the graph API,
which the documentation leads with, has node-level granularity. There is no
compaction primitive, no content addressing, no cross-thread query, and the state
schema is a TypedDict with reducers. Section 8 compares against it directly.

**Restate, Temporal, Azure Durable Functions.** Durable execution: journal every side
effect's result, replay the function with the journal supplying results. Restate is
itself written in Rust and has a Rust SDK. These are servers with their own
operational footprint and their unit of replay is a handler invocation; they are the
right choice when the workflow is the product. lithify is the same idea as a library
with no server and no re-execution of code, which is appropriate when the state is
the product and the "workflow" is `loop { step }`.

**Event-sourcing crates** (`eventsourced`, `evento`, `cqrs-es`, `eventually`,
`coerce`'s persistent actors). All provide a journal; most impose an aggregate /
command / event vocabulary from domain-driven design that an agent loop does not
need, and none provide forks, effects or compaction as primitives. `eventsourced`'s
release history (forty-odd releases, half of them breaking) is a fair indication of
how hard this seam is to pin down.

**Datomic.** A single transactor, an immutable log of facts, snapshots shipped to
peers who each hold their own local copy, reads scaling horizontally, retraction as a
fact. This is the conceptual target for the multi-agent extension, and the closest
realisation of the full original idea; it is also a JVM product with a licence.

**Cloudflare Durable Objects, Orleans, Akka Persistence, Erlang/OTP.** The actor
answer to the unit-of-ownership question. Persistence is per actor (journal plus
snapshot in Akka's case), sharing is message passing, distribution is placement.
lithify takes the per-actor journal and leaves the runtime to the consumer.

**Agent memory systems** (MemGPT/Letta, mem0, Zep, LangGraph's memory store). Solve
the "what should the agent remember across sessions" problem with retrieval over
summaries. They are about *what* to keep; lithify is about *how* to keep it such that
nothing is irrecoverable. They are complementary and a memory system could be a
`Compactor`.

**CRDT libraries** (Automerge, Yjs/yrs, Loro) and **Hydro**. Prior art for the
multi-agent extension's merge and distribution respectively; Hydro, from the CALM
authors, is the most ambitious attempt at a distribution seam in Rust (write one
program, the compiler partitions it). Neither addresses the single-agent durability
problem this paper is about.

**`left-right`, `arc-swap`, `im`/`imbl`.** In-process read/write splits and
persistent structures. They are what one reaches for if the journal must live in
memory with many live branches; lithify defers them because materialising from a
snapshot is microseconds at agent scale.

## 6. The case for the library

The argument is not that any single capability is new. Journals, forks, effect
journaling and snapshots each exist somewhere. The argument is that they are *one
mechanism* and that treating them as one makes each of them nearly free given the
others.

Durability is the log. Time travel is materialisation at an arbitrary sequence, which
the log already requires for recovery. Forking is a row that points into the log.
Effect sharing across forks is the effect index consulted through the fork chain,
which recovery already consults. Compaction is a snapshot, which replay already uses.
The results matrix is an event kind plus a join. None of these added a new storage
concept; the whole library is six tables, and the agent loop that uses all of it is a
hundred and fifty lines, three calls of which (`effect`, `apply`, `compact`) carry
the durability story.

The second half of the argument is about what the library *refuses* to do. It does
not know about models, prompts or tools. It does not run anything. It does not
distribute anything. It does not query anything. Each refusal keeps a seam open for a
consumer to fill with their own choices, and keeps the library small enough to be
understood in an afternoon, which for a research tool is the whole value: you can
trust what you can read.

The third half is the one a sceptic should weigh most. A hello world that resumes
after `kill -9`, answers "what did it know at step 37", and forks a sweep from there
is twenty lines and always works, because none of it depends on a model behaving.
That is the serde test: `#[derive(Serialize)]` plus one line. If the hello world
needed more, the seam would be wrong.

## 7. Falsifiable hypotheses

Each hypothesis states what would refute it. Status as of this draft is given.

**H1 (replay equivalence).** For any sequence of applies, effects, compactions and
snapshots, the state materialised at every sequence number equals the state of a live
run that stopped there. *Refuted by* any divergence. *Status:* tested by a property
test over random op sequences (64 cases of up to 60 ops per run, every prefix
checked) and by the crash test; holds.

**H2 (effect idempotency).** An effect whose result is committed is never re-executed
on the same history or on any fork of it, across process restarts. *Refuted by* a
second execution in the trace. *Status:* tested directly and in the SIGKILL test
(six rounds, 400 steps); holds. The in-flight exception is as predicted by §3.3 and
is checked to be at most one per kill and only for the step in flight.

**H3 (fork cost).** Fork time is independent of history length and materialisation
time is bounded by replay distance from the nearest snapshot, not by history length.
*Refuted by* fork time growing with history or `at(s)` time growing with `s` when a
snapshot sits at `s`. *Status:* measured: ~57 µs per fork in memory over 2000 events
with 1000 forks; `at(snapshot)` 27 µs, `at(snapshot + 499)` 1.0 ms. Holds for the
in-memory backend; the on-disk fork additionally pays one fsync.

**H4 (crash blast radius).** Under random SIGKILL, the number of model calls
re-executed is approximately one per kill for lithify and for a node-granular
checkpointer alike (the floor), but the number of *other* effects re-executed is
zero for lithify and proportional to effects-per-step for the node-granular system.
*Refuted by* lithify re-executing committed tool calls, or by the baseline not
re-executing them. *Status:* tested against LangGraph + SqliteSaver, three trials of
eight kills each: lithify 4–8 model, 0 tool re-executions; LangGraph 5–8 model, 15–28
tool re-executions. Holds. Section 8 and `docs/comparison.md`.

**H5 (compaction monotonicity).** For a fixed corpus and file order, task recall is a
non-decreasing function of the context budget, reaching 1.0 at the budget where no
compaction occurs. *Refuted by* a budget increase lowering recall. *Status:* holds for
the deterministic mock compactor (0.17, 0.35, 0.72, 1.0, 1.0 at budgets 300, 600,
1200, 2400, 4800) in both implementations, which establishes the harness. The
interesting test is with an LLM compactor, where a violation (a larger budget
producing a worse summary) is plausible and would be a finding about the compactor,
not the journal. *Not yet run.*

**H6 (sweep sharing).** A sweep of $k$ forks at step $s$ performs exactly the model
and tool calls of $k$ tails plus one prefix, that is, no fork re-executes an effect
with a key journaled at or before $s$. *Refuted by* a prefix effect appearing more
than once in the trace. *Status:* follows from H2 and is observed in the sweep runs;
a dedicated trace assertion is a to-do.

**H7 (hello-world size).** An agent that resumes after `kill -9`, exposes `at(seq)`,
and forks a sweep needs no more than one `State` impl, one `Delta` enum and a loop
calling `effect`, `apply`, `compact`. *Refuted by* a consumer needing to touch
storage, serialisation or branch bookkeeping to get those properties. *Status:* the
proof-of-concept agent's loop is 148 lines including the ReAct step and metrics;
holds for this consumer. A second, dissimilar consumer is the real test.

**H8 (multi-agent, future).** With a lattice-structured corpus and provenance, the
survivor rate of incoming deltas under continued exploration decreases toward a
floor, and the Chao1 estimate of unseen claims converges; with free-text summaries
merged by an LLM, it does not (echo). *Refuted by* the lattice corpus failing to
saturate or the free-text corpus saturating cleanly. *Status:* not built.

## 8. Results so far

Full detail is in `docs/comparison.md`; the headlines are these.

The two implementations of the auditor agent, one on lithify and one on LangGraph
with SqliteSaver, produce identical results on identical inputs (151 model calls, 2
compactions, 83 findings, recall 1.0 at budget 2000) and identical recall-vs-budget
curves in a five-point sweep forked nineteen files in. The harness is sound.

Under random SIGKILL with 20 ms of injected model latency, both systems redo the
model call that was in flight, about once per kill, which is the floor argued in
§3.3. lithify redoes nothing else. LangGraph's graph API redoes the rest of the node:
fifteen to twenty-eight tool calls per eight kills in a step with seven tool calls.

The cost of per-event durability is visible against a mock model: 405 ms versus
172 ms for a 74-file run, essentially 755 fsyncs versus 79. It is an honest number
and the reason `Options::lazy_deltas` exists.

Time travel in lithify is per event with content hashes; in LangGraph it is per
node. The sweep in lithify is one call and one query; in LangGraph it is a workaround
over `update_state` and a hand-written sqlite scan.

## 9. Limitations and risks

The multi-agent story is a plan, not a result, and it is where the research risk
lives: relevance, echo chambers, and compaction quality are all open. This paper
argues the durability layer is worth having regardless; a reader who disagrees should
weigh the library on §8 alone.

Event schema evolution is handled the way everyone handles it (self-describing
payloads, defaults) and will hurt the first time a `Delta` variant is renamed. A
versioned `Delta` with upcasting is the known fix and is not built.

The `State` is serialised whole at every snapshot and every compaction. For contexts
of megabytes this is wasteful and a structural-sharing representation (`imbl`,
`rkyv`) would be warranted. At agent scale it has not mattered.

Effect keys are the consumer's responsibility. Two calls with the same key on one
history see one execution; that is the feature, and it is also the footgun. A step
counter in the key is the usual discipline.

Content addressing uses `serde_json`'s field order as the canonical form. It is
canonical for a fixed set of Rust types and not across languages; a true canonical
JSON (RFC 8785) would be needed for cross-implementation hash agreement.

redb is single-process. Two processes opening one journal is undefined; the
multi-process story, if any, is a server, which this library is not.

The at-least-once floor is real and this paper has tried not to hide it behind the
word "durable". A consumer who needs exactly-once model calls needs an idempotency
key the provider honours, and `effect`'s key is the natural thing to send.

## 10. Roadmap

1. A `#[derive(Delta)]` for structs, generating a field-wise delta enum, once a
   second consumer exists to show what is common. Prior art: `struct-patch`,
   `serde-diff`, `dipa`.
2. An `Lattice` marker trait for delta types whose merge is a join, with a
   `merge(a, b)` on branches that is defined only when it is coordination-free, per
   §3.5.
3. The corpus: a claims-with-provenance `State` whose `Compactor` extracts claims from
   agent notes, with the survivor-rate and Chao1 instrumentation of §3.6. This is the
   H8 experiment.
4. Parquet export of the matrix via `arrow`/`parquet`, so the analysis off-ramp is a
   file rather than a CSV.
5. A second storage backend, deliberately dissimilar (an append-only file with an
   index, or a remote KV), to pressure-test the `Journal` API before anything about
   it is called stable.
6. The LLM-compactor run of H5, against any OpenAI-compatible endpoint; the auditor
   already supports it.

## References

Almeida, P. S., Shoker, A., Baquero, C. *Delta State Replicated Data Types.* Journal
of Parallel and Distributed Computing, 2018.

Angiuli, C., Morehouse, E., Licata, D. R., Harper, R. *Homotopical Patch Theory.*
ICFP 2014.

Bykov, S., Geller, A., Kliot, G., Larus, J., Pandya, R., Thelin, J. *Orleans: Cloud
Computing for Everyone.* SoCC 2011.

Chao, A. *Nonparametric estimation of the number of classes in a population.*
Scandinavian Journal of Statistics, 1984.

Cloudflare. *Workers Durable Objects.* 2020.

Gjengset, J., et al. *Noria: dynamic, partially-stateful data-flow for
high-performance web applications.* OSDI 2018. And the `left-right` crate.

Good, I. J. *The population frequencies of species and the estimation of population
parameters.* Biometrika, 1953.

Hellerstein, J. M., Alvaro, P. *Keeping CALM: When Distributed Consistency Is Easy.*
Communications of the ACM, 2020.

Hellerstein, J. M., et al. *New Directions in Cloud Programming.* CIDR 2021. (Hydro.)

Hickey, R. *The Database as a Value.* Strange Loop, 2012. (Datomic.)

Kleene, S. C. *Introduction to Metamathematics.* 1952. Tarski, A. *A lattice-theoretical
fixpoint theorem and its applications.* Pacific J. Math., 1955.

LangGraph documentation, *Persistence* and *Functional API*. 2025–2026.

Mimram, S., Di Giusto, C. *A Categorical Theory of Patches.* ENTCS, 2013.

Packer, C., et al. *MemGPT: Towards LLMs as Operating Systems.* 2023.

Restate documentation, *Durable Execution.* 2024–2026.

Shannon, C. E. *Coding Theorems for a Discrete Source with a Fidelity Criterion.* IRE
National Convention Record, 1959.

Shapiro, M., Preguiça, N., Baquero, C., Zawirski, M. *Conflict-free Replicated Data
Types.* SSS 2011.

Berner, C. `redb`: an embedded key-value database in pure Rust.
