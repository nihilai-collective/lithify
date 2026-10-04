//! # lithify
//!
//! *Sediment in, rock out.* A journaled, forkable, compactable context store for
//! long-running agents, on [`redb`].
//!
//! The consumer writes a plain `State` type and a `Delta` enum. lithify gives back:
//!
//! - **durability**: every delta and every side-effect result is journaled before it
//!   is used, so a process can be `kill -9`'d at any instant and resume at the exact
//!   step it died on, without re-executing anything that already happened;
//! - **time travel**: `branch.at(seq)` is the state exactly as it was;
//! - **forks**: `branch.fork_at(seq)` is O(1) and shares the prefix, effects
//!   included, so a parameter sweep from step 37 pays for steps 1..=37 once;
//! - **compaction**: a user-supplied `Compactor` replaces the state with a smaller
//!   one, journaled and snapshotted, so a context budget is enforced without losing
//!   the ability to ask what was known before compaction;
//! - **a results matrix**: `branch.observe(metric, value)` rows joined to each
//!   branch's parameters, exported as CSV for analysis elsewhere.
//!
//! The model is git's: an append-only log of content-addressed events, branch refs
//! that point at a sequence number, snapshots as trees, compaction as squash, replay
//! as checkout. Everything is a point lookup or a prefix range in redb; there are no
//! secondary indexes you did not build.
//!
//! ```no_run
//! use lithify::{Journal, State};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Default, Clone, Serialize, Deserialize)]
//! struct Ctx { messages: Vec<String> }
//!
//! #[derive(Serialize, Deserialize)]
//! enum Delta { Say(String) }
//!
//! impl State for Ctx {
//!     type Delta = Delta;
//!     fn apply(&mut self, d: &Delta) {
//!         match d { Delta::Say(s) => self.messages.push(s.clone()) }
//!     }
//!     fn size(&self) -> usize { self.messages.iter().map(|m| m.len()).sum() }
//! }
//!
//! # fn main() -> lithify::Result<()> {
//! let journal = Journal::<Ctx>::open("run.redb")?;   // resumes if it exists
//! let ctx = journal.root();
//! let reply: String = ctx.effect("llm:step-1", || Ok::<_, std::io::Error>("hello".to_string()))?;
//! ctx.apply(Delta::Say(reply))?;
//! let then = ctx.at(1)?;                               // time travel
//! let variant = ctx.fork_at(1, None, serde_json::json!({"temperature": 0.2}))?;
//! # let _ = (then, variant);
//! # Ok(()) }
//! ```

mod branch;
mod error;
mod event;
mod journal;
mod state;

pub use branch::Branch;
pub use error::{Error, Result};
pub use event::{BranchMeta, EventKind, Record};
pub use journal::{Journal, ObservationRow, Options, Segment};
pub use state::{Compactor, FnCompactor, State};

/// Branch identifier. The root is 0.
pub type BranchId = u64;
/// Sequence number within a branch's resolved history. Events start at 1.
pub type Seq = u64;
/// A blake3 digest.
pub type Hash = [u8; 32];

pub(crate) const ZERO_HASH: Hash = [0u8; 32];

/// Hex-encode a hash for display.
pub fn hex(h: &Hash) -> String {
    let mut s = String::with_capacity(64);
    for b in h {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

pub mod prelude {
    pub use crate::{Branch, BranchId, Compactor, Journal, Result, Seq, State};
}
