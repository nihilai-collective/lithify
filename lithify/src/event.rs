use serde::{Deserialize, Serialize};

use crate::{BranchId, Hash, Seq};

/// What a journal entry is. Stored as self-describing JSON so that the schema can
/// evolve and so that `redb`'s file can be inspected by hand.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    /// A consumer delta, applied to the state on replay.
    Delta { delta: serde_json::Value },
    /// A journaled side effect. Replay reads `result` instead of re-executing.
    Effect {
        /// blake3 of the caller's key bytes; the index key.
        key_hash: Hash,
        /// The key itself when it was valid UTF-8 and short, for humans.
        key_hint: Option<String>,
        result: serde_json::Value,
    },
    /// The state was replaced wholesale by a compactor. A snapshot is written at
    /// the same seq.
    Compact { state: serde_json::Value },
    /// A measurement, for the results matrix. No effect on state.
    Observe { name: String, value: f64 },
}

impl EventKind {
    pub fn name(&self) -> &'static str {
        match self {
            EventKind::Delta { .. } => "delta",
            EventKind::Effect { .. } => "effect",
            EventKind::Compact { .. } => "compact",
            EventKind::Observe { .. } => "observe",
        }
    }
}

/// One journal entry as stored: a hash chain link plus the event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub branch: BranchId,
    pub seq: Seq,
    /// blake3(parent || canonical(event)).
    pub hash: Hash,
    /// Hash of the previous event in this branch's *resolved* history
    /// (so a fork's first event points at its fork point), zero for seq 1.
    pub parent: Hash,
    /// Unix milliseconds at append.
    pub ts_ms: u64,
    pub event: EventKind,
}

impl Record {
    pub(crate) fn compute_hash(parent: &Hash, event: &EventKind) -> Hash {
        let mut h = blake3::Hasher::new();
        h.update(parent);
        // serde_json preserves field order for structs/enums, so this is canonical
        // enough for our purposes: identical events produce identical bytes.
        h.update(&serde_json::to_vec(event).expect("EventKind is always serializable"));
        *h.finalize().as_bytes()
    }
}

/// Metadata for one branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchMeta {
    pub id: BranchId,
    pub name: Option<String>,
    /// `None` for the root.
    pub parent: Option<BranchId>,
    /// The parent's seq this branch forked from. This branch's own events start at
    /// `fork_seq + 1`. Zero for the root.
    pub fork_seq: Seq,
    /// Highest seq this branch has an event at (own or inherited). Equal to
    /// `fork_seq` for a fresh fork.
    pub head: Seq,
    /// Hash of the event at `head`, for chaining.
    pub head_hash: Hash,
    /// Free-form parameters, merged over ancestors' when reported in the matrix.
    pub params: serde_json::Value,
    pub created_ms: u64,
}

pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
