use crate::{BranchId, Seq};

/// Everything that can go wrong inside lithify.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("database: {0}")]
    Database(#[from] redb::DatabaseError),
    #[error("transaction: {0}")]
    Transaction(#[from] redb::TransactionError),
    #[error("table: {0}")]
    Table(#[from] redb::TableError),
    #[error("storage: {0}")]
    Storage(#[from] redb::StorageError),
    #[error("commit: {0}")]
    Commit(#[from] redb::CommitError),
    #[error("durability: {0}")]
    Durability(#[from] redb::SetDurabilityError),
    #[error("serialization: {0}")]
    Json(#[from] serde_json::Error),
    #[error("no such branch: {0}")]
    NoSuchBranch(BranchId),
    #[error("branch {branch} has no event at seq {seq} (head is {head})")]
    SeqOutOfRange {
        branch: BranchId,
        seq: Seq,
        head: Seq,
    },
    #[error("corrupt journal: {0}")]
    Corrupt(String),
    #[error("effect failed: {0}")]
    Effect(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("compactor failed: {0}")]
    Compactor(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
