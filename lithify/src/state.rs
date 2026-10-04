use serde::{de::DeserializeOwned, Serialize};

use crate::{Branch, Result};

/// The consumer's context type.
///
/// A `State` is a plain value that knows how to apply its own deltas. lithify never
/// mutates it except through [`State::apply`] and never stores it except as JSON, so
/// the consumer owns the whole shape of the thing. `Default` is the empty context at
/// sequence 0.
///
/// The one design constraint worth internalising: **deltas are the unit of truth, the
/// state is a cache**. Anything you want to be able to see later, fork from, or
/// replay must arrive as a delta.
pub trait State: Serialize + DeserializeOwned + Default + Clone + Send + Sync + 'static {
    /// The event type. Usually a hand-written `enum`.
    type Delta: Serialize + DeserializeOwned + Send + Sync + 'static;

    /// Apply one delta. Must be deterministic: the same `(state, delta)` must always
    /// produce the same new state, or replay will diverge from the live run.
    fn apply(&mut self, delta: &Self::Delta);

    /// Approximate size of the context, in whatever unit the consumer budgets in
    /// (tokens, bytes, messages). Used by [`Branch::over_budget`]. Defaults to 0,
    /// meaning "never over budget".
    fn size(&self) -> usize {
        0
    }
}

/// Produces a smaller state from a larger one.
///
/// The compactor is handed the branch so that anything expensive or
/// non-deterministic it does (an LLM summarisation call, say) can be journaled with
/// [`Branch::effect`], which makes compaction itself replayable and crash-safe.
pub trait Compactor<S: State> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn compact(&self, branch: &Branch<S>, state: &S) -> Result<S, Self::Error>;
}

/// A compactor from a closure.
pub struct FnCompactor<F>(pub F);

impl<S, F> Compactor<S> for FnCompactor<F>
where
    S: State,
    F: Fn(&Branch<S>, &S) -> Result<S, crate::Error>,
{
    type Error = crate::Error;

    fn compact(&self, branch: &Branch<S>, state: &S) -> Result<S, Self::Error> {
        (self.0)(branch, state)
    }
}
