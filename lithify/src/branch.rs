use std::future::Future;
use std::sync::{Arc, Mutex};

use serde::{de::DeserializeOwned, Serialize};

use crate::event::{BranchMeta, EventKind, Record};
use crate::{BranchId, Compactor, Error, Hash, Journal, Result, Seq, State};

/// A handle to one branch of a journal.
///
/// Cheap to clone. Two handles to the same branch share the journal but not the
/// in-memory state cache; the cache is validated against the journal head on every
/// read, so stale handles are safe, just slower.
pub struct Branch<S: State> {
    journal: Journal<S>,
    id: BranchId,
    cache: Arc<Mutex<Option<(Seq, S)>>>,
}

impl<S: State> Clone for Branch<S> {
    fn clone(&self) -> Self {
        Branch {
            journal: self.journal.clone(),
            id: self.id,
            cache: self.cache.clone(),
        }
    }
}

impl<S: State> Branch<S> {
    pub(crate) fn new(journal: Journal<S>, id: BranchId) -> Self {
        Branch {
            journal,
            id,
            cache: Arc::new(Mutex::new(None)),
        }
    }

    pub fn id(&self) -> BranchId {
        self.id
    }

    pub fn journal(&self) -> &Journal<S> {
        &self.journal
    }

    pub fn meta(&self) -> Result<BranchMeta> {
        self.journal.meta(self.id)
    }

    /// Sequence number of the latest event visible on this branch.
    pub fn head(&self) -> Result<Seq> {
        Ok(self.meta()?.head)
    }

    /// Hash of the event at the head: a content address for "everything that has
    /// happened on this branch".
    pub fn head_hash(&self) -> Result<Hash> {
        Ok(self.meta()?.head_hash)
    }

    // ---- writing ----------------------------------------------------------------

    /// Append a delta. Returns its seq.
    pub fn apply(&self, delta: S::Delta) -> Result<Seq> {
        let value = serde_json::to_value(&delta)?;
        let rec = self
            .journal
            .append(self.id, EventKind::Delta { delta: value }, false)?;
        // Keep the cache warm if it was exactly one step behind.
        let mut c = self.cache.lock().unwrap();
        match c.as_mut() {
            Some((seq, st)) if *seq + 1 == rec.seq => {
                st.apply(&delta);
                *seq = rec.seq;
            }
            _ => *c = None,
        }
        Ok(rec.seq)
    }

    /// Run a side effect exactly once per key, per history.
    ///
    /// If an effect with this key is already journaled anywhere in this branch's
    /// resolved history, its stored result is returned and `f` is **not** called.
    /// Otherwise `f` runs, its result is committed with immediate durability, and
    /// then returned. A fork therefore inherits every effect of its prefix.
    ///
    /// The key is an idempotency key: include whatever makes two calls distinct
    /// (a step counter, the request hash). Two calls with the same key on one history
    /// see one execution; that is the feature.
    pub fn effect<T, E, F>(&self, key: impl AsRef<[u8]>, f: F) -> Result<T>
    where
        T: Serialize + DeserializeOwned,
        E: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
        F: FnOnce() -> std::result::Result<T, E>,
    {
        let key = key.as_ref();
        let key_hash: Hash = *blake3::hash(key).as_bytes();
        if let Some(v) = self.journal.find_effect(self.id, &key_hash)? {
            return Ok(serde_json::from_value(v)?);
        }
        let out = f().map_err(|e| Error::Effect(e.into()))?;
        self.record_effect(key, key_hash, &out)?;
        Ok(out)
    }

    /// Async twin of [`Branch::effect`].
    pub async fn effect_async<T, E, F, Fut>(&self, key: impl AsRef<[u8]>, f: F) -> Result<T>
    where
        T: Serialize + DeserializeOwned,
        E: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
        F: FnOnce() -> Fut,
        Fut: Future<Output = std::result::Result<T, E>>,
    {
        let key = key.as_ref();
        let key_hash: Hash = *blake3::hash(key).as_bytes();
        if let Some(v) = self.journal.find_effect(self.id, &key_hash)? {
            return Ok(serde_json::from_value(v)?);
        }
        let out = f().await.map_err(|e| Error::Effect(e.into()))?;
        self.record_effect(key, key_hash, &out)?;
        Ok(out)
    }

    fn record_effect<T: Serialize>(&self, key: &[u8], key_hash: Hash, out: &T) -> Result<()> {
        let key_hint = std::str::from_utf8(key)
            .ok()
            .map(|s| s.chars().take(120).collect::<String>());
        let rec = self.journal.append(
            self.id,
            EventKind::Effect {
                key_hash,
                key_hint,
                result: serde_json::to_value(out)?,
            },
            true,
        )?;
        self.bump_cache_seq(rec.seq);
        Ok(())
    }

    /// Record a measurement for the results matrix.
    pub fn observe(&self, name: impl Into<String>, value: f64) -> Result<Seq> {
        let rec = self.journal.append(
            self.id,
            EventKind::Observe {
                name: name.into(),
                value,
            },
            false,
        )?;
        self.bump_cache_seq(rec.seq);
        Ok(rec.seq)
    }

    /// Replace the state with the compactor's output, journaled as a `Compact` event
    /// with a snapshot at the same seq.
    pub fn compact<C: Compactor<S>>(&self, compactor: &C) -> Result<Seq> {
        let before = self.state()?;
        let after = compactor
            .compact(self, &before)
            .map_err(|e| Error::Compactor(Box::new(e)))?;
        let rec = self.journal.append(
            self.id,
            EventKind::Compact {
                state: serde_json::to_value(&after)?,
            },
            true,
        )?;
        *self.cache.lock().unwrap() = Some((rec.seq, after));
        Ok(rec.seq)
    }

    /// Write a snapshot at the current head without changing anything. Makes later
    /// `at(head)` and `state()` calls O(1) instead of O(replay).
    pub fn snapshot(&self) -> Result<Seq> {
        let head = self.head()?;
        let st = self.state()?;
        self.journal.write_snapshot(self.id, head, &st)?;
        Ok(head)
    }

    /// `state().size() > budget`.
    pub fn over_budget(&self, budget: usize) -> Result<bool> {
        Ok(self.state()?.size() > budget)
    }

    // ---- reading ----------------------------------------------------------------

    /// The state at the head. Cached; the cache is validated against the head.
    pub fn state(&self) -> Result<S> {
        let head = self.head()?;
        {
            let c = self.cache.lock().unwrap();
            if let Some((seq, st)) = c.as_ref() {
                if *seq == head {
                    return Ok(st.clone());
                }
            }
        }
        let st = self.journal.materialize(self.id, head)?;
        *self.cache.lock().unwrap() = Some((head, st.clone()));
        Ok(st)
    }

    /// Borrow the head state without cloning it out.
    pub fn with_state<R>(&self, f: impl FnOnce(&S) -> R) -> Result<R> {
        let head = self.head()?;
        let mut c = self.cache.lock().unwrap();
        let fresh = !matches!(c.as_ref(), Some((seq, _)) if *seq == head);
        if fresh {
            *c = Some((head, self.journal.materialize(self.id, head)?));
        }
        Ok(f(&c.as_ref().unwrap().1))
    }

    /// The state as of `seq`: time travel. `at(0)` is `S::default()`.
    pub fn at(&self, seq: Seq) -> Result<S> {
        self.journal.materialize(self.id, seq)
    }

    /// Resolved records in `lo..=hi` (clamped to the head), including inherited ones.
    pub fn history(&self, lo: Seq, hi: Seq) -> Result<Vec<Record>> {
        self.journal.records(self.id, lo, hi)
    }

    /// Every record from the beginning of time to the head.
    pub fn full_history(&self) -> Result<Vec<Record>> {
        self.journal.records(self.id, 1, Seq::MAX)
    }

    // ---- forking ----------------------------------------------------------------

    /// Fork at the head.
    pub fn fork(&self, name: Option<String>, params: serde_json::Value) -> Result<Branch<S>> {
        let head = self.head()?;
        self.fork_at(head, name, params)
    }

    /// Fork at `seq`. The new branch shares every event and every effect up to and
    /// including `seq`, and nothing after. O(1): it is one row in the branch table.
    pub fn fork_at(
        &self,
        seq: Seq,
        name: Option<String>,
        params: serde_json::Value,
    ) -> Result<Branch<S>> {
        let meta = self.journal.create_fork(self.id, seq, name, params)?;
        Ok(Branch::new(self.journal.clone(), meta.id))
    }

    fn bump_cache_seq(&self, seq: Seq) {
        // Effects and observations don't change the state, so a cache that was
        // exactly one behind is still valid; just advance its seq.
        let mut c = self.cache.lock().unwrap();
        match c.as_mut() {
            Some((s, _)) if *s + 1 == seq => *s = seq,
            _ => *c = None,
        }
    }
}

impl<S: State> std::fmt::Debug for Branch<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Branch").field("id", &self.id).finish()
    }
}
