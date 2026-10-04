use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use redb::{
    backends::InMemoryBackend, Database, Durability, MultimapTableDefinition, ReadableDatabase,
    ReadableTable, ReadableTableMetadata, TableDefinition,
};
use serde::{Deserialize, Serialize};

use crate::event::{now_ms, BranchMeta, EventKind, Record};
use crate::{Branch, BranchId, Error, Hash, Result, Seq, State, ZERO_HASH};

// ---- tables ----------------------------------------------------------------------
//
// Every access pattern lithify needs is a point lookup or a prefix range over one of
// these. There are deliberately no secondary indexes beyond BY_HASH and EFFECTS: if
// you find yourself wanting one, that is the library telling you something about the
// shape of your query.

/// branch id -> BranchMeta (json)
const BRANCHES: TableDefinition<u64, &[u8]> = TableDefinition::new("branches");
/// parent id -> child id*
const CHILDREN: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("children");
/// (branch, seq) -> Record (json)
const EVENTS: TableDefinition<(u64, u64), &[u8]> = TableDefinition::new("events");
/// (branch, blake3(key)) -> seq of the Effect event holding the result
const EFFECTS: TableDefinition<(u64, &[u8; 32]), u64> = TableDefinition::new("effects");
/// (branch, seq) -> State (json)
const SNAPSHOTS: TableDefinition<(u64, u64), &[u8]> = TableDefinition::new("snapshots");
/// event hash -> (branch, seq)
const BY_HASH: TableDefinition<&[u8; 32], (u64, u64)> = TableDefinition::new("by_hash");

pub(crate) const ROOT: BranchId = 0;

/// Tuning knobs. The defaults are the safe ones.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// Commit deltas and observations without fsync. Effects, compactions, snapshots
    /// and forks are always committed with `Durability::Immediate` regardless, because
    /// the guarantee "a result observed is a result journaled" is the whole point.
    ///
    /// With this on, a power cut (not a `kill -9`) can lose trailing deltas; the
    /// journal stays consistent, it just ends earlier than you remember.
    pub lazy_deltas: bool,
}

/// A contiguous run of one branch's own events, `lo..=hi`, as part of a resolved
/// history. A branch's full history is a list of these from the root down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub branch: BranchId,
    pub lo: Seq,
    pub hi: Seq,
}

pub(crate) struct Inner {
    pub(crate) db: Database,
    pub(crate) opts: Options,
}

/// A journal: one redb database holding any number of branches of one `State` type.
///
/// Cheap to clone; all clones share the database.
pub struct Journal<S: State> {
    pub(crate) inner: Arc<Inner>,
    _s: PhantomData<fn() -> S>,
}

impl<S: State> Clone for Journal<S> {
    fn clone(&self) -> Self {
        Journal {
            inner: self.inner.clone(),
            _s: PhantomData,
        }
    }
}

/// One row of the results matrix: an observation with its branch's merged params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationRow {
    pub branch: BranchId,
    pub branch_name: Option<String>,
    pub seq: Seq,
    pub metric: String,
    pub value: f64,
    pub params: serde_json::Value,
}

impl<S: State> Journal<S> {
    /// Open (or create) a journal file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, Options::default())
    }

    pub fn open_with(path: impl AsRef<Path>, opts: Options) -> Result<Self> {
        let db = Database::create(path)?;
        Self::from_db(db, opts)
    }

    /// A journal that lives only in memory. Same code path as the on-disk one; this is
    /// what tests use.
    pub fn in_memory() -> Result<Self> {
        let db = Database::builder().create_with_backend(InMemoryBackend::new())?;
        Self::from_db(db, Options::default())
    }

    fn from_db(db: Database, opts: Options) -> Result<Self> {
        // Make sure every table exists so reads never fail on an empty journal.
        let txn = db.begin_write()?;
        {
            txn.open_table(BRANCHES)?;
            txn.open_multimap_table(CHILDREN)?;
            txn.open_table(EVENTS)?;
            txn.open_table(EFFECTS)?;
            txn.open_table(SNAPSHOTS)?;
            txn.open_table(BY_HASH)?;
        }
        txn.commit()?;
        let j = Journal {
            inner: Arc::new(Inner { db, opts }),
            _s: PhantomData,
        };
        j.ensure_root()?;
        Ok(j)
    }

    fn ensure_root(&self) -> Result<()> {
        let txn = self.inner.db.begin_write()?;
        {
            let mut t = txn.open_table(BRANCHES)?;
            if t.get(ROOT)?.is_none() {
                let meta = BranchMeta {
                    id: ROOT,
                    name: Some("root".into()),
                    parent: None,
                    fork_seq: 0,
                    head: 0,
                    head_hash: ZERO_HASH,
                    params: serde_json::Value::Object(Default::default()),
                    created_ms: now_ms(),
                };
                t.insert(ROOT, serde_json::to_vec(&meta)?.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// The root branch (id 0). Always exists.
    pub fn root(&self) -> Branch<S> {
        Branch::new(self.clone(), ROOT)
    }

    /// A handle to an existing branch.
    pub fn branch(&self, id: BranchId) -> Result<Branch<S>> {
        self.meta(id)?;
        Ok(Branch::new(self.clone(), id))
    }

    pub fn meta(&self, id: BranchId) -> Result<BranchMeta> {
        let txn = self.inner.db.begin_read()?;
        let t = txn.open_table(BRANCHES)?;
        read_meta(&t, id)
    }

    /// All branches, in id order.
    pub fn branches(&self) -> Result<Vec<BranchMeta>> {
        let txn = self.inner.db.begin_read()?;
        let t = txn.open_table(BRANCHES)?;
        let mut out = Vec::with_capacity(t.len()? as usize);
        for kv in t.iter()? {
            let (_, v) = kv?;
            out.push(serde_json::from_slice(v.value())?);
        }
        Ok(out)
    }

    /// Direct children of a branch.
    pub fn children(&self, id: BranchId) -> Result<Vec<BranchId>> {
        let txn = self.inner.db.begin_read()?;
        let t = txn.open_multimap_table(CHILDREN)?;
        let mut out = Vec::new();
        for v in t.get(id)? {
            out.push(v?.value());
        }
        Ok(out)
    }

    /// Where an event with this hash lives, if anywhere.
    pub fn lookup_hash(&self, hash: &Hash) -> Result<Option<(BranchId, Seq)>> {
        let txn = self.inner.db.begin_read()?;
        let t = txn.open_table(BY_HASH)?;
        Ok(t.get(hash)?.map(|g| g.value()))
    }

    /// The resolved history of `branch` up to and including `upto`, as segments from
    /// the root down. This is the walk that a SQL `WITH RECURSIVE` would hide.
    pub fn chain(&self, branch: BranchId, upto: Seq) -> Result<Vec<Segment>> {
        let txn = self.inner.db.begin_read()?;
        let t = txn.open_table(BRANCHES)?;
        chain_in(&t, branch, upto)
    }

    /// Fork `base` at `at` once per parameter set. Returns the new branches in order.
    pub fn sweep<I>(&self, base: BranchId, at: Seq, params: I) -> Result<Vec<Branch<S>>>
    where
        I: IntoIterator<Item = serde_json::Value>,
    {
        let base = self.branch(base)?;
        params
            .into_iter()
            .enumerate()
            .map(|(i, p)| base.fork_at(at, Some(format!("sweep-{i}")), p))
            .collect()
    }

    /// Every observation in the journal, joined (map-side, by hand) to its branch's
    /// merged params. A full scan, on purpose: this is the analysis off-ramp, not a
    /// hot path. Export it and query it elsewhere.
    pub fn matrix(&self) -> Result<Vec<ObservationRow>> {
        let txn = self.inner.db.begin_read()?;
        let branches = txn.open_table(BRANCHES)?;
        let events = txn.open_table(EVENTS)?;

        let mut metas: std::collections::BTreeMap<BranchId, BranchMeta> = Default::default();
        for kv in branches.iter()? {
            let (_, v) = kv?;
            let m: BranchMeta = serde_json::from_slice(v.value())?;
            metas.insert(m.id, m);
        }
        let merged = |id: BranchId| -> serde_json::Value {
            // ancestors first, so children override
            let mut lineage = Vec::new();
            let mut cur = Some(id);
            while let Some(c) = cur {
                let m = &metas[&c];
                lineage.push(m);
                cur = m.parent;
            }
            let mut out = serde_json::Map::new();
            for m in lineage.iter().rev() {
                if let serde_json::Value::Object(o) = &m.params {
                    for (k, v) in o {
                        out.insert(k.clone(), v.clone());
                    }
                }
            }
            serde_json::Value::Object(out)
        };

        let mut rows = Vec::new();
        for kv in events.iter()? {
            let (_, v) = kv?;
            let rec: Record = serde_json::from_slice(v.value())?;
            if let EventKind::Observe { name, value } = rec.event {
                rows.push(ObservationRow {
                    branch: rec.branch,
                    branch_name: metas.get(&rec.branch).and_then(|m| m.name.clone()),
                    seq: rec.seq,
                    metric: name,
                    value,
                    params: merged(rec.branch),
                });
            }
        }
        Ok(rows)
    }

    /// Render matrix rows as CSV with one column per distinct param key.
    pub fn matrix_csv(rows: &[ObservationRow]) -> String {
        let mut keys: Vec<String> = Vec::new();
        for r in rows {
            if let serde_json::Value::Object(o) = &r.params {
                for k in o.keys() {
                    if !keys.contains(k) {
                        keys.push(k.clone());
                    }
                }
            }
        }
        keys.sort();
        let mut s = String::from("branch,branch_name,seq,metric,value");
        for k in &keys {
            s.push(',');
            s.push_str(k);
        }
        s.push('\n');
        for r in rows {
            s.push_str(&format!(
                "{},{},{},{},{}",
                r.branch,
                r.branch_name.clone().unwrap_or_default(),
                r.seq,
                r.metric,
                r.value
            ));
            for k in &keys {
                s.push(',');
                if let Some(v) = r.params.get(k) {
                    match v {
                        serde_json::Value::String(x) => s.push_str(x),
                        other => s.push_str(&other.to_string()),
                    }
                }
            }
            s.push('\n');
        }
        s
    }

    // ---- internals used by Branch ------------------------------------------------

    /// Append one event to `branch` at `head + 1`. Returns the record.
    pub(crate) fn append(
        &self,
        branch: BranchId,
        event: EventKind,
        immediate: bool,
    ) -> Result<Record> {
        let mut txn = self.inner.db.begin_write()?;
        if !immediate && self.inner.opts.lazy_deltas {
            txn.set_durability(Durability::None)?;
        }
        let rec = {
            let mut branches = txn.open_table(BRANCHES)?;
            let mut meta = read_meta(&branches, branch)?;
            let seq = meta.head + 1;
            let parent = meta.head_hash;
            let hash = Record::compute_hash(&parent, &event);
            let rec = Record {
                branch,
                seq,
                hash,
                parent,
                ts_ms: now_ms(),
                event,
            };
            {
                let mut events = txn.open_table(EVENTS)?;
                events.insert((branch, seq), serde_json::to_vec(&rec)?.as_slice())?;
            }
            {
                let mut by_hash = txn.open_table(BY_HASH)?;
                by_hash.insert(&hash, (branch, seq))?;
            }
            match &rec.event {
                EventKind::Effect { key_hash, .. } => {
                    let mut effects = txn.open_table(EFFECTS)?;
                    effects.insert((branch, key_hash), seq)?;
                }
                EventKind::Compact { state } => {
                    let mut snaps = txn.open_table(SNAPSHOTS)?;
                    snaps.insert((branch, seq), serde_json::to_vec(state)?.as_slice())?;
                }
                _ => {}
            }
            meta.head = seq;
            meta.head_hash = hash;
            branches.insert(branch, serde_json::to_vec(&meta)?.as_slice())?;
            rec
        };
        txn.commit()?;
        Ok(rec)
    }

    /// Write a snapshot of `state` at `(branch, seq)`.
    pub(crate) fn write_snapshot(&self, branch: BranchId, seq: Seq, state: &S) -> Result<()> {
        let txn = self.inner.db.begin_write()?;
        {
            let mut snaps = txn.open_table(SNAPSHOTS)?;
            snaps.insert((branch, seq), serde_json::to_vec(state)?.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Create a fork of `parent` at `at`.
    pub(crate) fn create_fork(
        &self,
        parent: BranchId,
        at: Seq,
        name: Option<String>,
        params: serde_json::Value,
    ) -> Result<BranchMeta> {
        let txn = self.inner.db.begin_write()?;
        let meta = {
            let mut branches = txn.open_table(BRANCHES)?;
            let pmeta = read_meta(&branches, parent)?;
            if at > pmeta.head {
                return Err(Error::SeqOutOfRange {
                    branch: parent,
                    seq: at,
                    head: pmeta.head,
                });
            }
            let head_hash = if at == 0 {
                ZERO_HASH
            } else {
                let events = txn.open_table(EVENTS)?;
                let segs = chain_in(&branches, parent, at)?;
                let (b, s) = resolve_in_segments(&segs, at).ok_or_else(|| {
                    Error::Corrupt(format!("seq {at} not covered by chain of branch {parent}"))
                })?;
                let rec: Record = serde_json::from_slice(
                    events
                        .get((b, s))?
                        .ok_or_else(|| Error::Corrupt(format!("missing event ({b},{s})")))?
                        .value(),
                )?;
                rec.hash
            };
            let id = branches
                .last()?
                .map(|(k, _)| k.value() + 1)
                .unwrap_or(ROOT + 1);
            let meta = BranchMeta {
                id,
                name,
                parent: Some(parent),
                fork_seq: at,
                head: at,
                head_hash,
                params,
                created_ms: now_ms(),
            };
            branches.insert(id, serde_json::to_vec(&meta)?.as_slice())?;
            let mut children = txn.open_multimap_table(CHILDREN)?;
            children.insert(parent, id)?;
            meta
        };
        txn.commit()?;
        Ok(meta)
    }

    /// Materialise the state of `branch` as of `seq`: nearest snapshot at or before,
    /// then replay.
    pub(crate) fn materialize(&self, branch: BranchId, seq: Seq) -> Result<S> {
        let txn = self.inner.db.begin_read()?;
        let branches = txn.open_table(BRANCHES)?;
        let meta = read_meta(&branches, branch)?;
        if seq > meta.head {
            return Err(Error::SeqOutOfRange {
                branch,
                seq,
                head: meta.head,
            });
        }
        if seq == 0 {
            return Ok(S::default());
        }
        let segs = chain_in(&branches, branch, seq)?;
        let events = txn.open_table(EVENTS)?;
        let snaps = txn.open_table(SNAPSHOTS)?;

        // Nearest snapshot at or before `seq`, searching newest segment first.
        let mut state = S::default();
        let mut from: Seq = 1;
        'outer: for seg in segs.iter().rev() {
            let mut r = snaps.range((seg.branch, seg.lo)..=(seg.branch, seg.hi))?;
            if let Some(kv) = r.next_back() {
                let (k, v) = kv?;
                state = serde_json::from_slice(v.value())?;
                from = k.value().1 + 1;
                break 'outer;
            }
        }

        for seg in &segs {
            if seg.hi < from {
                continue;
            }
            let lo = seg.lo.max(from);
            for kv in events.range((seg.branch, lo)..=(seg.branch, seg.hi))? {
                let (_, v) = kv?;
                let rec: Record = serde_json::from_slice(v.value())?;
                apply_record::<S>(&mut state, &rec)?;
            }
        }
        Ok(state)
    }

    /// Find a journaled effect result for `key_hash` visible from `(branch, head)`.
    pub(crate) fn find_effect(
        &self,
        branch: BranchId,
        key_hash: &Hash,
    ) -> Result<Option<serde_json::Value>> {
        let txn = self.inner.db.begin_read()?;
        let branches = txn.open_table(BRANCHES)?;
        let meta = read_meta(&branches, branch)?;
        let segs = chain_in(&branches, branch, meta.head)?;
        let effects = txn.open_table(EFFECTS)?;
        let events = txn.open_table(EVENTS)?;
        for seg in segs.iter().rev() {
            if let Some(g) = effects.get((seg.branch, key_hash))? {
                let seq = g.value();
                if seq >= seg.lo && seq <= seg.hi {
                    let rec: Record = serde_json::from_slice(
                        events
                            .get((seg.branch, seq))?
                            .ok_or_else(|| {
                                Error::Corrupt(format!(
                                    "effect index points at missing event ({},{seq})",
                                    seg.branch
                                ))
                            })?
                            .value(),
                    )?;
                    if let EventKind::Effect { result, .. } = rec.event {
                        return Ok(Some(result));
                    }
                    return Err(Error::Corrupt(format!(
                        "effect index points at non-effect event ({},{seq})",
                        seg.branch
                    )));
                }
            }
        }
        Ok(None)
    }

    /// Resolved records of `branch` in `lo..=hi`.
    pub(crate) fn records(&self, branch: BranchId, lo: Seq, hi: Seq) -> Result<Vec<Record>> {
        let txn = self.inner.db.begin_read()?;
        let branches = txn.open_table(BRANCHES)?;
        let meta = read_meta(&branches, branch)?;
        let hi = hi.min(meta.head);
        let lo = lo.max(1);
        if lo > hi {
            return Ok(Vec::new());
        }
        let segs = chain_in(&branches, branch, hi)?;
        let events = txn.open_table(EVENTS)?;
        let mut out = Vec::new();
        for seg in &segs {
            if seg.hi < lo {
                continue;
            }
            let l = seg.lo.max(lo);
            for kv in events.range((seg.branch, l)..=(seg.branch, seg.hi))? {
                let (_, v) = kv?;
                out.push(serde_json::from_slice(v.value())?);
            }
        }
        Ok(out)
    }
}

// ---- free helpers ------------------------------------------------------------------

fn read_meta(t: &impl ReadableTable<u64, &'static [u8]>, id: BranchId) -> Result<BranchMeta> {
    let g = t.get(id)?.ok_or(Error::NoSuchBranch(id))?;
    Ok(serde_json::from_slice(g.value())?)
}

fn chain_in(
    t: &impl ReadableTable<u64, &'static [u8]>,
    branch: BranchId,
    upto: Seq,
) -> Result<Vec<Segment>> {
    let mut segs = Vec::new();
    let mut cur = read_meta(t, branch)?;
    let mut hi = upto.min(cur.head);
    loop {
        let lo = cur.fork_seq + 1;
        if hi >= lo {
            segs.push(Segment {
                branch: cur.id,
                lo,
                hi,
            });
        }
        match cur.parent {
            Some(p) => {
                hi = cur.fork_seq.min(hi);
                cur = read_meta(t, p)?;
            }
            None => break,
        }
    }
    segs.reverse();
    Ok(segs)
}

fn resolve_in_segments(segs: &[Segment], seq: Seq) -> Option<(BranchId, Seq)> {
    segs.iter()
        .find(|s| seq >= s.lo && seq <= s.hi)
        .map(|s| (s.branch, seq))
}

pub(crate) fn apply_record<S: State>(state: &mut S, rec: &Record) -> Result<()> {
    match &rec.event {
        EventKind::Delta { delta } => {
            let d: S::Delta = serde_json::from_value(delta.clone())?;
            state.apply(&d);
        }
        EventKind::Compact { state: s } => {
            *state = serde_json::from_value(s.clone())?;
        }
        EventKind::Effect { .. } | EventKind::Observe { .. } => {}
    }
    Ok(())
}
