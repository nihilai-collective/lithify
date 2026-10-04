use std::sync::atomic::{AtomicUsize, Ordering};

use lithify::{Compactor, EventKind, Journal, State};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct Counter {
    total: i64,
    log: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Delta {
    Add(i64),
    Clear,
}

impl State for Counter {
    type Delta = Delta;
    fn apply(&mut self, d: &Delta) {
        match d {
            Delta::Add(n) => {
                self.total += n;
                self.log.push(*n);
            }
            Delta::Clear => self.log.clear(),
        }
    }
    fn size(&self) -> usize {
        self.log.len()
    }
}

/// Keeps the total, forgets the log.
struct Squash;
impl Compactor<Counter> for Squash {
    type Error = lithify::Error;
    fn compact(&self, _b: &lithify::Branch<Counter>, s: &Counter) -> Result<Counter, Self::Error> {
        Ok(Counter {
            total: s.total,
            log: vec![],
        })
    }
}

fn io_ok<T>(t: T) -> Result<T, std::io::Error> {
    Ok(t)
}

#[test]
fn replay_matches_live() {
    let j = Journal::<Counter>::in_memory().unwrap();
    let b = j.root();
    for i in 1..=50 {
        b.apply(Delta::Add(i)).unwrap();
        if i % 7 == 0 {
            b.apply(Delta::Clear).unwrap();
        }
    }
    let live = b.state().unwrap();
    // A fresh handle has no cache and must materialise from the log.
    let fresh = j.branch(0).unwrap().state().unwrap();
    assert_eq!(live, fresh);
    assert_eq!(live.total, (1..=50).sum::<i64>());
    // Every prefix is reconstructible.
    let head = b.head().unwrap();
    let mut model = Counter::default();
    let hist = b.full_history().unwrap();
    assert_eq!(hist.len() as u64, head);
    for rec in &hist {
        if let EventKind::Delta { delta } = &rec.event {
            let d: Delta = serde_json::from_value(delta.clone()).unwrap();
            model.apply(&d);
        }
        assert_eq!(b.at(rec.seq).unwrap(), model, "mismatch at seq {}", rec.seq);
    }
}

#[test]
fn effects_execute_once_and_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("j.redb");
    let calls = AtomicUsize::new(0);
    {
        let j = Journal::<Counter>::open(&path).unwrap();
        let b = j.root();
        let a: i64 = b
            .effect("k1", || {
                calls.fetch_add(1, Ordering::SeqCst);
                io_ok(41)
            })
            .unwrap();
        let a2: i64 = b
            .effect("k1", || {
                calls.fetch_add(1, Ordering::SeqCst);
                io_ok(99)
            })
            .unwrap();
        assert_eq!((a, a2), (41, 41));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        b.apply(Delta::Add(a)).unwrap();
    }
    {
        let j = Journal::<Counter>::open(&path).unwrap();
        let b = j.root();
        let a: i64 = b
            .effect("k1", || {
                calls.fetch_add(1, Ordering::SeqCst);
                io_ok(123)
            })
            .unwrap();
        assert_eq!(a, 41);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "effect re-executed after reopen"
        );
        assert_eq!(b.state().unwrap().total, 41);
        assert_eq!(b.head().unwrap(), 2);
    }
}

#[test]
fn fork_shares_prefix_including_effects() {
    let j = Journal::<Counter>::in_memory().unwrap();
    let root = j.root();
    for i in 1..=5 {
        root.apply(Delta::Add(i)).unwrap();
    }
    let _: i64 = root.effect("shared", || io_ok(1000)).unwrap(); // seq 6
    root.apply(Delta::Add(6)).unwrap(); // seq 7
    let _: i64 = root.effect("later", || io_ok(2000)).unwrap(); // seq 8

    let fork = root.fork_at(6, Some("f".into()), json!({"x": 1})).unwrap();
    assert_eq!(fork.head().unwrap(), 6);
    assert_eq!(fork.at(6).unwrap(), root.at(6).unwrap());
    assert_eq!(fork.state().unwrap().total, 15);

    // Inherited effect: not re-run.
    let v: i64 = fork.effect("shared", || io_ok(-1)).unwrap();
    assert_eq!(v, 1000);
    assert_eq!(fork.head().unwrap(), 6, "inherited effect must not append");

    // Effect after the fork point on the parent is NOT visible.
    let v: i64 = fork.effect("later", || io_ok(-2)).unwrap();
    assert_eq!(v, -2);
    assert_eq!(fork.head().unwrap(), 7);

    // Siblings do not see each other's post-fork effects.
    let sib = root.fork_at(6, Some("g".into()), json!({"x": 2})).unwrap();
    let v: i64 = sib.effect("later", || io_ok(-3)).unwrap();
    assert_eq!(v, -3);

    // Diverge and check neither disturbs the other or the parent.
    fork.apply(Delta::Add(100)).unwrap();
    sib.apply(Delta::Add(200)).unwrap();
    assert_eq!(fork.state().unwrap().total, 115);
    assert_eq!(sib.state().unwrap().total, 215);
    assert_eq!(root.state().unwrap().total, 21);

    // Fork of a fork.
    let ff = fork.fork(None, json!({})).unwrap();
    assert_eq!(ff.state().unwrap(), fork.state().unwrap());
    ff.apply(Delta::Add(1)).unwrap();
    assert_eq!(ff.state().unwrap().total, 116);
    assert_eq!(fork.state().unwrap().total, 115);

    // History is resolved through the chain.
    let h = ff.full_history().unwrap();
    assert_eq!(h.len() as u64, ff.head().unwrap());
    assert_eq!(h[0].branch, 0);
    assert_eq!(h.last().unwrap().branch, ff.id());
    // Hash chain is continuous across fork boundaries.
    for w in h.windows(2) {
        assert_eq!(w[1].parent, w[0].hash);
    }
}

#[test]
fn compaction_is_lossy_forward_and_lossless_backward() {
    let j = Journal::<Counter>::in_memory().unwrap();
    let b = j.root();
    for i in 1..=10 {
        b.apply(Delta::Add(i)).unwrap();
    }
    assert!(b.over_budget(5).unwrap());
    let seq = b.compact(&Squash).unwrap();
    assert_eq!(seq, 11);
    let s = b.state().unwrap();
    assert_eq!(s.total, 55);
    assert!(s.log.is_empty());
    assert!(!b.over_budget(5).unwrap());
    // Time travel to before the compaction still has the detail.
    assert_eq!(b.at(10).unwrap().log.len(), 10);
    // Continue after compaction; replay from a fresh handle uses the snapshot.
    b.apply(Delta::Add(1)).unwrap();
    let fresh = j.branch(0).unwrap().state().unwrap();
    assert_eq!(fresh.total, 56);
    assert_eq!(fresh.log, vec![1]);
    // A fork at the compaction point starts from the compacted state.
    let f = b.fork_at(11, None, json!({})).unwrap();
    assert_eq!(f.state().unwrap().log.len(), 0);
    assert_eq!(f.state().unwrap().total, 55);
}

#[test]
fn sweep_and_matrix() {
    let j = Journal::<Counter>::in_memory().unwrap();
    let root = j.root();
    for i in 1..=3 {
        root.apply(Delta::Add(i)).unwrap();
    }
    let forks = j
        .sweep(
            0,
            3,
            [
                json!({"budget": 10}),
                json!({"budget": 20}),
                json!({"budget": 40}),
            ],
        )
        .unwrap();
    assert_eq!(forks.len(), 3);
    for (i, f) in forks.iter().enumerate() {
        f.apply(Delta::Add(i as i64)).unwrap();
        f.observe("total", f.state().unwrap().total as f64).unwrap();
        // nested fork inherits and overrides params
        if i == 0 {
            let g = f.fork(None, json!({"seed": 7})).unwrap();
            g.observe("total", 1.0).unwrap();
        }
    }
    let rows = j.matrix().unwrap();
    assert_eq!(rows.len(), 4);
    let r0 = rows.iter().find(|r| r.branch == forks[0].id()).unwrap();
    assert_eq!(r0.params, json!({"budget": 10}));
    assert_eq!(r0.value, 6.0);
    let nested = rows
        .iter()
        .find(|r| r.params.get("seed").is_some())
        .unwrap();
    assert_eq!(nested.params, json!({"budget": 10, "seed": 7}));
    let csv = Journal::<Counter>::matrix_csv(&rows);
    assert!(csv.starts_with("branch,branch_name,seq,metric,value,budget,seed\n"));
    assert_eq!(csv.lines().count(), 5);
}

#[test]
fn hashes_are_content_addresses() {
    let a = Journal::<Counter>::in_memory().unwrap();
    let b = Journal::<Counter>::in_memory().unwrap();
    for j in [&a, &b] {
        let r = j.root();
        r.apply(Delta::Add(1)).unwrap();
        r.apply(Delta::Add(2)).unwrap();
    }
    assert_eq!(a.root().head_hash().unwrap(), b.root().head_hash().unwrap());
    let h = a.root().head_hash().unwrap();
    assert_eq!(a.lookup_hash(&h).unwrap(), Some((0, 2)));
    b.root().apply(Delta::Add(3)).unwrap();
    assert_ne!(a.root().head_hash().unwrap(), b.root().head_hash().unwrap());
}

#[test]
fn seq_out_of_range_is_an_error() {
    let j = Journal::<Counter>::in_memory().unwrap();
    let b = j.root();
    b.apply(Delta::Add(1)).unwrap();
    assert!(matches!(b.at(2), Err(lithify::Error::SeqOutOfRange { .. })));
    assert!(matches!(
        b.fork_at(5, None, json!({})),
        Err(lithify::Error::SeqOutOfRange { .. })
    ));
    assert!(matches!(
        j.branch(42),
        Err(lithify::Error::NoSuchBranch(42))
    ));
}

// ---- property test: arbitrary op sequences against a model ---------------------------

mod prop {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug, Clone)]
    enum Op {
        Add(i64),
        Clear,
        Observe,
        Effect(u8),
        Compact,
        Snapshot,
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            6 => (-100i64..100).prop_map(Op::Add),
            1 => Just(Op::Clear),
            1 => Just(Op::Observe),
            2 => any::<u8>().prop_map(Op::Effect),
            1 => Just(Op::Compact),
            1 => Just(Op::Snapshot),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn every_prefix_replays(ops in proptest::collection::vec(op(), 1..60)) {
            let j = Journal::<Counter>::in_memory().unwrap();
            let b = j.root();
            // model: state after each seq
            let mut model = Counter::default();
            let mut states: Vec<Counter> = vec![model.clone()];
            for o in &ops {
                match o {
                    Op::Add(n) => { b.apply(Delta::Add(*n)).unwrap(); model.apply(&Delta::Add(*n)); states.push(model.clone()); }
                    Op::Clear => { b.apply(Delta::Clear).unwrap(); model.apply(&Delta::Clear); states.push(model.clone()); }
                    Op::Observe => { b.observe("m", 1.0).unwrap(); states.push(model.clone()); }
                    Op::Effect(k) => {
                        let before = b.head().unwrap();
                        let _: u8 = b.effect(format!("e{k}"), || io_ok(*k)).unwrap();
                        if b.head().unwrap() != before { states.push(model.clone()); }
                    }
                    Op::Compact => { b.compact(&Squash).unwrap(); model = Counter { total: model.total, log: vec![] }; states.push(model.clone()); }
                    Op::Snapshot => { b.snapshot().unwrap(); }
                }
                prop_assert_eq!(b.state().unwrap(), model.clone());
            }
            let head = b.head().unwrap();
            prop_assert_eq!(head as usize + 1, states.len());
            for (seq, expected) in states.iter().enumerate() {
                prop_assert_eq!(&b.at(seq as u64).unwrap(), expected);
            }
            // a fresh handle agrees
            prop_assert_eq!(j.branch(0).unwrap().state().unwrap(), model);
        }
    }
}

#[test]
fn fork_is_cheap_and_materialisation_is_bounded_by_snapshot_distance() {
    use std::time::Instant;
    let j = Journal::<Counter>::in_memory().unwrap();
    let root = j.root();
    for i in 1..=2000 {
        root.apply(Delta::Add(i)).unwrap();
        if i % 500 == 0 {
            root.snapshot().unwrap();
        }
    }
    let t = Instant::now();
    let forks: Vec<_> = (0..1000)
        .map(|i| {
            root.fork_at(1000 + (i % 1000), None, json!({"i": i}))
                .unwrap()
        })
        .collect();
    let fork_us = t.elapsed().as_micros() as f64 / 1000.0;
    // A fork is one row; it must not scale with history length.
    assert!(fork_us < 5_000.0, "fork took {fork_us} us");

    // at(1000) sits on a snapshot: O(1). at(1499) replays 499 deltas.
    let t = Instant::now();
    let _ = forks[0].at(1000).unwrap();
    let on_snap = t.elapsed();
    let t = Instant::now();
    let _ = forks[499].at(1499).unwrap();
    let off_snap = t.elapsed();
    eprintln!(
        "fork: {fork_us:.1} us each; at(snapshot) {:?}; at(snapshot+499) {:?}",
        on_snap, off_snap
    );
    assert_eq!(forks[499].at(1499).unwrap().total, (1..=1499).sum::<i64>());
}
