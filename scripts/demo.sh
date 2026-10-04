#!/usr/bin/env bash
# The whole pitch in one minute: durability, time travel, forks, a sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${1:-$(ls -d ~/.cargo/registry/src/*/redb-* 2>/dev/null | head -1)}"
[ -n "$REPO" ] || { echo "usage: scripts/demo.sh <repo-to-audit>"; exit 1; }
cargo build --release -q
A=./target/release/auditor
DB=/tmp/lithify-demo.redb
rm -f "$DB"

echo "== 1. crash-safety: SIGKILL the agent four times, then finish"
$A --db "$DB" crash-demo --repo "$REPO" --kills 4 --budget 1500 --max-files 200 --interval-ms 250

echo; echo "== 2. time travel: what did the agent know at event 40?"
$A --db "$DB" at --seq 40

echo; echo "== 3. history: the first dozen events, with their content hashes"
$A --db "$DB" history --from 1 --to 12

echo; echo "== 4. sweep: fork at event 191 (19 files in), one branch per budget"
$A --db "$DB" sweep --at 191 --budgets 300,600,1200,2400,4800 2>/dev/null | grep -E "metric|recall"

echo; echo "== 5. branches"
$A --db "$DB" status | grep '^branch'
