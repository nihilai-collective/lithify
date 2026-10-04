#!/usr/bin/env bash
# Run the lithify agent and the LangGraph baseline through the same three experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${1:-$(ls -d ~/.cargo/registry/src/*/redb-* 2>/dev/null | head -1)}"
[ -n "$REPO" ] || { echo "usage: scripts/compare.sh <repo-to-audit>"; exit 1; }
cargo build --release -q
A=./target/release/auditor
PY="python3 compare/langgraph/auditor.py"
T=/tmp/lithify-compare; rm -rf "$T"; mkdir -p "$T"

echo "== plain run (budget 2000)"
$A  --db $T/a.redb   run --repo "$REPO" --budget 2000 --max-files 200 2>/dev/null | tail -2
$PY --db $T/a.sqlite run --repo "$REPO" --budget 2000 --max-files 200 2>/dev/null | tail -2

echo; echo "== sweep from 19 files in"
$A  --db $T/a.redb   sweep --at 191   --budgets 300,600,1200,2400,4800 2>/dev/null | grep recall
$PY --db $T/a.sqlite sweep --index 19 --budgets 300,600,1200,2400,4800 2>/dev/null | grep recall

echo; echo "== crash: 8 SIGKILLs each, 20 ms injected model latency"
for t in 1 2 3; do
  echo "-- lithify trial $t";   $A  --db $T/c$t.redb   crash-demo --repo "$REPO" --kills 8 --budget 1500 --max-files 200 --interval-ms 250 | grep -E "trace|re-executed"
done
for t in 1 2 3; do
  echo "-- langgraph trial $t"; $PY --db $T/c$t.sqlite crash-demo --repo "$REPO" --kills 8 --budget 1500 --max-files 200 --interval-ms 450 | grep -E "trace|re-executed"
done
