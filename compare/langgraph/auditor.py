#!/usr/bin/env python3
"""The same auditor agent, written the idiomatic LangGraph way with a SqliteSaver
checkpointer. Same mock model, same heuristics, same metrics, so the two can be
compared on behaviour rather than on prose.

Commands mirror the Rust binary: run, status, at, history, fork, sweep, matrix,
crash-demo.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

# ---- domain (identical to auditor/src/audit.rs) -------------------------------------

CATEGORIES = [
    ("unsafe", "unsafe "),
    ("fixme", "FIXME"),
    ("panic", "panic!("),
    ("todo", "TODO"),
    ("unwrap", ".unwrap()"),
    ("expect", ".expect("),
]
SEVERITY = {"unsafe": 6, "fixme": 5, "panic": 4, "todo": 3, "unwrap": 2, "expect": 1}
EXTS = {"rs", "py", "go", "js", "ts", "c", "h", "cpp", "java", "toml", "md", "sh"}
SKIP = {".git", "target", "node_modules", ".venv", "__pycache__"}


def approx_tokens(s: str) -> int:
    return math.ceil(len(s) / 4)


def scan(path: str, contents: str):
    return [(path, cat, contents.count(n)) for cat, n in CATEGORIES if contents.count(n) > 0]


def fline(f) -> str:
    return f"F: {f[0]} | {f[1]} | {f[2]}"


def parse_f(line: str):
    line = line.strip()
    if not line.startswith("F:"):
        return None
    parts = [p.strip() for p in line[2:].split("|")]
    if len(parts) != 3:
        return None
    try:
        return (parts[0], parts[1], int(parts[2]))
    except ValueError:
        return None


def list_files(root: Path, max_files: int):
    out = []
    for p in root.rglob("*"):
        if any(part in SKIP for part in p.relative_to(root).parts):
            continue
        if p.is_file() and p.suffix[1:] in EXTS and p.stat().st_size < 200_000:
            out.append(p.relative_to(root).as_posix())
    out.sort()
    return out[:max_files]


def ground_truth(root: str, files):
    s = set()
    for f in files:
        try:
            c = (Path(root) / f).read_text(errors="replace")
        except OSError:
            c = ""
        s.update(scan(f, c))
    return s


# ---- mock model (identical behaviour to model.rs::Mock) ----------------------------

TRACE: Path | None = None


def trace(line: str):
    if TRACE:
        with TRACE.open("a") as fh:
            fh.write(line + "\n")


def slow():
    ms = os.environ.get("AUDITOR_SLOW_MS")
    if ms:
        time.sleep(int(ms) / 1000)


def model_plan(path: str, contents: str) -> list[str]:
    trace(f"model plan {path}")
    slow()
    return [c for c, _ in CATEGORIES]


def tool_count(contents: str, category: str) -> int:
    return contents.count(dict(CATEGORIES)[category])


def model_analyze(path: str, contents: str, counts: list[tuple[str, int]]) -> str:
    trace(f"model analyze {path}")
    slow()
    findings = [(path, c, n) for c, n in counts if n > 0]
    note = f"## {path}\n{len(contents.splitlines())} lines, {len(contents)} bytes.\n"
    if not findings:
        note += "No findings.\n"
    for f in findings:
        note += fline(f) + "\n"
        needle = dict(CATEGORIES)[f[1]]
        shown = 0
        for i, l in enumerate(contents.splitlines()):
            if needle in l:
                note += f"    L{i + 1}: {l.strip()[:90]}\n"
                shown += 1
                if shown == 2:
                    break
    return note


def model_summarize(notes, budget: int) -> str:
    trace(f"model summarize {len(notes)} notes to {budget}")
    slow()
    allf = sorted(
        {f for n in notes for f in map(parse_f, n.splitlines()) if f},
        key=lambda f: (-SEVERITY.get(f[1], 0), -f[2], f[0]),
    )
    allowance = budget // 2
    out, kept = "", 0
    for f in allf:
        line = fline(f)
        if approx_tokens(out) + approx_tokens(line) + 1 > allowance:
            break
        out += line + "\n"
        kept += 1
    return f"## compacted ({kept} of {len(allf)} findings retained; {len(notes)} notes squashed)\n{out}"


def model_report(notes) -> str:
    trace(f"model report from {len(notes)} notes")
    slow()
    fs = sorted({f for n in notes for f in map(parse_f, n.splitlines()) if f})
    return f"# Audit report\n\n{len(fs)} findings.\n\n" + "".join(fline(f) + "\n" for f in fs)


# ---- graph ---------------------------------------------------------------------------


class S(TypedDict, total=False):
    root: str
    budget: int
    pending: list[str]
    done: list[str]
    notes: list[str]
    report: str | None
    compactions: int
    model_calls: int
    tokens_in: int
    metrics: dict


def ctx_size(s: S) -> int:
    return sum(approx_tokens(n) for n in s["notes"])


def step(s: S) -> dict:
    """One agent step. This node is LangGraph's unit of durability: a checkpoint is
    written after it returns. Anything that happens inside it (the file read, the
    model call) is re-done if the process dies before the checkpoint."""
    if s["report"] is not None:
        return {}
    if not s["pending"]:
        report = model_report(s["notes"])
        tokens = sum(approx_tokens(n) for n in s["notes"])
        truth = ground_truth(s["root"], s["done"])
        found = {f for f in map(parse_f, report.splitlines()) if f}
        hit = len(truth & found)
        metrics = {
            "recall": 1.0 if not truth else hit / len(truth),
            "precision": 1.0 if not found else hit / len(found),
            "compactions": s["compactions"],
            "model_calls": s["model_calls"] + 1,
            "tokens_in": s["tokens_in"] + tokens,
            "files": len(s["done"]),
            "budget": s["budget"],
        }
        return {
            "report": report,
            "model_calls": s["model_calls"] + 1,
            "tokens_in": s["tokens_in"] + tokens,
            "metrics": metrics,
        }
    if ctx_size(s) > s["budget"]:
        summary = model_summarize(s["notes"], s["budget"])
        return {
            "notes": [summary],
            "compactions": s["compactions"] + 1,
            "model_calls": s["model_calls"] + 1,
            "tokens_in": s["tokens_in"] + ctx_size(s),
        }
    # One ReAct-style step: read (tool), plan (model), count (tools), analyze (model).
    # All of it lives in one node because that is the natural LangGraph structure; the
    # checkpoint is written when the node returns.
    path = s["pending"][0]
    trace(f"tool read {path}")
    contents = (Path(s["root"]) / path).read_text(errors="replace")
    plan = model_plan(path, contents)
    counts = []
    for cat in plan:
        trace(f"tool count {path} {cat}")
        counts.append((cat, tool_count(contents, cat)))
    note = model_analyze(path, contents, counts)
    return {
        "pending": s["pending"][1:],
        "done": s["done"] + [path],
        "notes": s["notes"] + [note],
        "model_calls": s["model_calls"] + 2,
        "tokens_in": s["tokens_in"] + 2 * approx_tokens(contents) + ctx_size(s),
    }


def route(s: S) -> str:
    return END if s.get("report") is not None else "step"


def build(db: Path):
    conn = sqlite3.connect(str(db), check_same_thread=False)
    saver = SqliteSaver(conn)
    g = StateGraph(S)
    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", route, {"step": "step", END: END})
    return g.compile(checkpointer=saver)


def cfg(thread: str, checkpoint_id: str | None = None):
    c = {"configurable": {"thread_id": thread}, "recursion_limit": 100_000}
    if checkpoint_id:
        c["configurable"]["checkpoint_id"] = checkpoint_id
    return c


def describe(s: S) -> str:
    notes_f = {f for n in s.get("notes", []) for f in map(parse_f, n.splitlines()) if f}
    out = (
        f"root={s.get('root')} budget={s.get('budget')} done={len(s.get('done', []))} "
        f"pending={len(s.get('pending', []))} notes={len(s.get('notes', []))} "
        f"ctx_tokens={ctx_size(s) if 'notes' in s else 0} compactions={s.get('compactions')} "
        f"model_calls={s.get('model_calls')} tokens_in={s.get('tokens_in')}\n"
        f"findings in notes: {len(notes_f)}\n"
    )
    if s.get("report"):
        rf = {f for f in map(parse_f, s["report"].splitlines()) if f}
        out += f"report: {len(rf)} findings, {len(s['report'])} bytes\n"
    return out


# ---- commands -------------------------------------------------------------------------


def cmd_run(a):
    graph = build(a.db)
    c = cfg(a.thread)
    snap = graph.get_state(c)
    if not snap.values:
        if not a.repo:
            sys.exit("fresh thread: --repo is required")
        root = str(Path(a.repo).resolve())
        files = list_files(Path(root), a.max_files)
        print(f"init: {len(files)} files under {root}, budget {a.budget}", file=sys.stderr)
        init = dict(root=root, budget=a.budget, pending=files, done=[], notes=[], report=None,
                    compactions=0, model_calls=0, tokens_in=0, metrics={})
        t0 = time.time()
        graph.invoke(init, c)
    else:
        t0 = time.time()
        graph.invoke(None, c)  # resume from the last checkpoint
    final = graph.get_state(c).values
    print(describe(final))
    print(f"wall: {int((time.time() - t0) * 1000)} ms  checkpoints: {sum(1 for _ in graph.get_state_history(c))}")
    print("metrics:", "  ".join(f"{k}={v}" for k, v in final.get("metrics", {}).items()))


def cmd_history(a):
    graph = build(a.db)
    hist = list(graph.get_state_history(cfg(a.thread)))
    for i, s in enumerate(reversed(hist)):
        v = s.values
        print(f"{i:>5} {s.config['configurable']['checkpoint_id']} next={s.next} done={len(v.get('done', []))} "
              f"notes={len(v.get('notes', []))} compactions={v.get('compactions')}")


def cmd_at(a):
    graph = build(a.db)
    hist = list(reversed(list(graph.get_state_history(cfg(a.thread)))))
    s = hist[a.index]
    print(describe(s.values))
    if a.notes:
        print(f"--- notes at checkpoint {a.index} ---")
        for n in s.values.get("notes", []):
            print(n)


def fork_at(graph, thread: str, index: int, budget: int, new_thread: str):
    """Fork = copy the checkpoint's state into a new thread with a new budget. LangGraph's
    native fork (update_state on a past checkpoint) stays inside the same thread_id, which
    makes the matrix awkward to read back; a new thread is the closer analogue to a lithify
    branch. Either way the past is shared as *state*, not as replayable effects."""
    hist = list(reversed(list(graph.get_state_history(cfg(thread)))))
    src = hist[index].values
    values = dict(src)
    values["budget"] = budget
    c = cfg(new_thread)
    graph.update_state(c, values, as_node="step")
    return c


def cmd_fork(a):
    graph = build(a.db)
    name = a.name or f"{a.thread}-fork-{a.index}-{a.budget}"
    fork_at(graph, a.thread, a.index, a.budget, name)
    print(f"thread {name!r} forked from {a.thread!r} at checkpoint {a.index} with budget {a.budget}")


def cmd_sweep(a):
    graph = build(a.db)
    rows = []
    for b in a.budgets:
        name = f"sweep-{b}"
        c = fork_at(graph, a.thread, a.index, b, name)
        t0 = time.time()
        graph.invoke(None, c)
        v = graph.get_state(c).values
        m = v["metrics"]
        print(f"thread {name}: {m['compactions']} compactions, recall {m['recall']:.3f}, "
              f"{int((time.time() - t0) * 1000)} ms", file=sys.stderr)
        for k, val in m.items():
            rows.append((name, k, val, b))
    print("thread,metric,value,budget")
    for r in rows:
        print(",".join(map(str, r)))


def cmd_matrix(a):
    # No cross-thread index exists in LangGraph's checkpointer API; enumerate threads by
    # reading the sqlite file directly.
    conn = sqlite3.connect(str(a.db))
    threads = [r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")]
    graph = build(a.db)
    print("thread,metric,value,budget")
    for t in sorted(threads):
        v = graph.get_state(cfg(t)).values
        for k, val in v.get("metrics", {}).items():
            print(f"{t},{k},{val},{v.get('budget')}")


def cmd_crash_demo(a):
    tr = a.db.with_suffix(".trace")
    for p in (a.db, tr, a.db.with_name(a.db.name + "-wal"), a.db.with_name(a.db.name + "-shm")):
        if p.exists():
            p.unlink()
    env = dict(os.environ, AUDITOR_SLOW_MS="20")
    rng = random.Random(1)
    for k in range(1, a.kills + 1):
        child = subprocess.Popen(
            [sys.executable, __file__, "--db", str(a.db), "run", "--repo", str(a.repo), "--budget", str(a.budget),
             "--max-files", str(a.max_files), "--trace", str(tr)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        wait = a.interval_ms + rng.randrange(a.interval_ms)
        time.sleep(wait / 1000)
        child.kill()  # SIGKILL
        child.wait()
        graph = build(a.db)
        v = graph.get_state(cfg("main")).values
        n = sum(1 for _ in graph.get_state_history(cfg("main")))
        print(f"kill {k}: SIGKILL after {wait} ms -> {n} checkpoints ({len(v.get('done', []))} files analysed, "
              f"{v.get('compactions')} compactions)")
    print("resuming to completion in-process...")
    global TRACE
    TRACE = tr
    graph = build(a.db)
    graph.invoke(None, cfg("main"))
    v = graph.get_state(cfg("main")).values
    print(f"done: {len(v['done'])} files, {v['compactions']} compactions, {v['model_calls']} model calls recorded in state")
    lines = tr.read_text().splitlines() if tr.exists() else []
    counts = {}
    for l in lines:
        counts[l] = counts.get(l, 0) + 1
    model_extra = sum(c - 1 for k, c in counts.items() if k.startswith("model") and c > 1)
    tool_extra = sum(c - 1 for k, c in counts.items() if k.startswith("tool") and c > 1)
    print(f"trace: {len(lines)} invocations for {len(counts)} distinct calls across {a.kills + 1} process lifetimes")
    print(f"re-executed: {model_extra} model call(s), {tool_extra} tool call(s)")
    dups = sorted(k for k, c in counts.items() if k.startswith("model") and c > 1)
    if dups:
        print(f"  model calls run more than once: {dups}")
    print("metrics:", "  ".join(f"{k}={v}" for k, v in v.get("metrics", {}).items()))


def main():
    p = argparse.ArgumentParser(description="LangGraph + SqliteSaver auditor (comparison baseline)")
    p.add_argument("--db", type=Path, default=Path("auditor.sqlite"))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--repo", type=Path)
    r.add_argument("--thread", default="main")
    r.add_argument("--budget", type=int, default=3000)
    r.add_argument("--max-files", type=int, default=80)
    r.add_argument("--trace", type=Path)
    r.set_defaults(fn=cmd_run)

    h = sub.add_parser("history")
    h.add_argument("--thread", default="main")
    h.set_defaults(fn=cmd_history)

    at = sub.add_parser("at")
    at.add_argument("--index", type=int, required=True, help="checkpoint index (0 = initial)")
    at.add_argument("--thread", default="main")
    at.add_argument("--notes", action="store_true")
    at.set_defaults(fn=cmd_at)

    f = sub.add_parser("fork")
    f.add_argument("--index", type=int, required=True)
    f.add_argument("--budget", type=int, required=True)
    f.add_argument("--thread", default="main")
    f.add_argument("--name")
    f.set_defaults(fn=cmd_fork)

    sw = sub.add_parser("sweep")
    sw.add_argument("--index", type=int, required=True)
    sw.add_argument("--budgets", type=lambda s: [int(x) for x in s.split(",")], required=True)
    sw.add_argument("--thread", default="main")
    sw.set_defaults(fn=cmd_sweep)

    sub.add_parser("matrix").set_defaults(fn=cmd_matrix)

    cd = sub.add_parser("crash-demo")
    cd.add_argument("--repo", type=Path, required=True)
    cd.add_argument("--kills", type=int, default=3)
    cd.add_argument("--budget", type=int, default=3000)
    cd.add_argument("--max-files", type=int, default=80)
    cd.add_argument("--interval-ms", type=int, default=150)
    cd.set_defaults(fn=cmd_crash_demo)

    a = p.parse_args()
    global TRACE
    if getattr(a, "trace", None):
        TRACE = a.trace
    a.fn(a)


if __name__ == "__main__":
    main()
