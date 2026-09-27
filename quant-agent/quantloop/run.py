"""Wiring + CLI.

    python -m quantloop.run --run-dir runs/demo            # offline stub LLM, synthetic data
    python -m quantloop.run --run-dir runs/demo --approve  # auto-answer the human review (demo only)

Real LLMs: build `LLMClient(LangChainBackend({"strong": chat_model, "cheap": chat_model2}), ...)`
and pass it to `make_deps`.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from langgraph.types import Command

from .config import Config
from .data import Snapshot, synthetic_snapshot
from .graph import build_graph
from .ledger import Ledger
from .llm import LLMClient
from .nodes import Deps
from .prompts import VERSION
from .safety.sandbox import SubprocessSandbox
from .store import ArtifactStore, EventLog
from .stub import StubBackend


def make_deps(run_dir: str | Path, cfg: Config | None = None, snapshot: Snapshot | None = None,
              llm: LLMClient | None = None, sandbox=None, allow_unsafe_local_sandbox: bool = False) -> Deps:
    """`SubprocessSandbox` does not isolate network or filesystem. It is only the default
    for the offline stub (whose code we wrote); real LLM output must run in `DockerSandbox`."""
    cfg = cfg or Config()
    if llm is not None and sandbox is None and not allow_unsafe_local_sandbox:
        raise ValueError("real LLM output needs an isolating sandbox: pass sandbox=DockerSandbox(image=...) "
                         "or allow_unsafe_local_sandbox=True for local experiments")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    events = EventLog(run_dir / "events.jsonl")
    if llm is None:
        llm = LLMClient(StubBackend(), cfg.pricing.table, VERSION,
                        on_call=lambda **kw: events.emit("llm_call", **{k: v for k, v in kw.items() if k != "payload"}))
    return Deps(cfg=cfg, snapshot=snapshot or synthetic_snapshot(cfg.splits),
                ledger=Ledger(run_dir / "ledger.sqlite"), store=ArtifactStore(run_dir / "artifacts"),
                sandbox=sandbox or SubprocessSandbox(cfg.limits.sandbox_timeout_s),
                llm=llm, events=events, run_dir=run_dir)


def checkpointer_for(run_dir: Path):
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver(sqlite3.connect(str(run_dir / "checkpoints.sqlite"), check_same_thread=False))
    except ImportError:
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/demo")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--approve", action="store_true", help="auto-approve human review (demo only)")
    args = ap.parse_args()

    deps = make_deps(args.run_dir)
    app = build_graph(deps, checkpointer_for(deps.run_dir))
    run_id = args.run_id or uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 500}

    snap = app.get_state(config)
    if snap.values.get("report") and not snap.next:
        print(f"run {run_id} already finished; see {deps.run_dir}/report_{run_id}.json")
        return
    result = app.invoke({"run_id": run_id} if not snap.next else None, config)
    while "__interrupt__" in result:
        packet = result["__interrupt__"][0].value
        print("\n=== HUMAN REVIEW REQUIRED ===")
        print(json.dumps({k: v for k, v in packet.items() if k != "code"}, indent=2, default=str))
        print("--- code ---\n" + packet["code"])
        if args.approve:
            answer = {"decision": "approve", "reviewer": "demo-auto", "rationale": "auto-approved in offline demo"}
        else:
            answer = {"decision": input("approve/reject: ").strip(), "reviewer": input("reviewer: ").strip(),
                      "rationale": input("rationale: ").strip()}
        result = app.invoke(Command(resume=answer), config)

    rep = result["report"]
    print("\n=== REPORT ===")
    print(json.dumps({k: rep[k] for k in ("run_id", "stop_reason", "halt_reason", "final_threshold", "n_trials_on_snapshot", "eligible",
                                          "finalists", "oos", "champion", "lockbox", "review",
                                          "reflection_arms", "reflection_lift", "lessons", "spent",
                                          "threshold_curve")}, indent=2, default=str))
    for idea in rep["ideas"]:
        print(f"  idea {idea['idea_id']:>16}  {idea['family']:<15} {idea['outcome']}")
    for v in rep["variants"]:
        print(f"  {v['variant_id']:>20} {v['arm']:<8} {v['status']:<9} "
              f"sr={v.get('oof_sharpe', float('nan')):>6} dsr={v.get('dsr', float('nan'))!s:>6} "
              f"fails={v.get('gate_fails', v.get('reason'))}")


if __name__ == "__main__":
    main()
