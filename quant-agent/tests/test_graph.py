"""End-to-end tests of the LangGraph loop with the offline stub LLM."""
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from quantloop.config import Config, Limits
from quantloop.context import FORBIDDEN_SUBSTRINGS
from quantloop.data import synthetic_snapshot
from quantloop.graph import build_graph
from quantloop.run import make_deps
from quantloop.safety.sandbox import SandboxResult

TRENDY = 0.00004     # synthetic market with a real, persistent trend component


def run(tmp_path, cfg=None, snapshot=None, thread="t1", sandbox=None, name="run"):
    cfg = cfg or Config()
    deps = make_deps(tmp_path / name, cfg, snapshot=snapshot, sandbox=sandbox)
    app = build_graph(deps, InMemorySaver())
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 500}
    return deps, app, config, app.invoke({"run_id": thread}, config)


def test_full_loop_debugs_and_reports_honestly(tmp_path):
    deps, _, _, out = run(tmp_path)
    rep = out["report"]
    variants = rep["variants"]
    # the three injected bugs were each caught and repaired by the self-debug loop
    routes = [e for e in deps.events.read() if e["kind"] == "route" and e["to"] == "debugger"]
    assert {r["why"] for r in routes} >= {"E_RUNTIME", "E_LOOKAHEAD_STATIC", "E_LOOKAHEAD_DYNAMIC"}
    assert any(v["debug_attempts"] > 0 and v["status"] != "abandoned" for v in variants)
    # identical code is never counted as a new trial
    assert rep["n_trials_on_snapshot"] == len([v for v in variants if v["status"] != "abandoned"])
    # counted loops respected
    assert len(rep["ideas"]) <= deps.cfg.limits.max_ideas
    per_idea = {}
    for v in variants:
        per_idea[v["idea_id"]] = per_idea.get(v["idea_id"], 0) + 1
    assert max(per_idea.values()) <= deps.cfg.limits.max_refines_per_idea + 1
    # both refine arms ran, so reflection lift is measurable
    assert rep["reflection_arms"]["reflect"]["n"] > 0 and rep["reflection_arms"]["control"]["n"] > 0
    assert (tmp_path / "run" / f"report_{rep['run_id']}.json").exists()


def test_llm_never_sees_holdout_information(tmp_path):
    deps, _, _, out = run(tmp_path, Config(limits=Limits(max_ideas=2)), synthetic_snapshot(Config().splits,
                          trend_strength=TRENDY))
    assert out.get("oos")                      # OOS was actually evaluated in this run...
    blob = json.dumps(deps.llm.backend.payloads).lower()
    for s in FORBIDDEN_SUBSTRINGS:             # ...but no prompt ever carried holdout keys
        assert f'"{s}' not in blob
    oos_values = {str(r["oos_sharpe"]) for r in out["oos"]}
    assert not any(v in blob for v in oos_values)


def test_hitl_interrupt_validates_resume_and_lockbox_is_single_use(tmp_path):
    cfg = Config(limits=Limits(max_ideas=2))
    snap = synthetic_snapshot(cfg.splits, trend_strength=TRENDY)
    deps, app, config, out = run(tmp_path, cfg, snap)
    assert "__interrupt__" in out, "expected a champion and a pending human review"
    packet = out["__interrupt__"][0].value
    assert packet["lockbox"]["status"] == "evaluated" and "code" in packet
    out = app.invoke(Command(resume={"decision": "maybe"}), config)          # invalid -> asked again
    assert "__interrupt__" in out and out["__interrupt__"][0].value["previous_error"]
    out = app.invoke(Command(resume={"decision": "approve", "reviewer": "qa",
                                     "rationale": "ok for paper trading only"}), config)
    assert out["report"]["review"]["decision"] == "approve"

    # a second run on the same snapshot and ledger cannot spend the lockbox again
    app2 = build_graph(deps, InMemorySaver())
    c2 = {"configurable": {"thread_id": "t2"}, "recursion_limit": 500}
    out2 = app2.invoke({"run_id": "t2"}, c2)
    lb = (out2.get("lockbox") or (out2.get("__interrupt__") and out2["__interrupt__"][0].value["lockbox"]))
    if lb:
        assert lb.get("status") == "unavailable" or lb.get("variant_id") == packet["champion"]
    assert deps.ledger.oos_count(snap.snapshot_id) <= cfg.limits.max_oos_per_snapshot


def test_budget_exhaustion_halts_gracefully(tmp_path):
    _, _, _, out = run(tmp_path, Config(limits=Limits(usd=0.08)))
    rep = out["report"]
    assert rep["halt_reason"] in ("budget", "budget_usd")
    assert 0 < rep["spent"]["usd"] <= 0.08          # pre-call estimates keep spend under the cap
    assert rep["variants"]                          # it did real work before stopping


class BrokenInfra:
    def run(self, code, df, grid):
        return SandboxResult(False, "E_INFRA", "docker daemon not reachable")


def test_infra_failure_halts_without_poisoning_lessons(tmp_path):
    _, _, _, out = run(tmp_path, sandbox=BrokenInfra())
    rep = out["report"]
    assert rep["halt_reason"] == "infra"
    assert rep["lessons"] == [] and len(rep["ideas"]) == 1


def test_resume_from_sqlite_checkpoint_in_a_fresh_app(tmp_path):
    """Pause at human review, rebuild everything from disk (as a new process would), resume."""
    from quantloop.run import checkpointer_for

    cfg = Config(limits=Limits(max_ideas=2))
    snap = synthetic_snapshot(cfg.splits, trend_strength=TRENDY)
    config = {"configurable": {"thread_id": "p1"}, "recursion_limit": 500}
    deps = make_deps(tmp_path / "persist", cfg, snapshot=snap)
    out = build_graph(deps, checkpointer_for(deps.run_dir)).invoke({"run_id": "p1"}, config)
    assert "__interrupt__" in out
    n_trials = deps.ledger.n_trials(snap.snapshot_id)

    deps2 = make_deps(tmp_path / "persist", cfg, snapshot=snap)
    app2 = build_graph(deps2, checkpointer_for(deps2.run_dir))
    assert app2.get_state(config).next == ("human_review",)
    out = app2.invoke(Command(resume={"decision": "reject", "reviewer": "qa",
                                      "rationale": "drawdown too deep for mandate"}), config)
    assert out["report"]["review"]["decision"] == "reject"
    assert deps2.ledger.n_trials(snap.snapshot_id) == n_trials      # resume did not re-run research
