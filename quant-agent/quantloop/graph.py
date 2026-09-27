"""The research graph: a single StateGraph, three counted loops, pure routers.

    START -> init_run -> strategist -> coder -> static_check -> sandbox_validate -> backtest_is -> gate_is
                            ^                     ^    |              |                               |
                            |                     |    +--> debugger <+  (loop A: fix, <=4, same-hash<=2)
                            |                     +--------+    |                                     |
                            |                                   +--> abandon_variant                  |
                            |          coder <-- reflect / control_change <---------------------------+
                            |                    (loop B: refine <=3, stop when satisfied or stalled)  |
                            +---- close_idea <---------------------------------------------------------+
                                  (loop C: ideas <=6)
                                     |
                                     v
                                  finalize (re-deflate, shadow, pre-registered finalists, OOS once)
                                     |-- champion --> lockbox_eval --> human_review [interrupt] --> report --> END
                                     +-- none ---------------------------------------------------> report --> END

All routing decisions - including "is this good enough" - are made here, in code.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import Deps, build_nodes
from .state import ResearchState


def build_graph(deps: Deps, checkpointer=None):
    L = deps.cfg.limits
    n = build_nodes(deps)

    def over_budget(s: ResearchState) -> str | None:
        sp = s.get("spent") or {}
        if s.get("halt_reason"):
            return s["halt_reason"]
        if sp.get("usd", 0) >= L.usd:
            return "budget_usd"
        if sp.get("active_s", 0) >= L.active_seconds:
            return "budget_time"
        if deps.ledger.n_trials(deps.snapshot.snapshot_id) >= L.max_trials_per_snapshot:
            return "trial_cap"
        return None

    def log(s: ResearchState, frm: str, to: str, why: str = "") -> str:
        deps.events.emit("route", frm=frm, to=to, why=why, run_id=s.get("run_id"))
        return to

    # ------------------------------------------------------------ routers
    def after_init(s):
        return log(s, "init_run", "strategist")

    def after_strategist(s):
        if (why := over_budget(s)) or not s.get("idea"):
            return log(s, "strategist", "finalize", why or "no_idea")
        return log(s, "strategist", "coder")

    def after_coder(s):
        if why := over_budget(s):
            return log(s, "coder", "close_idea", why)
        return log(s, "coder", "static_check")

    def _debug_or_abandon(s, frm):
        v = s["variant"]
        if over_budget(s):
            return log(s, frm, "close_idea", over_budget(s))
        if v["debug_attempts"] >= L.max_debug_attempts:
            return log(s, frm, "abandon_variant", "debug_attempts")
        if max(v["error_counts"].values(), default=0) >= L.max_same_error:
            return log(s, frm, "abandon_variant", "repeated_error")
        return log(s, frm, "debugger", v["last_error"]["class"])

    def after_static(s):
        if s["variant"]["status"] == "static_ok":
            return log(s, "static_check", "sandbox_validate")
        return _debug_or_abandon(s, "static_check")

    def after_sandbox(s):
        if s.get("halt_reason"):
            return log(s, "sandbox_validate", "close_idea", s["halt_reason"])
        st = s["variant"]["status"]
        if st == "valid":
            return log(s, "sandbox_validate", "backtest_is")
        if st == "degenerate":
            return log(s, "sandbox_validate", "abandon_variant", "degenerate")
        return _debug_or_abandon(s, "sandbox_validate")

    def after_debugger(s):
        if why := over_budget(s):
            return log(s, "debugger", "close_idea", why)
        return log(s, "debugger", "static_check")

    def _refine_or_close(s, frm):
        idea = s["idea"]
        if why := over_budget(s):
            return log(s, frm, "close_idea", why)
        if idea["passed"]:
            return log(s, frm, "close_idea", "satisfied")           # stop as soon as the gate is met
        if idea["best"] is None:
            return log(s, frm, "close_idea", "no_valid_variant")
        if idea["refine_iter"] >= L.max_refines_per_idea:
            return log(s, frm, "close_idea", "refines_exhausted")
        if idea["stalls"] >= L.max_stalls:
            return log(s, frm, "close_idea", "stalled")
        global_refine = sum(1 for a in s.get("attempts", []) if a.get("kind") == "variant"
                            and a.get("arm") in ("reflect", "control"))
        if global_refine % L.control_arm_every == L.control_arm_every - 1:
            return log(s, frm, "control_change", "control_arm")
        return log(s, frm, "reflect")

    def after_gate(s):
        return _refine_or_close(s, "gate_is")

    def after_abandon(s):
        return _refine_or_close(s, "abandon_variant")

    def after_reflect(s):
        if s.get("reflection") and not over_budget(s):
            return log(s, "reflect", "coder")
        return log(s, "reflect", "close_idea", over_budget(s) or "reflection_invalid")

    def after_close(s):
        if why := over_budget(s):
            return log(s, "close_idea", "finalize", why)
        if s["idea_seq"] >= L.max_ideas:
            return log(s, "close_idea", "finalize", "ideas_exhausted")
        return log(s, "close_idea", "strategist")

    def after_finalize(s):
        return log(s, "finalize", "lockbox_eval" if s.get("champion") else "report")

    # ------------------------------------------------------------ graph
    g = StateGraph(ResearchState)
    for name, fn in n.items():
        g.add_node(name, fn)
    g.add_edge(START, "init_run")
    g.add_conditional_edges("init_run", after_init, ["strategist"])
    g.add_conditional_edges("strategist", after_strategist, ["coder", "finalize"])
    g.add_conditional_edges("coder", after_coder, ["static_check", "close_idea"])
    g.add_conditional_edges("static_check", after_static,
                            ["sandbox_validate", "debugger", "abandon_variant", "close_idea"])
    g.add_conditional_edges("sandbox_validate", after_sandbox,
                            ["backtest_is", "debugger", "abandon_variant", "close_idea"])
    g.add_conditional_edges("debugger", after_debugger, ["static_check", "close_idea"])
    g.add_edge("backtest_is", "gate_is")
    g.add_conditional_edges("gate_is", after_gate, ["reflect", "control_change", "close_idea"])
    g.add_conditional_edges("abandon_variant", after_abandon, ["reflect", "control_change", "close_idea"])
    g.add_conditional_edges("reflect", after_reflect, ["coder", "close_idea"])
    g.add_edge("control_change", "coder")
    g.add_conditional_edges("close_idea", after_close, ["strategist", "finalize"])
    g.add_conditional_edges("finalize", after_finalize, ["lockbox_eval", "report"])
    g.add_edge("lockbox_eval", "human_review")
    g.add_edge("human_review", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)
