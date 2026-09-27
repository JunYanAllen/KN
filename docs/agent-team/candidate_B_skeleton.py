"""Architecture B skeleton: 3 nested StateGraphs, compiled subgraphs added as nodes.
LLM / sandbox / engine are stubs so topology, state passing, checkpointer, interrupt are testable."""
from __future__ import annotations
import hashlib, json, operator, re
from dataclasses import dataclass, field
from typing import Annotated, Literal
from typing_extensions import TypedDict, NotRequired  # pydantic needs this on py<3.12 (interrupt response_schema)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, RetryPolicy, interrupt
from langgraph.errors import NodeError
from langgraph.runtime import Runtime

# ============ shared small types & reducers ============
class Cost(TypedDict, total=False):
    usd: float; in_tok: int; out_tok: int; backtests: int; wall_s: float

def add_cost(a: Cost | None, b: Cost | None) -> Cost:
    a, b = a or {}, b or {}
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("usd", "in_tok", "out_tok", "backtests", "wall_s")}

def over(spent: Cost, cap: Cost) -> bool:
    return any(spent.get(k, 0) >= v for k, v in cap.items() if v is not None)

def remaining(cap: Cost, spent: Cost) -> Cost:
    return {k: max(0, v - spent.get(k, 0)) for k, v in cap.items()}

def min_cost(a: Cost, b: Cost) -> Cost:
    return {k: min(a.get(k, float("inf")), b.get(k, float("inf"))) for k in set(a) | set(b)}

class Lesson(TypedDict):
    key: str; scope: Literal["family", "mechanism", "implementation", "oos_category"]
    family: str; claim: str; evidence: Literal["measured_is", "oos_category", "infeasible"]

LESSON_CAP = 12
def merge_lessons(old: list[Lesson] | None, new: list[Lesson] | None) -> list[Lesson]:
    seen: dict[str, Lesson] = {l["key"]: l for l in (old or [])}
    for l in new or []:
        seen.pop(l["key"], None); seen[l["key"]] = l      # newest wins, move to end
    return list(seen.values())[-LESSON_CAP:]

class StrategySpec(TypedDict):
    idea_id: str; family: str; thesis: str; signal_logic: str; param_space: dict

class VariantRecord(TypedDict):
    variant_id: str; code_ref: str; code_hash: str; params: dict
    wf_sharpe: float; wf_sharpe_se: float; pos_fold_frac: float; min_fold_trades: int
    max_dd: float; cost_share: float; neighbor_ratio: float; n_params: int; diag_ref: str

# ============ layer boundary contracts ("mailbox" keys) ============
class DebugRequest(TypedDict):
    variant_id: str; spec: StrategySpec; mode: Literal["generate", "patch"]
    base_code_ref: str | None; change: dict | None; allowance: Cost; max_attempts: int

class DebugResult(TypedDict):
    variant_id: str; status: Literal["ok", "gave_up", "budget", "crashed"]
    code_ref: str | None; code_hash: str | None; error_class: str | None
    error_digest: str | None; attempts: int; spent: Cost

class RefineRequest(TypedDict):
    idea_id: str; spec: StrategySpec; allowance: Cost; max_iters: int

class RefineResult(TypedDict):
    idea_id: str; status: Literal["candidate", "no_candidate", "infeasible", "budget", "crashed"]
    champion: VariantRecord | None; pbo: float | None; n_trials: int
    lessons: list[Lesson]; iters: int; spent: Cost

# ============ runtime context: non-serialisable deps live here, NOT in state ============
@dataclass
class Deps:
    run_id: str
    store: dict = field(default_factory=dict)          # artifact store stub
    ledger: dict = field(default_factory=dict)         # trial ledger stub (idempotent by trial_id)
    oos_tokens: set = field(default_factory=set)       # single-use OOS permits
    lockbox_used: bool = False
    fake_fail_first: bool = True

def put_artifact(d: Deps, kind: str, obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    aid = f"{kind}:{hashlib.sha1(blob.encode()).hexdigest()[:12]}"
    d.store[aid] = obj
    return aid

# =====================================================================
# LAYER 1: Debug subgraph   coder <-> validate <-> sandbox_run
# =====================================================================
MAX_DEBUG_ATTEMPTS = 4
SAME_HASH_GIVE_UP = 3
LOOKAHEAD_GIVE_UP = 2

class ErrorRecord(TypedDict):
    attempt: int; stage: Literal["validate", "sandbox"]; cls: str; hash: str; digest: str

class CheckReport(TypedDict):
    ok: bool; stage: str; cls: str | None; raw: str | None

class FixPlan(TypedDict):
    tier: Literal["small", "mid"]; strategy: Literal["patch", "explain_then_patch", "regenerate"]

class DebugIn(TypedDict):
    debug_request: DebugRequest
class DebugOut(TypedDict):
    debug_result: DebugResult
class DebugState(DebugIn, DebugOut):
    attempt: NotRequired[int]
    code_ref: NotRequired[str | None]
    code_hash: NotRequired[str | None]
    last_check: NotRequired[CheckReport | None]
    fix_plan: NotRequired[FixPlan | None]
    errors: Annotated[list[ErrorRecord], operator.add]
    spent: Annotated[Cost, add_cost]

def d_coder(s: DebugState, runtime: Runtime[Deps]) -> dict:
    """LLM. generate / patch / fix. stub: returns code text."""
    req, plan = s["debug_request"], s.get("fix_plan")
    code = f"# {req['spec']['idea_id']} {req['variant_id']} attempt={s.get('attempt', 0)} plan={plan}\n"
    if runtime.context.fake_fail_first and s.get("attempt", 0) == 0:
        code += "BUG"
    ref = put_artifact(runtime.context, "code", code)
    return {"code_ref": ref, "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16],
            "attempt": s.get("attempt", 0) + 1,
            "spent": {"usd": 0.04 if (plan or {}).get("tier") != "small" else 0.02, "in_tok": 6000, "out_tok": 3000}}

def d_validate(s: DebugState, runtime: Runtime[Deps]) -> dict:
    """deterministic: AST scan, import whitelist, schema of PARAM_SPACE."""
    return {"last_check": {"ok": True, "stage": "validate", "cls": None, "raw": None}}

def d_sandbox_run(s: DebugState, runtime: Runtime[Deps]) -> dict:
    """deterministic: throwaway container, smoke run + truncation-invariance + output contract."""
    code = runtime.context.store[s["code_ref"]]
    if "BUG" in code:
        return {"last_check": {"ok": False, "stage": "sandbox", "cls": "E_RUNTIME",
                               "raw": "KeyError: 'close_ma' at line 12 (value=0x7f..)"}}
    return {"last_check": {"ok": True, "stage": "sandbox", "cls": None, "raw": None}}

def normalize_error(raw: str) -> str:
    raw = re.sub(r"0x[0-9a-f]+|\d+(\.\d+)?", "N", raw)
    raw = re.sub(r"'[^']*'", "S", raw)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]

def d_triage(s: DebugState) -> dict:
    """deterministic: classify, hash, choose escalation. No LLM."""
    chk = s["last_check"]
    h = normalize_error(chk["raw"] or chk["cls"])
    rec: ErrorRecord = {"attempt": s["attempt"], "stage": chk["stage"], "cls": chk["cls"],
                        "hash": h, "digest": (chk["raw"] or "")[:600]}
    n_same = 1 + sum(e["hash"] == h for e in s.get("errors", []))
    left = s["debug_request"]["max_attempts"] - s["attempt"]
    if left <= 1 or n_same >= 2 and chk["cls"] in ("E_OUTPUT_SCHEMA", "E_LOOKAHEAD"):
        plan: FixPlan = {"tier": "mid", "strategy": "regenerate"}
    elif n_same >= 2:
        plan = {"tier": "mid", "strategy": "explain_then_patch"}
    else:
        plan = {"tier": "small", "strategy": "explain_then_patch" if chk["cls"] == "E_RUNTIME" else "patch"}
    return {"errors": [rec], "fix_plan": plan}

def route_after_check(s: DebugState) -> Literal["sandbox_run", "finish", "triage"]:
    chk = s["last_check"]
    if chk["ok"]:
        return "sandbox_run" if chk["stage"] == "validate" else "finish"
    return "triage"

def route_after_triage(s: DebugState) -> Literal["coder", "finish"]:
    req, errs = s["debug_request"], s["errors"]
    last = errs[-1]
    if over(s.get("spent", {}), req["allowance"]):                      return "finish"
    if s["attempt"] >= req["max_attempts"]:                              return "finish"
    if last["cls"] == "E_INFRA":                                         return "finish"
    if sum(e["hash"] == last["hash"] for e in errs) >= SAME_HASH_GIVE_UP: return "finish"
    if sum(e["cls"] == "E_LOOKAHEAD" for e in errs) >= LOOKAHEAD_GIVE_UP: return "finish"
    return "coder"

def d_finish(s: DebugState) -> dict:
    req, chk, errs = s["debug_request"], s.get("last_check"), s.get("errors", [])
    ok = bool(chk and chk["ok"] and chk["stage"] == "sandbox")
    status = "ok" if ok else ("budget" if over(s.get("spent", {}), req["allowance"]) else "gave_up")
    return {"debug_result": {
        "variant_id": req["variant_id"], "status": status,
        "code_ref": s.get("code_ref") if ok else None, "code_hash": s.get("code_hash") if ok else None,
        "error_class": None if ok else (errs[-1]["cls"] if errs else None),
        "error_digest": None if ok else (errs[-1]["digest"][:300] if errs else None),
        "attempts": s.get("attempt", 0), "spent": s.get("spent", {})}}

def build_debug_graph():
    g = StateGraph(DebugState, context_schema=Deps, input_schema=DebugIn, output_schema=DebugOut)
    g.add_node("coder", d_coder, retry_policy=RetryPolicy(max_attempts=3))       # API 429/5xx only
    g.add_node("validate", d_validate)
    g.add_node("sandbox_run", d_sandbox_run, retry_policy=RetryPolicy(max_attempts=2))  # infra flake
    g.add_node("triage", d_triage)
    g.add_node("finish", d_finish)
    g.add_edge(START, "coder")
    g.add_edge("coder", "validate")
    g.add_conditional_edges("validate", route_after_check, ["sandbox_run", "finish", "triage"])
    g.add_conditional_edges("sandbox_run", route_after_check, ["sandbox_run", "finish", "triage"])
    g.add_conditional_edges("triage", route_after_triage, ["coder", "finish"])
    g.add_edge("finish", END)
    return g.compile()     # checkpointer=None -> inherit parent's, fresh namespace per invocation

# =====================================================================
# LAYER 2: Refine subgraph  debug -> backtest_IS(WF) -> gate -> reflect -> ...
# =====================================================================
IS_GATE = dict(min_wf_sharpe=1.0, min_pos_fold_frac=0.67, min_fold_trades=15, max_dd=0.25,
               max_cost_share=0.5, min_neighbor_ratio=0.7, max_params=4)
PLATEAU_PATIENCE = 2
REFUTED_STREAK_MAX = 3
MIN_IMPROVE = 0.10

class GateReport(TypedDict):
    passed: bool; failed_rules: list[str]; improved: bool

class Reflection(TypedDict):
    mechanism: str; change: dict; prediction: dict; evidence: list[dict]; valid: bool; fingerprint: str

class RefineIn(TypedDict):
    refine_request: RefineRequest
class RefineOut(TypedDict):
    refine_result: RefineResult
class RefineState(RefineIn, RefineOut):
    debug_request: NotRequired[DebugRequest]       # mailbox -> child
    debug_result: NotRequired[DebugResult]         # mailbox <- child
    iter: NotRequired[int]
    current: NotRequired[VariantRecord | None]
    champion: NotRequired[VariantRecord | None]
    champion_passed: NotRequired[bool]
    gate: NotRequired[GateReport | None]
    reflection: NotRequired[Reflection | None]
    no_progress: NotRequired[int]
    refuted_streak: NotRequired[int]
    history: Annotated[list[dict], operator.add]
    verdicts: Annotated[list[dict], operator.add]
    tabu: Annotated[list[str], operator.add]
    spent: Annotated[Cost, add_cost]

def r_init(s: RefineState) -> dict:
    return {"iter": 0, "champion": None, "current": None, "no_progress": 0, "refuted_streak": 0, "reflection": None}

def r_prepare_debug(s: RefineState) -> dict:
    req = s["refine_request"]; ref = s.get("reflection")
    base = s.get("champion")
    return {"debug_request": {
        "variant_id": f"{req['idea_id']}-v{s['iter']}", "spec": req["spec"],
        "mode": "patch" if ref else "generate",
        "base_code_ref": base["code_ref"] if base else None,
        "change": ref["change"] if ref else None,
        "allowance": min_cost({"usd": 0.40, "wall_s": 900}, remaining(req["allowance"], s.get("spent", {}))),
        "max_attempts": MAX_DEBUG_ATTEMPTS}}

def r_absorb_debug(s: RefineState) -> dict:
    return {"spent": s["debug_result"]["spent"], "iter": s["iter"] + 1}

def r_backtest_is(s: RefineState, runtime: Runtime[Deps]) -> dict:
    """deterministic engine: anchored walk-forward, purge+embargo, inner param selection, neighbor grid.
    Writes every evaluated (variant, params) to the trial ledger (idempotent key)."""
    dr, it = s["debug_result"], s["iter"]
    wf = 0.6 + 0.3 * it
    rec: VariantRecord = {"variant_id": dr["variant_id"], "code_ref": dr["code_ref"], "code_hash": dr["code_hash"],
                          "params": {"n": 20 + it}, "wf_sharpe": wf, "wf_sharpe_se": 0.08, "pos_fold_frac": 0.83,
                          "min_fold_trades": 30, "max_dd": 0.18, "cost_share": 0.3, "neighbor_ratio": 0.8,
                          "n_params": 3, "diag_ref": "diag:stub"}
    for combo in range(16):
        tid = hashlib.sha1(f"{runtime.context.run_id}|{dr['code_hash']}|{combo}".encode()).hexdigest()[:12]
        runtime.context.ledger[tid] = {"variant": dr["variant_id"], "combo": combo}
    return {"current": rec, "history": [{k: rec[k] for k in ("variant_id", "wf_sharpe", "max_dd")}],
            "spent": {"backtests": 16 * 6}}

def r_score_prediction(s: RefineState) -> dict:
    ref = s.get("reflection")
    if not ref or not ref["valid"]:
        return {}
    cur, champ = s["current"], s["champion"]
    p = ref["prediction"]
    delta = cur[p["metric"]] - champ[p["metric"]]
    ok = delta >= p["min_delta"] if p["direction"] == "increase" else -delta >= p["min_delta"]
    return {"verdicts": [{"fp": ref["fingerprint"], "confirmed": ok, "delta": delta}],
            "refuted_streak": 0 if ok else s["refuted_streak"] + 1,
            "tabu": [] if ok else [ref["fingerprint"]]}

def r_gate(s: RefineState) -> dict:
    c, champ, G = s["current"], s.get("champion"), IS_GATE
    failed = [name for name, bad in [
        ("wf_sharpe", c["wf_sharpe"] < G["min_wf_sharpe"]),
        ("pos_fold_frac", c["pos_fold_frac"] < G["min_pos_fold_frac"]),
        ("fold_trades", c["min_fold_trades"] < G["min_fold_trades"]),
        ("max_dd", c["max_dd"] > G["max_dd"]),
        ("cost_share", c["cost_share"] > G["max_cost_share"]),
        ("neighbor", c["neighbor_ratio"] < G["min_neighbor_ratio"]),
        ("n_params", c["n_params"] > G["max_params"])] if bad]
    improved = champ is None or c["wf_sharpe"] > champ["wf_sharpe"] + max(MIN_IMPROVE, 2 * c["wf_sharpe_se"])
    upd = {"gate": {"passed": not failed, "failed_rules": failed, "improved": improved}}
    if improved:
        upd |= {"champion": c, "champion_passed": not failed, "no_progress": 0}
    else:
        upd |= {"no_progress": s["no_progress"] + 1}
    return upd

def route_after_debug(s: RefineState) -> Literal["backtest_is", "reflect", "finalize"]:
    dr, req = s["debug_result"], s["refine_request"]
    if dr["status"] in ("budget", "crashed"):     return "finalize"
    if dr["status"] == "gave_up":
        if s.get("champion") is None:             return "finalize"   # idea cannot even be implemented
        if s["iter"] >= req["max_iters"]:         return "finalize"
        return "reflect"                                               # variant infeasible -> rethink change
    return "backtest_is"

def route_after_gate(s: RefineState) -> Literal["reflect", "finalize"]:
    req, g = s["refine_request"], s["gate"]
    if g["passed"] and g["improved"]:
        return "finalize"                                  # satisficing: stop at first IS pass
    if over(s.get("spent", {}), req["allowance"]):       return "finalize"
    if s["iter"] >= req["max_iters"]:                    return "finalize"
    if s["no_progress"] >= PLATEAU_PATIENCE:             return "finalize"
    if s["refuted_streak"] >= REFUTED_STREAK_MAX:        return "finalize"
    return "reflect"

def r_reflect(s: RefineState) -> dict:
    """LLM (strong). Input: diagnostic pack (IS only) + hypothesis ledger + tabu. Output: Reflection schema."""
    change = {"kind": "regime_filter", "target": "entry", "desc": "skip entries when ATR pct > 90th"}
    fp = hashlib.sha1(json.dumps(change, sort_keys=True).encode()).hexdigest()[:10] + f"-{s['iter']}"
    return {"reflection": {"mechanism": "regime_filter", "change": change,
                           "prediction": {"metric": "wf_sharpe", "direction": "increase", "min_delta": 0.1},
                           "evidence": [{"path": "by_regime.high_vol.sharpe", "observed": -0.4}],
                           "valid": True, "fingerprint": fp},
            "spent": {"usd": 0.09, "in_tok": 10000, "out_tok": 1500}}

def r_verify_reflection(s: RefineState) -> dict:
    """deterministic: evidence numbers must match diagnostic pack; fingerprint not in tabu; single change."""
    ref = dict(s["reflection"])
    if ref["fingerprint"] in s.get("tabu", []):
        ref["valid"] = False
    return {"reflection": ref}

def route_after_verify(s: RefineState) -> Literal["prepare_debug", "finalize"]:
    ref = s["reflection"]
    if ref["mechanism"] == "abandon_idea":   return "finalize"
    if not ref["valid"] and s["no_progress"] + 1 >= PLATEAU_PATIENCE:   return "finalize"
    return "prepare_debug"

def r_finalize(s: RefineState, runtime: Runtime[Deps]) -> dict:
    req, champ, g = s["refine_request"], s.get("champion"), s.get("gate")
    dr = s.get("debug_result")
    if champ is None:
        status = {"budget": "budget", "crashed": "crashed"}.get(dr["status"] if dr else "", "infeasible")
    else:
        status = "candidate" if s.get("champion_passed") else "no_candidate"
    lessons: list[Lesson] = []
    for v in s.get("verdicts", []):
        lessons.append({"key": f"{req['spec']['family']}|{v['fp'][:10]}", "scope": "mechanism",
                        "family": req["spec"]["family"],
                        "claim": f"change {v['fp'][:10]} {'helped' if v['confirmed'] else 'did not help'} (IS)",
                        "evidence": "measured_is"})
    n_trials = sum(1 for t in runtime.context.ledger.values() if t["variant"].startswith(req["idea_id"]))
    return {"refine_result": {"idea_id": req["idea_id"], "status": status,
                              "champion": champ if status == "candidate" else None,
                              "pbo": 0.2, "n_trials": n_trials, "lessons": lessons[-3:],
                              "iters": s.get("iter", 0), "spent": s.get("spent", {})}}

def debug_crash_handler(s: RefineState, error: NodeError) -> Command:
    req = s["debug_request"]
    return Command(update={"debug_result": {"variant_id": req["variant_id"], "status": "crashed", "code_ref": None,
                                            "code_hash": None, "error_class": "E_FRAMEWORK",
                                            "error_digest": repr(error.error)[:300], "attempts": 0, "spent": {}}},
                   goto="absorb_debug")

def build_refine_graph(debug_graph):
    g = StateGraph(RefineState, context_schema=Deps, input_schema=RefineIn, output_schema=RefineOut)
    g.add_node("init", r_init)
    g.add_node("prepare_debug", r_prepare_debug)
    g.add_node("debug", debug_graph, error_handler=debug_crash_handler)   # compiled subgraph as node
    g.add_node("absorb_debug", r_absorb_debug)
    g.add_node("backtest_is", r_backtest_is)
    g.add_node("score_prediction", r_score_prediction)
    g.add_node("gate", r_gate)
    g.add_node("reflect", r_reflect, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("verify_reflection", r_verify_reflection)
    g.add_node("finalize", r_finalize)
    g.add_edge(START, "init"); g.add_edge("init", "prepare_debug")
    g.add_edge("prepare_debug", "debug"); g.add_edge("debug", "absorb_debug")
    g.add_conditional_edges("absorb_debug", route_after_debug, ["backtest_is", "reflect", "finalize"])
    g.add_edge("backtest_is", "score_prediction"); g.add_edge("score_prediction", "gate")
    g.add_conditional_edges("gate", route_after_gate, ["reflect", "finalize"])
    g.add_edge("reflect", "verify_reflection")
    g.add_conditional_edges("verify_reflection", route_after_verify, ["prepare_debug", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()

# =====================================================================
# LAYER 3: Main graph ideate -> refine -> OOS -> stat gate -> lockbox -> human_review -> report
# =====================================================================
MAX_IDEAS = 6
MAX_OOS_USES = 6
IDEA_CAP: Cost = {"usd": 1.6, "wall_s": 1800}
STAT_GATE = dict(min_oos_sharpe=0.8, min_decay_ratio=0.4, min_dsr=0.95, max_pbo=0.30, min_oos_trades=60, max_oos_dd=0.25)

class HumanDecision(TypedDict):
    decision: Literal["approve", "reject"]; reviewer: str; note: str

class MainState(TypedDict):
    run_id: str
    budget: Cost
    data_manifest: NotRequired[dict]
    idea_n: NotRequired[int]
    current_spec: NotRequired[StrategySpec | None]
    refine_request: NotRequired[RefineRequest]
    refine_result: NotRequired[RefineResult]
    oos_uses: NotRequired[int]
    oos_report: NotRequired[dict | None]
    stat_gate: NotRequired[dict | None]
    finalist: NotRequired[VariantRecord | None]
    lockbox_report: NotRequired[dict | None]          # never rendered into any LLM prompt
    human: NotRequired[HumanDecision | None]
    report_ref: NotRequired[str | None]
    lessons: Annotated[list[Lesson], merge_lessons]
    tried_families: Annotated[list[str], operator.add]
    spent: Annotated[Cost, add_cost]

def m_init(s: MainState) -> dict:
    return {"idea_n": 0, "oos_uses": 0, "finalist": None,
            "data_manifest": {"is": "2020-01..2024-06", "oos": "2024-07..2025-06", "lockbox": "2025-07..2026-06",
                              "sha256": "stub", "engine": "bt-1.4.2"}}

def m_ideate(s: MainState) -> dict:
    """LLM (strong). Input: lessons (<=12 lines), tried_families, NOT any OOS number, NOT lockbox."""
    n = s["idea_n"] + 1
    fam = ["trend", "meanrev", "carry"][n % 3]
    return {"idea_n": n, "current_spec": {"idea_id": f"i{n}", "family": fam, "thesis": "...",
                                          "signal_logic": "...", "param_space": {"n": [10, 40]}},
            "spent": {"usd": 0.09, "in_tok": 8000, "out_tok": 2000}}

def m_prepare_refine(s: MainState) -> dict:
    return {"refine_request": {"idea_id": s["current_spec"]["idea_id"], "spec": s["current_spec"],
                               "allowance": min_cost(IDEA_CAP, remaining(s["budget"], s.get("spent", {}))),
                               "max_iters": 5}}

def m_absorb_refine(s: MainState) -> dict:
    r = s["refine_result"]
    return {"spent": r["spent"], "lessons": r["lessons"], "tried_families": [s["current_spec"]["family"]]}

def m_oos_eval(s: MainState, runtime: Runtime[Deps]) -> dict:
    champ = s["refine_result"]["champion"]
    tok = champ["variant_id"]
    if tok in runtime.context.oos_tokens:
        raise RuntimeError("OOS already consumed for this variant")   # infra-level single use
    runtime.context.oos_tokens.add(tok)
    return {"oos_uses": s["oos_uses"] + 1,
            "oos_report": {"variant_id": tok, "sharpe": 1.1, "trades": 80, "max_dd": 0.12, "t_stat": 2.9}}

def m_stat_gate(s: MainState, runtime: Runtime[Deps]) -> dict:
    o, r, G = s["oos_report"], s["refine_result"], STAT_GATE
    n_trials_run = len(runtime.context.ledger)
    dsr = 0.97       # deflated_sharpe(champion WF-OOF returns, n_trials=n_trials_run, skew, kurt)
    t_req = 1.96 + 0.35 * (s["oos_uses"] - 1)        # tighter every extra OOS look
    checks = {"oos_sharpe": o["sharpe"] >= G["min_oos_sharpe"],
              "decay": o["sharpe"] >= G["min_decay_ratio"] * r["champion"]["wf_sharpe"],
              "dsr": dsr >= G["min_dsr"], "pbo": (r["pbo"] or 1) <= G["max_pbo"],
              "trades": o["trades"] >= G["min_oos_trades"], "dd": o["max_dd"] <= G["max_oos_dd"],
              "t_stat": o["t_stat"] >= t_req}
    passed = all(checks.values())
    return {"stat_gate": {"passed": passed, "checks": checks, "dsr": dsr, "n_trials": n_trials_run},
            "finalist": r["champion"] if passed else None}

def route_after_refine(s: MainState) -> Literal["oos_eval", "distill_failure"]:
    r = s["refine_result"]
    if r["status"] == "candidate" and s["oos_uses"] < MAX_OOS_USES and not over(s["spent"], s["budget"]):
        return "oos_eval"
    return "distill_failure"

def route_after_stat_gate(s: MainState) -> Literal["lockbox_eval", "distill_failure"]:
    return "lockbox_eval" if s["stat_gate"]["passed"] else "distill_failure"

def m_distill_failure(s: MainState) -> dict:
    """deterministic: OOS failures become CATEGORY-only lessons (no numbers leak back to ideate)."""
    sg, fam = s.get("stat_gate"), s["current_spec"]["family"]
    out: list[Lesson] = []
    if s["refine_result"]["status"] == "infeasible":
        out.append({"key": f"{fam}|infeasible", "scope": "implementation", "family": fam,
                    "claim": "could not be implemented within contract", "evidence": "infeasible"})
    if sg and not sg["passed"] and s["oos_report"]["variant_id"] == (s["refine_result"]["champion"] or {}).get("variant_id"):
        cats = sorted(k for k, ok in sg["checks"].items() if not ok)
        out.append({"key": f"{fam}|oos|{','.join(cats)}", "scope": "oos_category", "family": fam,
                    "claim": f"failed held-out checks: {','.join(cats)}", "evidence": "oos_category"})
    return {"lessons": out, "stat_gate": None}

def route_next_idea(s: MainState) -> Literal["ideate", "report"]:
    if s["idea_n"] >= MAX_IDEAS or over(s["spent"], s["budget"]):
        return "report"
    if s.get("refine_result", {}).get("status") == "crashed":
        return "report"                     # infra failure: stop, do not burn ideas; human fixes infra
    return "ideate"

def m_lockbox_eval(s: MainState, runtime: Runtime[Deps]) -> dict:
    if runtime.context.lockbox_used:
        raise RuntimeError("lockbox already used")
    runtime.context.lockbox_used = True
    return {"lockbox_report": {"sharpe": 0.9, "max_dd": 0.1, "trades": 70}}

def m_human_review(s: MainState) -> dict:
    """NO side effects before interrupt(): it re-executes from the top on resume."""
    decision = interrupt({"kind": "strategy_review", "run_id": s["run_id"], "finalist": s["finalist"],
                          "oos": s["oos_report"], "stat_gate": s["stat_gate"], "lockbox": s["lockbox_report"]},
                         response_schema=HumanDecision)
    return {"human": decision}

def m_report(s: MainState, runtime: Runtime[Deps]) -> dict:
    ref = put_artifact(runtime.context, "report", {"finalist": s.get("finalist"), "human": s.get("human"),
                                                  "spent": s.get("spent"), "lessons": s.get("lessons")})
    return {"report_ref": ref}

def refine_crash_handler(s: MainState, error: NodeError) -> Command:
    return Command(update={"refine_result": {"idea_id": s["refine_request"]["idea_id"], "status": "crashed",
                                             "champion": None, "pbo": None, "n_trials": 0, "lessons": [],
                                             "iters": 0, "spent": {}}},
                   goto="absorb_refine")

def build_main_graph(checkpointer):
    refine = build_refine_graph(build_debug_graph())
    g = StateGraph(MainState, context_schema=Deps)
    g.add_node("init_run", m_init)
    g.add_node("ideate", m_ideate, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("prepare_refine", m_prepare_refine)
    g.add_node("refine", refine, error_handler=refine_crash_handler)      # compiled subgraph as node
    g.add_node("absorb_refine", m_absorb_refine)
    g.add_node("oos_eval", m_oos_eval)
    g.add_node("stat_gate", m_stat_gate)
    g.add_node("distill_failure", m_distill_failure)
    g.add_node("lockbox_eval", m_lockbox_eval)
    g.add_node("human_review", m_human_review)
    g.add_node("report", m_report)
    g.add_edge(START, "init_run"); g.add_edge("init_run", "ideate")
    g.add_edge("ideate", "prepare_refine"); g.add_edge("prepare_refine", "refine")
    g.add_edge("refine", "absorb_refine")
    g.add_conditional_edges("absorb_refine", route_after_refine, ["oos_eval", "distill_failure"])
    g.add_edge("oos_eval", "stat_gate")
    g.add_conditional_edges("stat_gate", route_after_stat_gate, ["lockbox_eval", "distill_failure"])
    g.add_conditional_edges("distill_failure", route_next_idea, ["ideate", "report"])
    g.add_edge("lockbox_eval", "human_review"); g.add_edge("human_review", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    app = build_main_graph(InMemorySaver())
    deps = Deps(run_id="run-001")
    cfg = {"configurable": {"thread_id": "run-001"}, "recursion_limit": 200}
    out = app.invoke({"run_id": "run-001", "budget": {"usd": 12.0, "wall_s": 7200, "backtests": 20000},
                      "lessons": [], "tried_families": [], "spent": {}}, cfg, context=deps, durability="sync")
    print("INTERRUPT:", [i.value["kind"] for i in out.get("__interrupt__", [])])
    snap = app.get_state(cfg, subgraphs=True)
    print("next:", snap.next, "| keys in main state:", sorted(snap.values))
    assert "debug_request" not in snap.values and "history" not in snap.values, "private key leaked"
    print("spent so far:", snap.values["spent"])
    out = app.invoke(Command(resume={"decision": "approve", "reviewer": "allen", "note": "ok"}), cfg, context=deps)
    print("human:", out["human"], "report:", out["report_ref"])
    print("ledger trials:", len(deps.ledger), "oos tokens:", deps.oos_tokens)
    hist = list(app.get_state_history(cfg))
    print("root checkpoints:", len(hist))
    # subgraph checkpoints exist under namespaces
    nss = {c.config["configurable"].get("checkpoint_ns") for c in app.checkpointer.list(None)}
    print("namespaces sample:", sorted(n for n in nss if n)[:4])
