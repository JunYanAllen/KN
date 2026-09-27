"""候選架構 C：族群演化式搜尋 — LangGraph 骨架（可執行，外部服務以 stub 代替）。
驗證環境：langgraph==1.2.12
"""
from __future__ import annotations

import hashlib
import math
import operator
import random
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Overwrite, RetryPolicy, Send, interrupt

# ───────────────────────── 常數（寫死在程式，不給 LLM 改） ─────────────────────────
N_ISLANDS = 4
POP_PER_GEN = 16            # 每代扇出數 N
ELITES_PER_ISLAND = 3       # 每島精英 k
MAX_GENERATIONS = 20
MIN_GENERATIONS = 5
MAX_DEBUG_ATTEMPTS = 3
MAX_TRIALS_PER_RUN = 400    # 試驗預算（DSR 門檻隨 N_eff 上升）
STAGNATION_GENS = 4
STAGNATION_EPS = 0.05
MIGRATE_EVERY = 3
HITL_EVERY = 5              # 期中人工檢查（只看 IS）
LAMBDA_COMPLEXITY = 0.15
LAMBDA_PLATEAU = 0.50
LAMBDA_NOVELTY = 1.00
CORR_DUP = 0.95             # 訊號相關 ≥ 此值視為重複
CORR_NICHE = 0.85           # 同島精英彼此相關上限
N_FINALISTS = 3
GATE = dict(dsr_min=0.95, dsr_raw_min=0.80, pbo_max=0.30, min_trades=100, max_dd=0.35,
            pos_fold_ratio=0.5, oos_sr_min=0.5, oos_is_ratio_min=0.4,
            oos_dd_mult_max=1.5, max_params=6, max_complexity=400)
BUDGET = dict(usd=60.0, tokens=6_000_000, wall_s=4 * 3600, backtests=6000)
OPERATOR_MIX = {"mutate_logic": 0.40, "crossover": 0.20, "hypothesis_test": 0.25,
                "fresh": 0.15}      # 參數擾動由程式在 worker 內做，不佔 LLM 名額


# ───────────────────────────── reducers ─────────────────────────────
def sum_costs(a: dict, b: dict) -> dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + v
    return out


def merge_by_id(a: dict, b: dict) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


def cap_list(limit: int):
    def _r(a: list, b: list) -> list:
        return ((a or []) + (b or []))[-limit:]
    return _r


# ───────────────────────────── State ─────────────────────────────
class Budget(TypedDict):
    usd: float
    tokens: int
    wall_s: float
    backtests: int


class VariantResult(TypedDict):
    variant_id: str
    generation: int
    island_id: int
    operator: str
    parent_ids: list[str]
    hypothesis_id: Optional[str]
    status: Literal["ok", "dead", "duplicate", "degenerate"]
    code_ref: Optional[str]          # artifact store key (程式碼)
    returns_ref: Optional[str]       # artifact store key (IS 日報酬序列)
    signal_hash: Optional[str]
    metrics: dict                    # fold 級小指標：median_sr, worst_sr, pos_fold_ratio, n_trades, max_dd, complexity, n_params, plateau_ratio, tags
    debug_attempts: int
    error_hashes: list[str]
    counted_as_trial: bool


class EliteRef(TypedDict):
    variant_id: str
    island_id: int
    fitness: float
    code_ref: str
    tags: list[str]


class Hypothesis(TypedDict):
    hyp_id: str
    statement: str
    mechanism: str
    evidence_pair_ids: list[str]
    predicted: dict                  # {"edit": tag, "metric": "median_sr", "direction": "+", "min_effect": 0.05}
    status: Literal["proposed", "testing", "confirmed", "refuted", "rejected"]
    born_gen: int
    tests: list[dict]                # [{variant_id, parent_id, delta}]


class EvoState(TypedDict):
    run_id: str
    data_snapshot_id: str
    engine_version: str
    prompt_versions: dict
    generation: int
    phase: Literal["evolve", "finalize", "done"]
    # 當代扇入緩衝：reducer 累加；select_elites 用 Overwrite([]) 清空
    inbox: Annotated[list[VariantResult], operator.add]
    plan: list[dict]                               # 當代 VariantTask（無 reducer，覆寫）
    islands: dict                                  # {island_id: [EliteRef]}（只放小資料）
    hall_of_fame: list[EliteRef]
    hypotheses: Annotated[dict, merge_by_id]       # {hyp_id: Hypothesis}
    lessons_digest: str                            # ≤ 800 tokens，反思節點覆寫
    reflection_raw: Optional[dict]                 # LLM 反思原始輸出（驗證後清空；全文另存 artifact）
    pitfalls: list[str]                            # 常見錯誤（除錯統計彙整）
    gen_stats: Annotated[list[dict], cap_list(50)] # 每代摘要（觀測用）
    best_fitness_hist: list[float]
    ledger_id: str                                 # 全域試驗帳本 (DB) 句柄
    n_trials: int                                  # 本 run 計數（權威值在帳本）
    n_eff: float
    counters: Annotated[dict, sum_costs]           # debug_iters / improve / reideate / oos_peeks / lockbox
    costs: Annotated[dict, sum_costs]              # usd / tokens / backtests
    wall_start: float
    budget: Budget
    finalists: list[dict]
    champion: Optional[dict]
    review_ref: Optional[str]
    human_decision: Optional[dict]
    stop_reason: Optional[str]


class VariantTask(TypedDict):
    """Send 扇出時每個 worker 拿到的 payload（只含小資料與 artifact key）。"""
    run_id: str
    variant_id: str
    generation: int
    slot: int
    island_id: int
    operator: Literal["mutate_logic", "crossover", "hypothesis_test", "fresh"]
    parent_ids: list[str]
    parent_code_refs: list[str]
    hypothesis: Optional[dict]        # hypothesis_test 時帶 {hyp_id, predicted, test_plan}
    instruction: str                  # 由程式模板填寫的突變指令
    lessons_digest: str
    pitfalls: list[str]
    data_snapshot_id: str
    engine_version: str
    models: dict                      # {"codegen": "...", "debug": "..."}
    limits: dict                      # {max_debug, max_tokens, timeout_s}


class WorkerState(TypedDict):
    task: VariantTask
    code_ref: Optional[str]
    last_error: Optional[dict]        # {kind, msg, trace_ref, hash}
    error_hashes: list[str]
    error_kinds: list[str]
    debug_attempts: int
    eval_out: Optional[dict]
    # ↓ 與父圖同名的鍵 → 子圖結束時寫回父圖（經父圖 reducer 累加）
    inbox: Annotated[list[VariantResult], operator.add]
    costs: Annotated[dict, sum_costs]
    counters: Annotated[dict, sum_costs]


# ───────────────────────────── 外部服務 stub ─────────────────────────────
class Stub:
    """真實系統：LLM(with_structured_output)、sandbox runner、artifact store、ledger DB。"""
    rng = random.Random(7)

    @staticmethod
    def llm_codegen(task: VariantTask) -> dict:
        return {"code": f"# strategy {task['variant_id']}", "param_space": {"n": [10, 50]},
                "rationale": "...", "tokens": 7000, "usd": 0.03}

    @staticmethod
    def llm_debug(code: str, err: dict) -> dict:
        return {"code": code + "\n# fixed", "tokens": 3000, "usd": 0.004}

    @staticmethod
    def static_checks(code: str) -> Optional[dict]:
        return None if Stub.rng.random() > 0.2 else {"kind": "SchemaError", "msg": "PARAM_SPACE missing"}

    @staticmethod
    def sandbox_eval(code_ref: str) -> dict:
        if Stub.rng.random() < 0.1:
            return {"error": {"kind": "RuntimeError", "msg": "KeyError: 'close_'"}}
        sr = Stub.rng.gauss(0.6, 0.5)
        return {"metrics": {"median_sr": sr, "worst_sr": sr - 0.8, "pos_fold_ratio": 0.66,
                            "n_trades": 240, "max_dd": 0.2, "complexity": Stub.rng.randint(80, 300),
                            "n_params": 4, "plateau_ratio": Stub.rng.uniform(0.4, 1.0),
                            "tags": Stub.rng.sample(["atr_stop", "funding", "vol_filter", "ema_x", "rsi"], 2)},
                "returns_ref": f"ret/{code_ref}", "signal_hash": hashlib.md5(code_ref.encode()).hexdigest()[:12],
                "backtests": 14}

    @staticmethod
    def ledger_upsert(result: dict) -> None: ...          # 以 variant_id 為主鍵，冪等

    @staticmethod
    def ledger_stats(ledger_id: str) -> dict:
        return {"n_trials": 0, "n_eff": 0.0, "sr_var": 0.25}

    @staticmethod
    def max_corr_to(returns_ref: str, others: list[str]) -> float:
        return Stub.rng.uniform(0, 0.9) if others else 0.0

    @staticmethod
    def dsr(returns_ref: str, n_eff: float, sr_var: float) -> float:
        return Stub.rng.uniform(0.5, 0.99)

    @staticmethod
    def pbo_cscv(returns_refs: list[str]) -> float:
        return Stub.rng.uniform(0.1, 0.5)

    @staticmethod
    def oos_eval(code_ref: str) -> dict:
        return {"sr": Stub.rng.gauss(0.5, 0.5), "max_dd": 0.2, "net_ret": 0.1}

    @staticmethod
    def lockbox_eval(code_ref: str) -> dict:
        return {"sr": 0.7, "max_dd": 0.18}

    @staticmethod
    def llm_reflect(digest: dict) -> dict:
        return {"hypotheses": [{"hyp_id": f"h{digest['generation']}_0", "statement": "ATR 停損降低最差 fold 虧損",
                                "mechanism": "高波動期截斷尾部", "evidence_pair_ids": ["p1", "p2"],
                                "predicted": {"edit": "atr_stop", "metric": "worst_sr", "direction": "+", "min_effect": 0.1}}],
                "anti_patterns": [], "retire": [], "lessons_digest": "…", "tokens": 30000, "usd": 0.25}


def now() -> float:
    import time
    return time.time()


# ───────────────────────────── worker 子圖（除錯短迴圈） ─────────────────────────────
def w_codegen(s: WorkerState) -> dict:
    out = Stub.llm_codegen(s["task"])
    code_ref = f"code/{s['task']['variant_id']}/v0"
    return {"code_ref": code_ref, "debug_attempts": 0, "error_hashes": [], "error_kinds": [],
            "costs": {"usd": out["usd"], "tokens": out["tokens"]}}


def _err_hash(err: dict) -> str:
    import re
    norm = re.sub(r"\d+", "N", f"{err['kind']}:{err['msg']}")
    return hashlib.sha1(norm.encode()).hexdigest()[:10]


def w_check_and_run(s: WorkerState) -> dict:
    """確定性：AST/schema/白名單 → 沙盒（截斷不變性 + IS walk-forward + 參數鄰域）。"""
    err = Stub.static_checks(s["code_ref"])
    if err is None:
        r = Stub.sandbox_eval(s["code_ref"])
        if "error" in r:
            err = r["error"]
        else:
            return {"last_error": None, "eval_out": r, "costs": {"backtests": r["backtests"]}}
    err["hash"] = _err_hash(err)
    return {"last_error": err, "eval_out": None}


def w_debug(s: WorkerState) -> dict:
    """小模型；只看程式碼 + traceback + 違反的規則，看不到任何績效。"""
    out = Stub.llm_debug(s["code_ref"], s["last_error"])
    n = s["debug_attempts"] + 1
    return {"code_ref": f"code/{s['task']['variant_id']}/v{n}", "debug_attempts": n,
            "error_hashes": s["error_hashes"] + [s["last_error"]["hash"]],
            "error_kinds": s["error_kinds"] + [s["last_error"]["kind"]],
            "costs": {"usd": out["usd"], "tokens": out["tokens"]}, "counters": {"debug_iters": 1}}


def route_after_run(s: WorkerState) -> Literal["w_debug", "w_emit"]:
    err = s["last_error"]
    if err is None:
        return "w_emit"
    if s["debug_attempts"] >= s["task"]["limits"]["max_debug"]:
        return "w_emit"                              # 放棄 → dead
    if err["hash"] in s["error_hashes"]:
        return "w_emit"                              # 同一錯誤重複 → 不再燒錢
    if err["kind"] == "LookaheadDetected" and "LookaheadDetected" in s["error_kinds"]:
        return "w_emit"                              # 前視偏差只給一次修正機會
    return "w_debug"


def w_emit(s: WorkerState) -> dict:
    t = s["task"]
    ok = s["eval_out"] is not None
    m = s["eval_out"]["metrics"] if ok else {}
    status = "ok" if ok else "dead"
    if ok and m.get("n_trades", 0) < 30:
        status = "degenerate"
    res: VariantResult = {
        "variant_id": t["variant_id"], "generation": t["generation"], "island_id": t["island_id"],
        "operator": t["operator"], "parent_ids": t["parent_ids"],
        "hypothesis_id": (t["hypothesis"] or {}).get("hyp_id"), "status": status,
        "code_ref": s["code_ref"], "returns_ref": s["eval_out"]["returns_ref"] if ok else None,
        "signal_hash": s["eval_out"]["signal_hash"] if ok else None, "metrics": m,
        "debug_attempts": s["debug_attempts"], "error_hashes": s["error_hashes"],
        "counted_as_trial": ok,                     # 有績效就是試驗（含 degenerate）；dead 不算
    }
    Stub.ledger_upsert(res)                         # 冪等 upsert：節點重跑不會重複計數
    kind = "reideate" if t["operator"] == "fresh" else "improve"
    return {"inbox": [res], "counters": {kind: 1}}


def build_worker():
    g = StateGraph(WorkerState)
    g.add_node("w_codegen", w_codegen, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("w_check_and_run", w_check_and_run,
               retry_policy=RetryPolicy(max_attempts=2, retry_on=(TimeoutError, ConnectionError)))
    g.add_node("w_debug", w_debug, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("w_emit", w_emit)
    g.add_edge(START, "w_codegen")
    g.add_edge("w_codegen", "w_check_and_run")
    g.add_conditional_edges("w_check_and_run", route_after_run, ["w_debug", "w_emit"])
    g.add_edge("w_debug", "w_check_and_run")
    g.add_edge("w_emit", END)
    return g.compile()


# ───────────────────────────── 父圖節點 ─────────────────────────────
def init_run(s: EvoState) -> dict:
    # 真實：凍結資料快照 hash、切 IS/OOS/lockbox、建立帳本 run 記錄、寫入 prompt/模型/引擎版本
    return {"generation": 0, "phase": "evolve", "islands": {i: [] for i in range(N_ISLANDS)},
            "hall_of_fame": [], "lessons_digest": "", "pitfalls": [], "best_fitness_hist": [],
            "n_trials": 0, "n_eff": 0.0, "wall_start": now(), "finalists": [], "champion": None,
            "stop_reason": None}


def budget_gate(s: EvoState) -> dict:
    return {}   # 純路由用節點（決策在 route_budget）


def _remaining(s: EvoState) -> dict:
    c, b = s.get("costs", {}), s["budget"]
    return {"usd": b["usd"] - c.get("usd", 0), "tokens": b["tokens"] - c.get("tokens", 0),
            "backtests": b["backtests"] - c.get("backtests", 0),
            "wall_s": b["wall_s"] - (now() - s["wall_start"]),
            "trials": MAX_TRIALS_PER_RUN - s["n_trials"]}


def _est_gen_cost(s: EvoState) -> dict:
    hist = [g for g in s.get("gen_stats", []) if g.get("usd")]
    per_var = (sum(g["usd"] for g in hist) / max(1, sum(g["n"] for g in hist))) if hist else 0.08
    return {"usd": 1.3 * (per_var * POP_PER_GEN + 0.35), "trials": POP_PER_GEN, "backtests": 16 * POP_PER_GEN}


def route_budget(s: EvoState) -> Literal["plan_generation", "select_finalists"]:
    r, e = _remaining(s), _est_gen_cost(s)
    if s["generation"] >= MAX_GENERATIONS:
        return "select_finalists"
    if r["usd"] < e["usd"] or r["trials"] < e["trials"] or r["backtests"] < e["backtests"] \
            or r["wall_s"] < 600 or r["tokens"] < 400_000:
        return "select_finalists"
    return "plan_generation"


def plan_generation(s: EvoState) -> dict:
    """確定性：挑親代、分配 operator、把待驗證假說排進 A/B 測試名額。"""
    gen = s["generation"] + 1
    rng = random.Random(hash((s["run_id"], gen)))
    testing = [h for h in s.get("hypotheses", {}).values() if h["status"] in ("proposed", "testing")]
    tasks: list[dict] = []
    ops = list(OPERATOR_MIX)
    weights = [OPERATOR_MIX[o] for o in ops]
    for slot in range(POP_PER_GEN):
        isl = slot % N_ISLANDS
        elites = s["islands"].get(isl, [])
        op = "fresh" if not elites else rng.choices(ops, weights)[0]
        if op == "hypothesis_test" and not testing:
            op = "mutate_logic"
        if op == "crossover" and len(elites) < 2:
            op = "mutate_logic"
        parents = [] if op == "fresh" else (
            rng.sample(elites, 2) if op == "crossover" else [max(elites, key=lambda e: e["fitness"] + rng.random() * 0.2)])
        hyp = None
        if op == "hypothesis_test":
            h = testing[slot % len(testing)]
            hyp = {"hyp_id": h["hyp_id"], "predicted": h["predicted"]}
        vid = hashlib.sha1(f"{s['run_id']}|{gen}|{slot}".encode()).hexdigest()[:12]
        tasks.append(VariantTask(
            run_id=s["run_id"], variant_id=vid, generation=gen, slot=slot, island_id=isl, operator=op,
            parent_ids=[p["variant_id"] for p in parents], parent_code_refs=[p["code_ref"] for p in parents],
            hypothesis=hyp, instruction=f"{op} template v3", lessons_digest=s["lessons_digest"],
            pitfalls=s["pitfalls"][-8:], data_snapshot_id=s["data_snapshot_id"],
            engine_version=s["engine_version"],
            models={"codegen": "claude-sonnet-5", "debug": "claude-haiku-4-5"},
            limits={"max_debug": MAX_DEBUG_ATTEMPTS, "max_tokens": 12000, "timeout_s": 300}))
    return {"plan": tasks, "generation": gen}


def fan_out(s: EvoState) -> list[Send]:
    """每個 VariantTask 變成一個 Send → 平行執行 variant_worker 子圖。"""
    return [Send("variant_worker", {"task": t, "inbox": [], "costs": {}, "counters": {},
                                    "error_hashes": [], "error_kinds": [], "debug_attempts": 0,
                                    "code_ref": None, "last_error": None, "eval_out": None})
            for t in s["plan"]]


def _fitness(m: dict) -> float:
    """寫死的適應度：穩健統計 + 複雜度懲罰 + 參數高原懲罰。"""
    return (m["median_sr"] + 0.5 * min(m["worst_sr"], 0.0)
            - LAMBDA_COMPLEXITY * (m["complexity"] / 100.0)
            - LAMBDA_PLATEAU * (1.0 - m["plateau_ratio"]))


def _eligible(m: dict) -> bool:
    return (m["n_trades"] >= GATE["min_trades"] and m["max_dd"] <= GATE["max_dd"]
            and m["pos_fold_ratio"] >= GATE["pos_fold_ratio"] and m["n_params"] <= GATE["max_params"]
            and m["complexity"] <= GATE["max_complexity"])


def select_elites(s: EvoState) -> dict:
    """確定性扇入：去重、計分、島內 niching 精英、名人堂、假說判定、帳本統計。"""
    results = s["inbox"]
    seen_hash = {e.get("signal_hash") for isl in s["islands"].values() for e in isl}
    islands = {k: list(v) for k, v in s["islands"].items()}
    scored = []
    for r in results:
        if r["status"] != "ok" or r["signal_hash"] in seen_hash or not _eligible(r["metrics"]):
            continue
        seen_hash.add(r["signal_hash"])
        f = _fitness(r["metrics"])
        corr = Stub.max_corr_to(r["returns_ref"], [e.get("returns_ref") for e in islands[r["island_id"]]])
        f_rank = f - LAMBDA_NOVELTY * max(0.0, corr - 0.6)       # 新穎性懲罰只用於排序
        scored.append((f_rank, f, corr, r))
    for isl in islands:
        pool = [(e["fitness"], e) for e in islands[isl]]
        pool += [(fr, {"variant_id": r["variant_id"], "island_id": isl, "fitness": f, "code_ref": r["code_ref"],
                       "tags": r["metrics"]["tags"], "returns_ref": r["returns_ref"],
                       "signal_hash": r["signal_hash"]})
                 for fr, f, c, r in scored if r["island_id"] == isl and c < CORR_DUP]
        pool.sort(key=lambda x: -x[0])
        chosen: list = []
        for _, e in pool:
            if len(chosen) >= ELITES_PER_ISLAND:
                break
            if all(Stub.max_corr_to(e.get("returns_ref"), [c.get("returns_ref")]) < CORR_NICHE for c in chosen):
                chosen.append(e)
        islands[isl] = chosen
    # 假說判定（預先登記的預測 vs. 實測 Δ）
    hyp_updates = {}
    parents_fit = {e["variant_id"]: e["fitness"] for isl in s["islands"].values() for e in isl}
    for r in results:
        hid = r["hypothesis_id"]
        if not hid or r["status"] != "ok":
            continue
        h = dict(s["hypotheses"][hid])
        base = parents_fit.get(r["parent_ids"][0]) if r["parent_ids"] else None
        if base is None:
            continue
        h["tests"] = h["tests"] + [{"variant_id": r["variant_id"], "delta": _fitness(r["metrics"]) - base}]
        h["status"] = _judge_hypothesis(h)
        hyp_updates[hid] = h
    stats = Stub.ledger_stats(s["ledger_id"])
    n_trials = s["n_trials"] + sum(r["counted_as_trial"] for r in results)
    all_elites = [e for v in islands.values() for e in v]
    hof = sorted({e["variant_id"]: e for e in s["hall_of_fame"] + all_elites}.values(),
                 key=lambda e: -e["fitness"])[:10]
    best = hof[0]["fitness"] if hof else float("-inf")
    return {"inbox": Overwrite([]),                         # 清空扇入緩衝
            "islands": islands, "hall_of_fame": hof, "hypotheses": hyp_updates,
            "n_trials": n_trials, "n_eff": max(stats["n_eff"], math.sqrt(n_trials)),
            "best_fitness_hist": s["best_fitness_hist"] + [best],
            "gen_stats": [{"gen": s["generation"], "n": len(results), "ok": len(scored), "best": best,
                           "dead": sum(r["status"] == "dead" for r in results), "usd": None}]}


def _judge_hypothesis(h: dict) -> str:
    """程式判定：≥3 次測試且 ≥2/3 方向一致且中位 Δ ≥ min_effect → confirmed。"""
    ds = sorted(t["delta"] for t in h["tests"])
    if len(ds) < 3:
        return "testing"
    sign_ok = sum(d > 0 for d in ds) / len(ds) if h["predicted"]["direction"] == "+" else sum(d < 0 for d in ds) / len(ds)
    med = ds[len(ds) // 2]
    if sign_ok >= 2 / 3 and abs(med) >= h["predicted"]["min_effect"]:
        return "confirmed"
    return "refuted" if len(ds) >= 4 else "testing"


def population_reflect(s: EvoState) -> dict:
    """LLM（強模型）：輸入確定性建好的 PopulationDigest；輸出結構化假說。"""
    digest = {"generation": s["generation"], "contrast_pairs": [], "tag_attribution": [],
              "hypothesis_outcomes": [h for h in s["hypotheses"].values() if h["status"] in ("confirmed", "refuted")]}
    out = Stub.llm_reflect(digest)
    return {"reflection_raw": out, "costs": {"usd": out["usd"], "tokens": out["tokens"]}}


def validate_reflection(s: EvoState) -> dict:
    """確定性：證據引用核對、方向核對、去重、數量上限；通過者 status=proposed。"""
    out = s["reflection_raw"]
    new = {}
    for h in out["hypotheses"][:3]:
        if len(h["evidence_pair_ids"]) < 2:
            continue                                         # 至少兩組對照證據
        new[h["hyp_id"]] = {**h, "status": "proposed", "born_gen": s["generation"], "tests": []}
    merged = {**s["hypotheses"], **new}
    return {"hypotheses": new, "lessons_digest": render_lessons(merged), "reflection_raw": None}


def render_lessons(hyps: dict, max_items: int = 12) -> str:
    """lessons 由程式從假說帳本渲染：只有『已驗證/已否證』能成為 lesson，LLM 自由文字不進 prompt。"""
    rows = []
    for h in sorted(hyps.values(), key=lambda h: -len(h["tests"])):
        if h["status"] in ("confirmed", "refuted"):
            ds = sorted(t["delta"] for t in h["tests"])
            rows.append(f"[{h['status'].upper()}] {h['predicted']['edit']} → {h['predicted']['metric']} "
                        f"{h['predicted']['direction']} (n={len(ds)}, medianΔ={ds[len(ds)//2]:+.2f}) :: {h['statement'][:120]}")
    return "\n".join(rows[:max_items])


def island_maintenance(s: EvoState) -> dict:
    """確定性：定期遷移（環狀、需低相關）與停滯島重置。"""
    islands = {k: list(v) for k, v in s["islands"].items()}
    if s["generation"] % MIGRATE_EVERY == 0:
        for i in islands:
            src = islands[(i - 1) % N_ISLANDS]
            if src:
                mig = dict(src[0], island_id=i)
                if all(Stub.max_corr_to(mig.get("returns_ref"), [e.get("returns_ref")]) < 0.7 for e in islands[i]):
                    islands[i] = (islands[i] + [mig])[:ELITES_PER_ISLAND + 1]
    return {"islands": islands}


def route_after_generation(s: EvoState) -> Literal["budget_gate", "checkpoint_review", "select_finalists"]:
    """寫死的停止規則：IS 達標提早停（再搜只會墊高 DSR 門檻）、停滯、定期 HITL。"""
    g, hist = s["generation"], s["best_fitness_hist"]
    if g >= MIN_GENERATIONS and s["hall_of_fame"]:
        top = s["hall_of_fame"][0]
        if Stub.dsr(top.get("returns_ref"), s["n_eff"], 0.25) >= GATE["dsr_min"] and g >= MIN_GENERATIONS + 2:
            return "select_finalists"
    if len(hist) > STAGNATION_GENS and max(hist[-STAGNATION_GENS:]) - max(hist[:-STAGNATION_GENS]) < STAGNATION_EPS \
            and g >= MIN_GENERATIONS:
        return "select_finalists"
    if g % HITL_EVERY == 0:
        return "checkpoint_review"
    return "budget_gate"


def checkpoint_review(s: EvoState) -> dict:
    """期中 HITL：只呈現 IS 儀表板（報告已由上游寫入 artifact）；本節點無副作用。"""
    decision = interrupt({"type": "midrun", "generation": s["generation"], "n_trials": s["n_trials"],
                          "n_eff": s["n_eff"], "hof": [e["variant_id"] for e in s["hall_of_fame"][:5]]})
    return {"human_decision": decision}


def route_after_checkpoint(s: EvoState) -> Literal["budget_gate", "select_finalists"]:
    d = s.get("human_decision") or {}
    return "select_finalists" if d.get("action") == "stop" else "budget_gate"


def select_finalists(s: EvoState) -> dict:
    """確定性：從名人堂挑 ≤3 個彼此低相關者，預先登記順序（依 IS DSR），之後才看 OOS。"""
    cands = []
    for e in s["hall_of_fame"]:
        d = Stub.dsr(e.get("returns_ref"), s["n_eff"], 0.25)          # 主：有效試驗數
        d_raw = Stub.dsr(e.get("returns_ref"), s["n_trials"], 0.25)   # 敏感度：原始試驗數
        cands.append({**e, "dsr": d, "dsr_raw": d_raw})
    cands.sort(key=lambda e: -e["dsr"])
    fin = []
    for e in cands:
        if len(fin) >= N_FINALISTS:
            break
        if all(Stub.max_corr_to(e.get("returns_ref"), [f.get("returns_ref")]) < 0.7 for f in fin):
            fin.append(e)
    pbo = Stub.pbo_cscv([e.get("returns_ref") for e in s["hall_of_fame"]])
    for f in fin:
        f["pbo"] = pbo
    return {"finalists": fin, "phase": "finalize"}


def oos_evaluate(s: EvoState) -> dict:
    """每個 finalist 恰好 1 次 OOS（帳本記錄 oos_peek，重跑冪等）。結果不回饋 LLM。"""
    out = []
    for f in s["finalists"]:
        o = Stub.oos_eval(f["code_ref"])
        out.append({**f, "oos": o})
    return {"finalists": out, "counters": {"oos_peeks": len(out)}}


def pick_champion(s: EvoState) -> dict:
    """依『預先登記順序』取第一個通過全部門檻者；OOS 只當 pass/fail 過濾器。"""
    for f in s["finalists"]:
        o = f["oos"]
        passed = (f["dsr"] >= GATE["dsr_min"] and f["dsr_raw"] >= GATE["dsr_raw_min"]
                  and f["pbo"] <= GATE["pbo_max"]
                  and o["sr"] >= GATE["oos_sr_min"] and o["net_ret"] > 0
                  and o["max_dd"] <= GATE["oos_dd_mult_max"] * GATE["max_dd"])
        if passed:
            return {"champion": f}
    return {"champion": None, "stop_reason": "no_finalist_passed"}


def route_champion(s: EvoState) -> Literal["lockbox_eval", "prepare_review"]:
    return "lockbox_eval" if s["champion"] else "prepare_review"


def lockbox_eval(s: EvoState) -> dict:
    lb = Stub.lockbox_eval(s["champion"]["code_ref"])      # 每資料快照僅一次；結果只進報告
    return {"champion": {**s["champion"], "lockbox": lb}, "counters": {"lockbox": 1}}


def prepare_review(s: EvoState) -> dict:
    """有副作用（寫報告 artifact）的步驟放在 interrupt 節點之前。"""
    return {"review_ref": f"reports/{s['run_id']}.html"}


def human_review(s: EvoState) -> dict:
    decision = interrupt({"type": "final", "review_ref": s["review_ref"],
                          "champion": (s["champion"] or {}).get("variant_id"),
                          "options": ["approve_paper_trading", "reject", "archive"]})
    return {"human_decision": decision, "phase": "done"}


# ───────────────────────────── 建圖 ─────────────────────────────
def build_graph(checkpointer=None):
    g = StateGraph(EvoState)
    g.add_node("init_run", init_run)
    g.add_node("budget_gate", budget_gate)
    g.add_node("plan_generation", plan_generation)
    g.add_node("variant_worker", build_worker())            # 子圖作為節點，被 Send 扇出
    g.add_node("select_elites", select_elites)
    g.add_node("population_reflect", population_reflect, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("validate_reflection", validate_reflection)
    g.add_node("island_maintenance", island_maintenance)
    g.add_node("checkpoint_review", checkpoint_review)
    g.add_node("select_finalists", select_finalists)
    g.add_node("oos_evaluate", oos_evaluate)
    g.add_node("pick_champion", pick_champion)
    g.add_node("lockbox_eval", lockbox_eval)
    g.add_node("prepare_review", prepare_review)
    g.add_node("human_review", human_review)

    g.add_edge(START, "init_run")
    g.add_edge("init_run", "budget_gate")
    g.add_conditional_edges("budget_gate", route_budget, ["plan_generation", "select_finalists"])
    g.add_conditional_edges("plan_generation", fan_out, ["variant_worker"])
    g.add_edge("variant_worker", "select_elites")            # 所有 Send 任務完成後才執行一次
    g.add_edge("select_elites", "population_reflect")
    g.add_edge("population_reflect", "validate_reflection")
    g.add_edge("validate_reflection", "island_maintenance")
    g.add_conditional_edges("island_maintenance", route_after_generation,
                            ["budget_gate", "checkpoint_review", "select_finalists"])
    g.add_conditional_edges("checkpoint_review", route_after_checkpoint, ["budget_gate", "select_finalists"])
    g.add_edge("select_finalists", "oos_evaluate")
    g.add_edge("oos_evaluate", "pick_champion")
    g.add_conditional_edges("pick_champion", route_champion, ["lockbox_eval", "prepare_review"])
    g.add_edge("lockbox_eval", "prepare_review")
    g.add_edge("prepare_review", "human_review")
    g.add_edge("human_review", END)
    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "run-001"}, "recursion_limit": 300}
    init = {"run_id": "run-001", "data_snapshot_id": "btcusdt_1h_sha256:ab12", "engine_version": "bt-1.4.0",
            "prompt_versions": {"codegen": "v3", "reflect": "v2", "debug": "v1"}, "ledger_id": "ledger-main",
            "budget": BUDGET, "hypotheses": {}, "inbox": [], "costs": {}, "counters": {}, "gen_stats": [],
            "plan": []}
    res = app.invoke(init, cfg)
    while "__interrupt__" in res:
        payload = res["__interrupt__"][0].value
        print("INTERRUPT:", payload["type"], {k: payload[k] for k in payload if k != "type"})
        resume = {"action": "continue"} if payload["type"] == "midrun" else {"action": "reject", "note": "demo"}
        res = app.invoke(Command(resume=resume), cfg)
    print("gen", res["generation"], "trials", res["n_trials"], "counters", res["counters"],
          "costs", {k: round(v, 3) for k, v in res["costs"].items()}, "champion",
          (res["champion"] or {}).get("variant_id"), "stop", res["stop_reason"], "inbox", len(res["inbox"]))
