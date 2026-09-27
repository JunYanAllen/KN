"""候選架構 D 骨架：Supervisor 多代理 + 多視角批判（可執行的 stub 版，用來驗證 LangGraph 1.2.12 API 與拓撲）。
LLM / 沙盒 / 回測皆為 stub；真實版替換 llm_* 與 run_* 函式即可。
執行：python candidate_D_skeleton.py
"""
from __future__ import annotations

import hashlib
import hmac
import json
import operator
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Overwrite, RetryPolicy, Send, interrupt

# ───────────────────────── 0. 常數 / 預算（由人審定、hash 進 run_meta） ─────────────────────────
LIMITS = dict(
    max_debug_per_variant=3, max_same_error_hash=2, max_debug_total=30,
    max_refine_per_hyp=4, max_ideation=6, plateau_k=3, plateau_eps=0.05,
    max_backtests=40, max_oos_peeks=3, max_usd=25.0, max_tokens=3_000_000, max_wall_s=6 * 3600,
)
GATE_SECRET = b"held-by-orchestrator-not-in-state"      # 只有 gate 節點持有（真實版放 KMS / env）

Action = Literal["REFINE", "NEW_HYPOTHESIS", "RESURRECT", "ESCALATE_HUMAN", "STOP"]
ACTION_TO_NODE: dict[str, str] = {
    "REFINE": "coder", "NEW_HYPOTHESIS": "researcher", "RESURRECT": "coder",
    "ESCALATE_HUMAN": "prepare_review", "STOP": "prepare_review",
}
SUPERVISOR_DESTS = ("coder", "researcher", "prepare_review")


# ───────────────────────── 1. reducers ─────────────────────────
def add_counts(a: dict[str, float] | None, b: dict[str, float] | None) -> dict[str, float]:
    """並行分支可安全相加的計數 / 花費 reducer。"""
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + v
    return out


LESSON_CAP, ARCHIVE_K = 12, 5


def merge_lessons(a: list[dict] | None, b: list[dict] | None) -> list[dict]:
    """lessons 以 key 去重、保留最新，並截斷到 LESSON_CAP 條（控制 token）。reducer 必須是 (a, b) -> c。"""
    by_key = {x["key"]: x for x in (a or [])}
    for x in b or []:
        by_key[x["key"]] = x
    return sorted(by_key.values(), key=lambda x: (-x["weight"], -x["ts"]))[:LESSON_CAP]


def merge_topk(a: list[dict] | None, b: list[dict] | None) -> list[dict]:
    by_id = {x["variant_id"]: x for x in (a or []) + (b or [])}
    return sorted(by_id.values(), key=lambda x: -x["score"])[:ARCHIVE_K]


# ───────────────────────── 2. State ─────────────────────────
class State(TypedDict, total=False):
    run_meta: dict                                   # 不可變：engine/data/gate-config hash、prompt 版本
    phase: str
    spend: Annotated[dict[str, float], add_counts]   # usd / tokens_in / tokens_out / wall_s
    counters: Annotated[dict[str, float], add_counts]  # debug_total / backtests / ideation / oos_peeks ...
    hyp_counters: dict[str, dict[str, int]]          # 每個 hypothesis 的 refine / no_improve（整體覆寫）
    hypothesis: dict                                 # 當前假說（含 pre-registered prediction）
    variant: dict                                    # 當前變體：code_ref / code_hash / param_space / parent
    debug: dict                                      # {attempts, error_hashes:{h:n}, last_error, escalated}
    exec_result: dict
    diagnostics: dict                                # {pack_ref, keys:[...], headline:{...}}（小）
    round_id: str
    critiques: Annotated[list[dict], operator.add]   # 每輪以 Overwrite([]) 重置
    reflection: dict
    gate: dict                                       # 確定性 gate 結果 + HMAC token
    oos: dict                                        # 只存 pass/fail + 粗分類（LLM 可見），數字存 artifact
    lockbox_ref: str                                 # LLM 永不可見（context projection 白名單排除）
    attempts: Annotated[list[dict], operator.add]
    lessons: Annotated[list[dict], merge_lessons]
    archive: Annotated[list[dict], merge_topk]
    directive: dict                                  # supervisor → 專家的指示（有長度上限）
    supervisor_log: Annotated[list[dict], operator.add]
    critic_stats: Annotated[dict[str, float], add_counts]  # 每位批判者 valid/dropped/adopted/hit
    review: dict
    events: Annotated[list[dict], operator.add]      # 可觀測性：node, model, prompt_ver, hashes, cost


# ───────────────────────── 3. stubs（真實版換掉） ─────────────────────────
_FAKE = {"n": 0}


def llm_structured(role: str, schema: str, ctx: dict) -> tuple[dict, dict]:
    """回傳 (結構化輸出, 花費)。真實版：with_structured_output / tool use + strict schema。"""
    cost = {"usd": 0.02, "tokens_in": 6000, "tokens_out": 1200}
    if role == "supervisor":
        allowed = ctx["allowed"]
        return {"action": allowed[0], "rationale": "stub", "target": None}, cost
    if role == "researcher":
        return {"hyp_id": f"H{uuid.uuid4().hex[:4]}", "family": "trend",
                "statement": "4h 趨勢 + 15m 回撤進場",
                "prediction": {"metric": "cv_sharpe_median", "direction": "up", "min_delta": 0.1}}, cost
    if role in ("coder", "coder_debug"):
        return {"code": "def generate_signals(df, params): ...", "param_space": {"n": [10, 20, 40]}}, cost
    # critics
    return {"findings": [{
        "category": {"quant_critic": "regime", "risk_critic": "drawdown", "stat_auditor": "overfit"}[role],
        "claim": "stub", "severity": 2,
        "evidence": [{"ref": "regime.high_vol.sharpe", "op": "<", "value": 0.0}],
        "proposal": {"target": "filter", "direction": "add", "text": "高波動停用"},
        "requested_checks": ["fee_x2"],
    }]}, {"usd": 0.01, "tokens_in": 5000, "tokens_out": 800}


def run_sandbox(variant: dict) -> dict:
    _FAKE["n"] += 1
    ok = _FAKE["n"] % 3 != 1          # 每 3 次第一次故意失敗，演示 debug 迴圈
    return {"ok": ok, "error": None if ok else "KeyError: 'close_'", "invariance_ok": ok}


def run_backtest_is(variant: dict) -> dict:
    return {"pack_ref": f"art://diag/{variant['code_hash']}", "n_trials_added": 12,
            "keys": ["regime.high_vol.sharpe", "cv.sharpe_median", "dd.max", "dsr", "pbo"],
            "values": {"regime.high_vol.sharpe": -0.4, "cv.sharpe_median": 1.3, "dd.max": 0.18,
                       "dsr": 0.97, "pbo": 0.2, "cv.pos_fold_ratio": 0.8, "trades": 400,
                       "plateau_ratio": 0.8, "sharpe_fee_x2": 0.7}}


def run_checks(check_ids: list[str], variant: dict) -> dict[str, bool]:
    return {c: True for c in check_ids}


# ───────────────────────── 4. 共用：context projection / guarded Command ─────────────────────────
LLM_VISIBLE = {  # 每個角色可見的 state 欄位白名單（lockbox_ref、gate token、oos 數字永遠不在內）
    "supervisor": ["phase", "reflection", "archive", "hyp_counters", "counters", "spend", "oos", "lessons"],
    "researcher": ["lessons", "archive", "directive", "review"],
    "coder": ["hypothesis", "directive", "lessons"],
    "coder_debug": ["variant", "debug"],
    "critic": [],  # 批判者只吃 Send payload，不讀 state
}


def project(role: str, state: State) -> dict:
    return {k: state.get(k) for k in LLM_VISIBLE[role]}


def guarded(goto: str | list, dests: tuple[str, ...], **kw) -> Command:
    """LangGraph 的 destinations 只用於畫圖、不在 runtime 強制；因此自己檢查。"""
    targets = [goto] if isinstance(goto, str) else goto
    for t in targets:
        name = t.node if isinstance(t, Send) else t
        assert name in dests, f"illegal goto {name} (allowed {dests})"
    return Command(goto=goto, **kw)


def over_budget(state: State) -> str | None:
    s, c = state.get("spend", {}), state.get("counters", {})
    if s.get("usd", 0) >= LIMITS["max_usd"]: return "usd"
    if s.get("tokens_in", 0) + s.get("tokens_out", 0) >= LIMITS["max_tokens"]: return "tokens"
    if s.get("wall_s", 0) >= LIMITS["max_wall_s"]: return "wall_clock"
    if c.get("backtests", 0) >= LIMITS["max_backtests"]: return "backtests"
    return None


# ───────────────────────── 5. Guardrail：Supervisor 只能在這些選項中挑 ─────────────────────────
def allowed_actions(state: State) -> tuple[list[str], str]:
    """純函式、可單元測試。回傳 (允許動作, 預設動作)。"""
    if (why := over_budget(state)) is not None:
        return ["STOP"], "STOP"
    c = state.get("counters", {})
    hyp = state.get("hypothesis") or {}
    hc = state.get("hyp_counters", {}).get(hyp.get("hyp_id", ""), {})
    allowed: list[str] = []
    variant_dead = state.get("oos", {}).get("variant_id") == state.get("variant", {}).get("variant_id") \
        and state.get("oos", {}).get("passed") is False                     # OOS 失敗的變體不可再往 OOS 方向微調
    if hyp and not variant_dead and hc.get("refine", 0) < LIMITS["max_refine_per_hyp"] \
            and hc.get("no_improve", 0) < LIMITS["plateau_k"]:
        allowed.append("REFINE")
    if c.get("ideation", 0) < LIMITS["max_ideation"]:
        allowed.append("NEW_HYPOTHESIS")
    if any(a["hyp_id"] != hyp.get("hyp_id") and not a.get("oos_failed") for a in state.get("archive", [])):
        allowed.append("RESURRECT")
    if state.get("reflection", {}).get("open_conflicts") and not state.get("review", {}).get("consulted"):
        allowed.append("ESCALATE_HUMAN")
    if not allowed:
        return ["STOP"], "STOP"
    allowed.append("STOP")   # 只有在還有其他選項時 STOP 才需要 LLM 判斷
    default = "REFINE" if "REFINE" in allowed and hc.get("no_improve", 0) == 0 else \
        ("NEW_HYPOTHESIS" if "NEW_HYPOTHESIS" in allowed else allowed[0])
    return allowed, default


# ───────────────────────── 6. 節點 ─────────────────────────
def init_run(state: State) -> dict:
    meta = {"run_id": uuid.uuid4().hex, "engine_ver": "bt-1.4.2", "data_hash": "sha256:...",
            "gate_cfg_hash": hashlib.sha256(json.dumps(LIMITS, sort_keys=True).encode()).hexdigest()[:12],
            "prompt_ver": {"supervisor": "sup-v3", "critic": "crit-v2"}}
    return {"run_meta": meta, "phase": "ideate", "counters": {}, "spend": {}, "hyp_counters": {}}


def supervisor(state: State) -> Command:
    allowed, default = allowed_actions(state)
    used_llm, violation, out = False, False, {}
    if len(allowed) == 1:
        action, rationale, cost = allowed[0], "forced_by_guardrail", {}
    else:
        # 真實版：schema 的 action enum 動態設為 allowed（constrained decoding）
        out, cost = llm_structured("supervisor", "SupervisorDecision", {"allowed": allowed, **project("supervisor", state)})
        used_llm = True
        action, rationale = out.get("action"), out.get("rationale", "")[:500]
        if action not in allowed:                     # 事後驗證：LLM 越權 → 用確定性預設
            action, violation = default, True
    # Supervisor 只能「引用」本輪已通過證據驗證的 directive id，不能自創改動方向
    valid_ids = {f["fid"]: f["critic"] for f in state.get("reflection", {}).get("directives", [])}
    picked = [i for i in (out.get("directive_ids", []) if used_llm else []) if i in valid_ids][:2]
    if action == "REFINE" and not picked and valid_ids:
        picked = [next(iter(valid_ids))]                     # 預設採用最高嚴重度的 directive
    upd: dict[str, Any] = {
        "supervisor_log": [{"allowed": allowed, "chosen": action, "used_llm": used_llm,
                            "violation": violation, "rationale": rationale, "directive_ids": picked}],
        "spend": cost,
        "directive": {"action": action, "text": rationale[:500], "adopted": [[i, valid_ids[i]] for i in picked]},
        "critic_stats": {f"{valid_ids[i]}.adopted": 1 for i in picked},
    }
    if action == "NEW_HYPOTHESIS":
        upd["counters"] = {"ideation": 1}
    elif action in ("REFINE", "RESURRECT"):
        if action == "RESURRECT":
            best = next(a for a in state["archive"] if a["hyp_id"] != state["hypothesis"]["hyp_id"])
            upd["hypothesis"] = best["hypothesis"]
        hid = (upd.get("hypothesis") or state["hypothesis"])["hyp_id"]
        hc = {k: dict(v) for k, v in state.get("hyp_counters", {}).items()}
        hc.setdefault(hid, {"refine": 0, "no_improve": 0})["refine"] += 1
        upd["hyp_counters"] = hc
    elif action in ("STOP", "ESCALATE_HUMAN"):
        upd["phase"] = "stopped" if action == "STOP" else "consult"
    return guarded(ACTION_TO_NODE[action], SUPERVISOR_DESTS, update=upd)


def researcher(state: State) -> dict:
    out, cost = llm_structured("researcher", "Hypothesis", project("researcher", state))
    # 確定性 novelty 檢查（family + 參數化指紋與 ledger 比對）略
    hc = dict(state.get("hyp_counters", {}))
    hc[out["hyp_id"]] = {"refine": 0, "no_improve": 0}
    return {"hypothesis": out, "hyp_counters": hc, "spend": cost, "phase": "code"}


def coder(state: State) -> dict:
    out, cost = llm_structured("coder", "SignalModule", project("coder", state))
    code_hash = hashlib.sha256(out["code"].encode()).hexdigest()[:12]
    variant = {"variant_id": uuid.uuid4().hex[:8], "code_ref": f"art://code/{code_hash}",
               "code_hash": code_hash, "param_space": out["param_space"],
               "hyp_id": state["hypothesis"]["hyp_id"]}
    return {"variant": variant, "debug": {"attempts": 0, "error_hashes": {}, "escalated": False},
            "spend": cost}


def static_check(state: State) -> dict:
    # AST 掃描（禁 shift(-k)、center=True、未來索引、import 白名單外）、schema、複雜度上限
    return {"exec_result": {"static_ok": True}}


def sandbox_exec(state: State) -> dict:
    r = run_sandbox(state["variant"])        # 拋棄式容器 --network none；含截斷不變性測試
    return {"exec_result": {**state.get("exec_result", {}), **r}, "spend": {"wall_s": 20}}


def norm_error_hash(err: str) -> str:
    import re
    core = re.sub(r"0x[0-9a-f]+|\d+|'[^']*'", "#", err.splitlines()[-1] if err else "")
    return hashlib.sha1(core.encode()).hexdigest()[:10]


def route_after_exec(state: State) -> str:
    r, d, c = state["exec_result"], state["debug"], state.get("counters", {})
    if r.get("static_ok") and r.get("ok") and r.get("invariance_ok"):
        return "backtest_is"
    h = norm_error_hash(r.get("error") or "invariance_violation")
    if d["attempts"] >= LIMITS["max_debug_per_variant"] or c.get("debug_total", 0) >= LIMITS["max_debug_total"]:
        return "abandon_variant"
    if d["error_hashes"].get(h, 0) >= LIMITS["max_same_error_hash"] and d["escalated"]:
        return "abandon_variant"
    return "coder_debug"


def coder_debug(state: State) -> dict:
    d = dict(state["debug"])
    h = norm_error_hash(state["exec_result"].get("error") or "invariance_violation")
    eh = dict(d["error_hashes"]); eh[h] = eh.get(h, 0) + 1
    escalate = eh[h] >= LIMITS["max_same_error_hash"]            # 同錯重複 → 換強模型 + 最小重現
    out, cost = llm_structured("coder_debug", "SignalModule", {**project("coder_debug", state), "escalate": escalate})
    code_hash = hashlib.sha256((out["code"] + str(d["attempts"])).encode()).hexdigest()[:12]
    v = {**state["variant"], "code_hash": code_hash, "code_ref": f"art://code/{code_hash}"}
    return {"variant": v, "debug": {"attempts": d["attempts"] + 1, "error_hashes": eh, "escalated": escalate or d["escalated"]},
            "counters": {"debug_total": 1}, "spend": cost}      # 修語法不記入試驗帳本


def abandon_variant(state: State) -> dict:
    return {"attempts": [{"variant_id": state["variant"]["variant_id"], "status": "exec_failed"}], "phase": "decide"}


def backtest_is(state: State) -> dict:
    diag = run_backtest_is(state["variant"])     # walk-forward purged CV；每個參數組合寫入全域試驗帳本
    return {"diagnostics": diag, "counters": {"backtests": 1, "trials": diag["n_trials_added"]},
            "round_id": uuid.uuid4().hex[:8], "spend": {"wall_s": 180}}


CRITICS = ("quant_critic", "risk_critic", "stat_auditor")


def dispatch_critics(state: State) -> Command:
    """確定性挑選要跑的批判者（預算、歷史命中率）→ Send 扇出。Overwrite 重置本輪 critiques。"""
    stats = state.get("critic_stats", {})
    def useful(c: str) -> bool:
        n_f = stats.get(f"{c}.valid", 0) + stats.get(f"{c}.dropped", 0)
        if n_f >= 10 and stats.get(f"{c}.dropped", 0) / n_f > 0.6:      # 證據常造假 → 停用
            return False
        n_a = stats.get(f"{c}.adopted", 0)
        return n_a < 8 or stats.get(f"{c}.hit", 0) / n_a >= 0.25        # 被採用後很少兌現 → 停用
    chosen = [c for c in CRITICS if c == "stat_auditor" or useful(c)]   # Stat Auditor 永遠在場
    payload = {"round_id": state["round_id"], "variant": state["variant"],
               "diag_keys": state["diagnostics"]["keys"], "headline": state["diagnostics"]["values"],
               "check_registry": ["fee_x2", "delay_1bar", "drop_top5_trades", "subperiod_split", "param_jitter"]}
    return guarded([Send(c, payload) for c in chosen], CRITICS, update={"critiques": Overwrite([])})


def make_critic(role: str):
    def critic(payload: dict) -> dict:          # 只讀 payload：看不到 researcher 的敘事、彼此意見、OOS
        out, cost = llm_structured(role, "Critique", payload)
        return {"critiques": [{"critic": role, "round_id": payload["round_id"], **f} for f in out["findings"]],
                "spend": cost}
    critic.__name__ = role
    return critic


OPS = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}


def aggregate(state: State) -> dict:
    """確定性聚合：證據斷言驗證 → 去重 → 衝突偵測 → 預測計分 → lessons。"""
    vals, rid = state["diagnostics"]["values"], state["round_id"]
    kept, dropped, stats = [], [], {}
    for f in state.get("critiques", []):
        if f["round_id"] != rid:
            continue
        ok = f["evidence"] and all(e["ref"] in vals and OPS[e["op"]](vals[e["ref"]], e["value"]) for e in f["evidence"])
        (kept if ok else dropped).append(f)
        stats[f"{f['critic']}.{'valid' if ok else 'dropped'}"] = stats.get(f"{f['critic']}.{'valid' if ok else 'dropped'}", 0) + 1
    by_target: dict[str, list[dict]] = {}
    for f in kept:
        by_target.setdefault(f["proposal"]["target"], []).append(f)
    conflicts = [t for t, fs in by_target.items() if len({x["proposal"]["direction"] for x in fs}) > 1]
    directives = [{**f, "fid": f"{rid}:{f['critic']}:{i}"}
                  for i, f in enumerate(sorted(kept, key=lambda f: -f["severity"])[:3])]
    checks = sorted({c for f in kept for c in f.get("requested_checks", [])})[:4]
    pred = (state.get("hypothesis") or {}).get("prediction", {})
    hid0 = state["hypothesis"]["hyp_id"]
    prev = next((a["metric"] for a in reversed(state.get("attempts", [])) if a.get("hyp_id") == hid0), None)
    now = vals.get("cv.sharpe_median")
    outcome = "n/a" if prev is None else ("confirmed" if (now - prev) >= pred.get("min_delta", 0) else "refuted")
    if outcome == "confirmed":                       # 事後歸因：上一輪被採用的 directive 是否兌現預測
        for _, critic in (state.get("directive") or {}).get("adopted", []):
            stats[f"{critic}.hit"] = stats.get(f"{critic}.hit", 0) + 1
    hid = state["hypothesis"]["hyp_id"]
    lesson = {"key": f"{hid}:{state['variant']['code_hash']}", "weight": 2 if outcome == "refuted" else 1,
              "ts": len(state.get("attempts", [])),
              "text": f"{hid} 變更預測 {pred.get('direction')}≥{pred.get('min_delta')}，實際 Δ={None if prev is None else round(now - prev, 3)} → {outcome}"}
    return {"reflection": {"round_id": rid, "directives": directives, "open_conflicts": conflicts,
                           "requested_checks": checks, "prediction_outcome": outcome, "dropped": len(dropped)},
            "lessons": [lesson], "critic_stats": stats}


def acceptance_gate(state: State) -> Command:
    """唯一能讓變體前進到 OOS 的地方。純程式，閾值來自 run_meta 鎖定的 gate config。"""
    v, d = state["variant"], state["diagnostics"]["values"]
    checks = run_checks(state["reflection"]["requested_checks"], v)   # LLM 只能「加」檢查，不能減
    rules = {
        "trades": d["trades"] >= 150, "cv_sharpe": d["cv.sharpe_median"] >= 1.0,
        "fold_consistency": d["cv.pos_fold_ratio"] >= 0.7, "dsr": d["dsr"] >= 0.95, "pbo": d["pbo"] <= 0.3,
        "max_dd": d["dd.max"] <= 0.25, "plateau": d["plateau_ratio"] >= 0.7, "cost_x2": d["sharpe_fee_x2"] >= 0.5,
        "requested_checks": all(checks.values()),
    }
    passed = all(rules.values())
    token = hmac.new(GATE_SECRET, f"{v['code_hash']}|{state['run_meta']['data_hash']}|{passed}".encode(), "sha256").hexdigest()
    hid = v["hyp_id"]
    prev = next((a for a in reversed(state.get("attempts", [])) if a.get("hyp_id") == hid), None)
    improved = prev is None or d["cv.sharpe_median"] > prev["metric"] + LIMITS["plateau_eps"]   # 首個變體 = 基準
    hc = {k: dict(x) for k, x in state.get("hyp_counters", {}).items()}
    hc.setdefault(hid, {"refine": 0, "no_improve": 0})["no_improve"] = 0 if improved else hc[hid]["no_improve"] + 1
    upd = {"gate": {"variant_id": v["variant_id"], "passed": passed, "rules": rules, "token": token},
           "hyp_counters": hc,
           "attempts": [{"variant_id": v["variant_id"], "hyp_id": hid, "metric": d["cv.sharpe_median"],
                         "gate": passed, "failed_rules": [k for k, ok in rules.items() if not ok]}],
           "archive": [{"variant_id": v["variant_id"], "hyp_id": hid, "hypothesis": state["hypothesis"],
                        "score": d["dsr"], "oos_failed": False}]}
    if passed and state.get("counters", {}).get("oos_peeks", 0) < LIMITS["max_oos_peeks"]:
        return guarded("oos_eval", ("oos_eval", "supervisor"), update=upd)
    return guarded("supervisor", ("oos_eval", "supervisor"), update=upd)


def oos_eval(state: State) -> Command:
    g, v = state["gate"], state["variant"]
    expect = hmac.new(GATE_SECRET, f"{v['code_hash']}|{state['run_meta']['data_hash']}|True".encode(), "sha256").hexdigest()
    assert g["passed"] and hmac.compare_digest(g["token"], expect), "gate bypass detected"
    assert state.get("oos", {}).get("variant_id") != v["variant_id"], "OOS already used for this variant"
    passed = True                                   # 真實版：引擎在 OOS 區間跑一次，數字寫 artifact
    upd = {"oos": {"variant_id": v["variant_id"], "passed": passed, "bucket": "ok" if passed else "severe_decay"},
           "counters": {"oos_peeks": 1}}
    if not passed:
        upd["archive"] = [{"variant_id": v["variant_id"], "hyp_id": v["hyp_id"], "hypothesis": state["hypothesis"],
                           "score": -1, "oos_failed": True}]
        return guarded("supervisor", ("lockbox_eval", "supervisor"), update=upd)
    return guarded("lockbox_eval", ("lockbox_eval", "supervisor"), update=upd)


def lockbox_eval(state: State) -> dict:
    assert not state.get("lockbox_ref"), "lockbox already consumed in this thread"
    return {"lockbox_ref": f"art://lockbox/{state['variant']['variant_id']}", "phase": "final_review"}


def prepare_review(state: State) -> dict:
    """副作用（寫報告 artifact）放在這裡，human_review 本身零副作用。"""
    kind = {"final_review": "final", "consult": "consult"}.get(state.get("phase"), "stopped")
    return {"review": {**state.get("review", {}), "kind": kind, "report_ref": f"art://report/{uuid.uuid4().hex[:6]}"}}


def human_review(state: State) -> Command:
    rv = state["review"]
    decision = interrupt({"kind": rv["kind"], "report_ref": rv["report_ref"],
                          "options": ["approve", "reject"] if rv["kind"] == "final" else ["continue", "stop"]})
    choice = decision.get("decision")
    if rv["kind"] == "consult" and choice == "continue":
        return guarded("supervisor", ("supervisor", "finalize"),
                       update={"review": {**rv, "consulted": True, "note": decision.get("note", "")[:1000]},
                               "phase": "decide"})
    return guarded("finalize", ("supervisor", "finalize"), update={"review": {**rv, "decision": choice}})


def finalize(state: State) -> dict:
    return {"phase": "done"}


# ───────────────────────── 7. 建圖 ─────────────────────────
def build_graph(checkpointer=None):
    llm_retry = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0)
    g = StateGraph(State)
    g.add_node("init_run", init_run)
    g.add_node("supervisor", supervisor, destinations=SUPERVISOR_DESTS, retry_policy=llm_retry)
    g.add_node("researcher", researcher, retry_policy=llm_retry)
    g.add_node("coder", coder, retry_policy=llm_retry)
    g.add_node("static_check", static_check)
    g.add_node("sandbox_exec", sandbox_exec)
    g.add_node("coder_debug", coder_debug, retry_policy=llm_retry)
    g.add_node("abandon_variant", abandon_variant)
    g.add_node("backtest_is", backtest_is)
    g.add_node("dispatch_critics", dispatch_critics, destinations=CRITICS)
    for c in CRITICS:
        g.add_node(c, make_critic(c), retry_policy=llm_retry)
    g.add_node("aggregate", aggregate)
    g.add_node("acceptance_gate", acceptance_gate, destinations=("oos_eval", "supervisor"))
    g.add_node("oos_eval", oos_eval, destinations=("lockbox_eval", "supervisor"))
    g.add_node("lockbox_eval", lockbox_eval)
    g.add_node("prepare_review", prepare_review)
    g.add_node("human_review", human_review, destinations=("supervisor", "finalize"))
    g.add_node("finalize", finalize)

    g.add_edge(START, "init_run")
    g.add_edge("init_run", "researcher")          # 第一步固定發想，不問 supervisor
    g.add_edge("researcher", "coder")
    g.add_edge("coder", "static_check")
    g.add_edge("static_check", "sandbox_exec")
    g.add_conditional_edges("sandbox_exec", route_after_exec,
                            {"backtest_is": "backtest_is", "coder_debug": "coder_debug",
                             "abandon_variant": "abandon_variant"})
    g.add_edge("coder_debug", "static_check")
    g.add_edge("abandon_variant", "supervisor")
    g.add_edge("backtest_is", "dispatch_critics")
    for c in CRITICS:
        g.add_edge(c, "aggregate")                 # 同一 superstep 的 Send 全部完成後 aggregate 只跑一次
    g.add_edge("aggregate", "acceptance_gate")     # 靜態邊：gate 不可被繞過
    g.add_edge("lockbox_eval", "prepare_review")
    g.add_edge("prepare_review", "human_review")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def assert_topology(app) -> None:
    """CI 測試：Supervisor 不得直達 gate/OOS/lockbox；OOS 只能由 gate 進入。"""
    edges = {(e.source, e.target) for e in app.get_graph().edges}
    for bad in ["acceptance_gate", "oos_eval", "lockbox_eval", "finalize", "backtest_is"]:
        assert ("supervisor", bad) not in edges, bad
    assert {s for s, t in edges if t == "oos_eval"} == {"acceptance_gate"}
    assert {s for s, t in edges if t == "lockbox_eval"} == {"oos_eval"}
    assert {s for s, t in edges if t == "acceptance_gate"} == {"aggregate"}


if __name__ == "__main__":
    app = build_graph()
    assert_topology(app)
    cfg = {"configurable": {"thread_id": "demo-1"}, "recursion_limit": 1000}
    for chunk in app.stream({}, cfg, stream_mode="updates"):
        print(list(chunk.keys()))
    snap = app.get_state(cfg)
    print("next:", snap.next, "| interrupt:", snap.tasks[0].interrupts[0].value if snap.tasks else None)
    for chunk in app.stream(Command(resume={"decision": "approve"}), cfg, stream_mode="updates"):
        print(list(chunk.keys()))
    final = app.get_state(cfg).values
    print("phase:", final["phase"], "| spend:", final["spend"], "| counters:", final["counters"])
    print("supervisor_log:", final.get("supervisor_log"))
    print("reflection:", {k: final["reflection"][k] for k in ("open_conflicts", "requested_checks", "dropped")})
