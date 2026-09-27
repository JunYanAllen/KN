"""Graph nodes.

LLM nodes: strategist, coder, debugger, reflect.
Everything else is deterministic code. Every node returns a partial state
update; `node()` adds active-time accounting and turns HaltError into a
graceful `halt_reason` instead of a crashed run.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langgraph.types import interrupt
from pydantic import ValidationError

from . import context, engine, prompts, stats
from .config import Config
from .data import Snapshot
from .ledger import Ledger, QuotaExceeded
from .llm import (CoderOut, DebugOut, HaltError, HumanDecision, LLMClient, Reflection,
                  StrategistOut)
from .safety import ast_rules
from .safety.sandbox import SubprocessSandbox
from .state import ResearchState
from .store import ArtifactStore, EventLog

# Untargeted changes for the control arm (measures whether reflection beats chance).
CONTROL_CHANGES = [
    {"kind": "exit_logic", "description": "smooth: average position over 12 bars"},
    {"kind": "filter", "description": "long_only: drop the short side"},
    {"kind": "filter", "description": "vol_filter: stand aside in abnormally high volatility"},
]


@dataclass
class Deps:
    cfg: Config
    snapshot: Snapshot
    ledger: Ledger
    store: ArtifactStore
    sandbox: SubprocessSandbox
    llm: LLMClient
    events: EventLog
    run_dir: Path


# ============================================================ helpers
def node(deps: Deps, name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(state: ResearchState) -> dict:
            t0 = time.monotonic()
            try:
                out = fn(state) or {}
            except HaltError as e:
                out = {"halt_reason": e.reason, "halt_detail": str(e), "spent": getattr(e, "spent", {})}
                deps.events.emit("halt", node=name, reason=e.reason, detail=str(e))
            spent = dict(out.get("spent") or {})
            spent["active_s"] = spent.get("active_s", 0) + (time.monotonic() - t0)
            out["spent"] = spent
            deps.events.emit("node", node=name, run_id=state.get("run_id"),
                             variant=(out.get("variant") or state.get("variant") or {}).get("variant_id"))
            return out
        return wrapped
    return deco


def budget_reason(deps: Deps, state: ResearchState) -> str | None:
    L, sp = deps.cfg.limits, (state.get("spent") or {})
    if sp.get("usd", 0) >= L.usd:
        return "budget_usd"
    if sp.get("active_s", 0) >= L.active_seconds:
        return "budget_time"
    if deps.ledger.n_trials(deps.snapshot.snapshot_id) >= L.max_trials_per_snapshot:
        return "trial_cap"
    return None


def remaining_usd(deps: Deps, state: ResearchState) -> float:
    return deps.cfg.limits.usd - (state.get("spent") or {}).get("usd", 0.0)


def error_hash(error_class: str, message: str) -> str:
    norm = re.sub(r"\d+", "#", message.split("\n")[0])
    return hashlib.sha256(f"{error_class}|{norm}".encode()).hexdigest()[:12]


def f(x) -> float:
    return round(float(x), 4)


# ============================================================ builders
def build_nodes(deps: Deps) -> dict:
    cfg, L = deps.cfg, deps.cfg.limits
    bpy = cfg.costs.bars_per_year
    research = deps.snapshot.upto("research")

    # -------------------------------------------------------- init
    @node(deps, "init_run")
    def init_run(state):
        n_obs = len(research) - cfg.wf.initial_train_bars
        prior = deps.ledger.trial_srs(deps.snapshot.snapshot_id)
        var = stats.trial_variance(prior, n_obs)          # null floor until real trials exist
        curve = {n: round(stats.required_annual_sr(n, n_obs, bpy, cfg.is_gate.min_dsr, var), 3)
                 for n in (1, 5, 10, 20, 40)}
        deps.events.emit("run_start", run_id=state["run_id"], snapshot=deps.snapshot.snapshot_id,
                         config=cfg.fingerprint(), threshold_curve=curve,
                         prior_trials=deps.ledger.n_trials(deps.snapshot.snapshot_id))
        return {"snapshot_id": deps.snapshot.snapshot_id, "config_fp": cfg.fingerprint(),
                "threshold_curve": curve, "idea_seq": 0, "lessons": [], "idea": None,
                "variant": None, "reflection": None, "halt_reason": None}

    # -------------------------------------------------------- loop C: strategist
    @node(deps, "strategist")
    def strategist(state):
        payload = context.strategist_payload(cfg, state.get("attempts", []), state.get("lessons", []))
        out, spent = deps.llm.call("strategist", "strong", StrategistOut, prompts.STRATEGIST,
                                   payload, remaining_usd(deps, state))
        tried = {a["family"] for a in state.get("attempts", []) if a.get("kind") == "idea"}
        tried_words = [set(a["thesis"].lower().split()) for a in state.get("attempts", []) if a.get("kind") == "idea"]

        def novelty(h):
            if h.family not in tried:
                return 2.0
            w = set(h.thesis.lower().split())
            return 1.0 - max((len(w & t) / max(len(w | t), 1) for t in tried_words), default=0.0)

        hyp = max(out.candidates, key=novelty).model_dump()
        seq = state["idea_seq"] + 1
        idea_id = f"{state['run_id']}.i{seq}"
        prereg = {"hypothesis": hyp, "registered_before_any_backtest": True}
        prereg_hash = hashlib.sha256(json.dumps(prereg, sort_keys=True).encode()).hexdigest()[:16]
        deps.ledger.preregister(idea_id, state["run_id"], prereg_hash, prereg)
        idea = {"idea_id": idea_id, "seq": seq, "family": hyp["family"], "hypothesis": hyp,
                "prereg_hash": prereg_hash, "refine_iter": 0, "stalls": 0, "best": None,
                "changes": [], "passed": False}
        return {"idea_seq": seq, "idea": idea, "variant": None, "reflection": None, "spent": spent}

    # -------------------------------------------------------- coder
    @node(deps, "coder")
    def coder(state):
        idea, refl = state["idea"], state.get("reflection")
        parent = idea["best"]
        parent_code = deps.store.get_text(parent["code_ref"]) if (refl and parent) else None
        payload = context.coder_payload(cfg, idea, idea["seq"], parent_code, refl and refl["change"])
        out, spent = deps.llm.call("coder", "strong", CoderOut, prompts.CODER, payload, remaining_usd(deps, state))
        arm = "initial" if not refl else ("control" if refl.get("control") else "reflect")
        variant = {"variant_id": f"{idea['idea_id']}.v{idea['refine_iter']}", "arm": arm,
                   "code_ref": deps.store.put_text(out.code), "status": "pending",
                   "debug_attempts": 0, "error_counts": {}, "last_error": None,
                   "locked_space": None, "param_space": None, "signals_ref": None,
                   "parent_id": parent["variant_id"] if (refl and parent) else None,
                   "change": refl and refl["change"], "prediction": refl and refl.get("prediction")}
        return {"variant": variant, "spent": spent}

    # -------------------------------------------------------- loop A: checks + debugger
    def _fail(v: dict, cls: str, msg: str, status: str) -> dict:
        h = error_hash(cls, msg)
        counts = dict(v["error_counts"])
        counts[h] = counts.get(h, 0) + 1
        return {**v, "status": status, "last_error": {"class": cls, "message": msg[:1500], "hash": h},
                "error_counts": counts}

    @node(deps, "static_check")
    def static_check(state):
        v = state["variant"]
        rep = ast_rules.check(deps.store.get_text(v["code_ref"]), L.max_params, L.max_param_grid)
        if not rep.ok:
            return {"variant": _fail(v, rep.error_class, rep.message, "check_failed")}
        if v["locked_space"] is not None and rep.param_space != v["locked_space"]:
            return {"variant": _fail(v, "E_INTERFACE", "the fix changed PARAM_SPACE; restore it exactly "
                                     f"as {v['locked_space']}", "check_failed")}
        return {"variant": {**v, "status": "static_ok", "param_space": rep.param_space}}

    @node(deps, "sandbox_validate")
    def sandbox_validate(state):
        v = state["variant"]
        dup = deps.ledger.has_code(deps.snapshot.snapshot_id, v["code_ref"])
        if dup:                                                 # identical code: never a new trial
            return {"variant": {**v, "status": "degenerate",
                                "last_error": {"class": "E_DUPLICATE", "message": f"identical to {dup}"}}}
        grid = ast_rules.expand_grid(v["param_space"])
        code = deps.store.get_text(v["code_ref"])
        res = deps.sandbox.run(code, research, grid)
        if res.error_class == "E_INFRA":                        # one retry, then halt (no lesson poisoning)
            res = deps.sandbox.run(code, research, grid)
            if res.error_class == "E_INFRA":
                return {"halt_reason": "infra", "halt_detail": res.message}
        v = {**v, "locked_space": v["param_space"]}
        if res.ok:
            return {"variant": {**v, "status": "valid", "signals_ref": deps.store.put_array(res.signals)}}
        if res.error_class == "E_DEGENERATE":                   # semantic failure: not a debugging job
            return {"variant": {**v, "status": "degenerate", "last_error": {"class": res.error_class,
                                                                            "message": res.message}}}
        return {"variant": _fail(v, res.error_class, res.message, "check_failed")}

    @node(deps, "debugger")
    def debugger(state):
        v, idea = state["variant"], state["idea"]
        attempt = v["debug_attempts"] + 1
        code = deps.store.get_text(v["code_ref"])
        err = v["last_error"]
        if attempt <= 2:
            # escalation ladder: cheap patch -> strong explain-then-patch -> regenerate
            tier = "cheap" if attempt == 1 else "strong"
            payload = context.debugger_payload(code, err["class"], err["message"], attempt)
            out, spent = deps.llm.call("debugger", tier, DebugOut, prompts.DEBUGGER, payload,
                                       remaining_usd(deps, state))
            new_code = out.code
        else:
            refl = state.get("reflection")
            parent = idea["best"]
            payload = context.coder_payload(cfg, idea, idea["seq"],
                                            deps.store.get_text(parent["code_ref"]) if (refl and parent) else None,
                                            refl and refl["change"])
            payload["previous_error"] = {"class": err["class"], "message": err["message"][:500]}
            out, spent = deps.llm.call("debugger", "strong", CoderOut, prompts.CODER, payload,
                                       remaining_usd(deps, state))
            new_code = out.code
        return {"variant": {**v, "debug_attempts": attempt, "status": "pending",
                            "code_ref": deps.store.put_text(new_code)}, "spent": spent}

    @node(deps, "abandon_variant")
    def abandon_variant(state):
        v, idea = state["variant"], state["idea"]
        reason = (v["last_error"]["class"] if v["status"] == "degenerate"
                  else f"debug_exhausted:{v['last_error']['class']}")
        deps.events.emit("variant_abandoned", variant=v["variant_id"], reason=reason)
        rec = {"kind": "variant", "variant_id": v["variant_id"], "idea_id": idea["idea_id"], "arm": v["arm"],
               "status": "abandoned", "reason": reason, "debug_attempts": v["debug_attempts"]}
        return {"attempts": [rec], "idea": {**idea, "stalls": idea["stalls"] + 1},
                "variant": {**v, "status": "abandoned"}}

    # -------------------------------------------------------- IS backtest (walk-forward)
    def _grid_pnl(signals: np.ndarray, frame, fee_mult=1.0, delay=0):
        runs = [engine.run(frame, signals[k], cfg.costs, fee_mult=fee_mult, delay=delay)
                for k in range(signals.shape[0])]
        return np.stack([r.pnl for r in runs], 1), np.stack([r.pos for r in runs], 1)

    @node(deps, "backtest_is")
    def backtest_is(state):
        v, idea = state["variant"], state["idea"]
        signals = deps.store.get_array(v["signals_ref"])
        grid = ast_rules.expand_grid(v["param_space"])
        rate = engine.cost_rate(cfg.costs)
        pnl, pos = _grid_pnl(signals, research)
        wf = engine.walk_forward(pnl, pos, cfg.wf, bpy, rate)
        m = engine.metrics(wf.oof_pnl, wf.oof_pos, bpy)
        pnl_fee2, _ = _grid_pnl(signals, research, fee_mult=2.0)
        pnl_delay, pos_delay = _grid_pnl(signals, research, delay=1)
        oof, half = wf.oof_pnl, len(wf.oof_pnl) // 2
        vol = np.abs(research["close"].pct_change().to_numpy())[wf.oof_index]
        hi = vol > np.nanmedian(vol)
        diag = {
            "oof_sharpe": m["sharpe"], "max_dd": m["max_dd"], "n_trades": m["n_trades"],
            "exposure": m["exposure"], "turnover_per_year": m["turnover_per_year"],
            "long_sharpe": m["long_sharpe"], "short_sharpe": m["short_sharpe"],
            "fold_sharpes": wf.fold_sharpes,
            "pos_fold_frac": f(np.mean([s > 0 for s in wf.fold_sharpes])),
            "fee_x2_sharpe": f(engine.sharpe(wf.apply(pnl_fee2, pos, 2 * rate), bpy)),
            "delay_sharpe": f(engine.sharpe(wf.apply(pnl_delay, pos_delay, rate), bpy)),
            "plateau_ratio": engine.plateau_ratio(pnl, grid, bpy),
            "first_half_sharpe": f(engine.sharpe(oof[:half], bpy)),
            "second_half_sharpe": f(engine.sharpe(oof[half:], bpy)),
            "high_vol_sharpe": f(engine.sharpe(oof[hi], bpy)),
            "low_vol_sharpe": f(engine.sharpe(oof[~hi], bpy)),
            # not LLM-visible:
            "skew": m["skew"], "kurt": m["kurt"], "n_obs": m["n_obs"],
            "chosen_params": grid[max(set(wf.chosen), key=wf.chosen.count)],
            "last_fold_params": grid[wf.chosen[-1]],
        }
        oof_ref = deps.store.put_array(wf.oof_pnl)
        sr_bar = stats.per_bar(m["sharpe"], bpy)
        deps.ledger.record_trial(v["variant_id"], deps.snapshot.snapshot_id, state["run_id"], idea["idea_id"],
                                 v["arm"], sr_bar, m["n_obs"], m["skew"], m["kurt"], v["code_ref"], diag)
        srs = deps.ledger.trial_srs(deps.snapshot.snapshot_id)
        diag["dsr"] = f(stats.dsr(sr_bar, m["n_obs"], m["skew"], m["kurt"], len(srs),
                                  stats.trial_variance(srs, m["n_obs"])))
        diag["n_trials_at_eval"] = len(srs)
        return {"variant": {**v, "status": "backtested", "diag": diag, "oof_ref": oof_ref},
                "spent": {"backtests": 1}}

    # -------------------------------------------------------- deterministic IS gate + prediction scoring
    def passes_is_gate(d: dict) -> tuple[bool, list[str]]:
        g = cfg.is_gate
        fails = []
        if d["oof_sharpe"] >= g.suspicious_sharpe:
            fails.append("suspicious_sharpe")
        if d["dsr"] < g.min_dsr:
            fails.append("dsr")
        if d["n_trades"] < g.min_trades:
            fails.append("n_trades")
        if d["pos_fold_frac"] < g.min_pos_fold_frac:
            fails.append("folds")
        if d["fee_x2_sharpe"] <= g.fee_x2_min_sharpe:
            fails.append("fee_x2")
        if d["oof_sharpe"] <= 0 or d["delay_sharpe"] / d["oof_sharpe"] < g.delay_min_retention:
            fails.append("delay")
        if d["plateau_ratio"] < g.plateau_min_ratio:
            fails.append("plateau")
        return not fails, fails

    @node(deps, "gate_is")
    def gate_is(state):
        v, idea = state["variant"], state["idea"]
        d = v["diag"]
        lessons = [dict(l) for l in state.get("lessons", [])]
        improved, delta, se, hit = True, None, None, None
        parent = idea["best"]
        if v["parent_id"] and parent:
            child = deps.store.get_array(v["oof_ref"])
            par = deps.store.get_array(parent["oof_ref"])
            delta, se = stats.paired_sharpe_delta(child, par, bpy, stats.seed_from(v["variant_id"]))
            improved = delta > max(0.10, 2 * se)                 # progress must beat the noise band
            pred = v["prediction"]
            if pred["metric"] == "oof_sharpe":
                eff = delta
            else:
                eff = d[pred["metric"]] - parent["diag"][pred["metric"]]
            signed = eff if pred["direction"] == "up" else -eff
            hit = bool(signed >= pred["min_effect"] and (pred["metric"] != "oof_sharpe" or improved))
            key = v["change"]["description"].split(":")[0].strip().lower()
            lesson = next((l for l in lessons if l["change"] == key and l["family"] == idea["family"]), None)
            if lesson is None:
                lesson = {"family": idea["family"], "change": key, "support": 0, "refute": 0,
                          "deltas": [], "arm": v["arm"]}
                lessons.append(lesson)
            lesson["support" if hit else "refute"] += 1
            lesson["deltas"] = lesson["deltas"] + [round(delta, 3)]
        ok, fails = passes_is_gate(d)
        new_best = {"variant_id": v["variant_id"], "code_ref": v["code_ref"], "oof_ref": v["oof_ref"],
                    "diag": d, "param_space": v["param_space"]} if (improved or parent is None) else parent
        idea = {**idea, "best": new_best, "stalls": 0 if improved else idea["stalls"] + 1,
                "changes": idea["changes"] + ([v["change"]["description"]] if v["change"] else [])}
        rec = {"kind": "variant", "variant_id": v["variant_id"], "idea_id": idea["idea_id"], "arm": v["arm"],
               "status": "is_pass" if ok else "is_fail", "gate_fails": fails,
               "oof_sharpe": d["oof_sharpe"], "dsr": d["dsr"], "n_trials": d["n_trials_at_eval"],
               "delta_vs_parent": delta, "delta_se": se, "prediction_hit": hit, "improved": improved,
               "change": v["change"] and v["change"]["description"], "debug_attempts": v["debug_attempts"]}
        out = {"variant": {**v, "status": "is_pass" if ok else "is_fail"}, "idea": idea,
               "attempts": [rec], "lessons": lessons, "reflection": None}
        if ok:
            out["idea"] = {**idea, "passed": True}
            out["eligible"] = [{"variant_id": v["variant_id"], "idea_id": idea["idea_id"],
                                "code_ref": v["code_ref"], "oof_ref": v["oof_ref"],
                                "param_space": v["param_space"], "diag": d}]
        deps.events.emit("is_gate", variant=v["variant_id"], passed=ok, fails=fails,
                         oof_sharpe=d["oof_sharpe"], dsr=d["dsr"], delta=delta, se=se, hit=hit)
        return out

    # -------------------------------------------------------- loop B: reflect / control
    @node(deps, "reflect")
    def reflect(state):
        idea = state["idea"]
        best = idea["best"]
        history = [{"change": a["change"], "delta_vs_parent": a["delta_vs_parent"],
                    "improved": a["improved"], "prediction_hit": a["prediction_hit"]}
                   for a in state.get("attempts", []) if a.get("idea_id") == idea["idea_id"] and a.get("change")]
        payload = context.reflect_payload(idea, deps.store.get_text(best["code_ref"]), best["diag"],
                                          history, state.get("lessons", []))
        visible = context.visible_diagnostics(best["diag"])
        spent_total: dict = {}
        for _ in range(2):
            try:
                out, spent = deps.llm.call("reflect", "strong", Reflection, prompts.REFLECT, payload,
                                           remaining_usd(deps, state) - spent_total.get("usd", 0))
            except HaltError as e:                       # keep the first call's cost on the books
                for k, val in getattr(e, "spent", {}).items():
                    spent_total[k] = spent_total.get(k, 0) + val
                e.spent = spent_total
                raise
            for k, val in spent.items():
                spent_total[k] = spent_total.get(k, 0) + val
            problems = validate_reflection(out, visible, idea["changes"])
            if not problems:
                return {"reflection": {"change": out.change.model_dump(), "prediction": out.prediction.model_dump(),
                                       "control": False},
                        "idea": {**idea, "refine_iter": idea["refine_iter"] + 1}, "spent": spent_total}
            payload = {**payload, "rejected_because": problems}
        deps.events.emit("reflection_rejected", idea=idea["idea_id"], problems=problems)
        return {"idea": {**idea, "stalls": L.max_stalls}, "reflection": None, "spent": spent_total}

    @node(deps, "control_change")
    def control_change(state):
        idea = state["idea"]
        used = {c.split(":")[0] for c in idea["changes"]}
        options = [c for c in CONTROL_CHANGES if c["description"].split(":")[0] not in used] or CONTROL_CHANGES
        pick = options[stats.seed_from(idea["idea_id"], idea["refine_iter"]) % len(options)]
        return {"reflection": {"change": pick, "control": True,
                               "prediction": {"metric": "oof_sharpe", "direction": "up", "min_effect": 0.1}},
                "idea": {**idea, "refine_iter": idea["refine_iter"] + 1}}

    # -------------------------------------------------------- close idea / memory compaction
    @node(deps, "close_idea")
    def close_idea(state):
        idea = state.get("idea")
        out: dict = {"idea": None, "variant": None, "reflection": None}
        if idea:
            best = idea["best"]
            outcome = "is_pass" if idea["passed"] else ("no_valid_variant" if best is None else
                                                        f"is_fail(best_oof_sr={best['diag']['oof_sharpe']:.2f})")
            out["attempts"] = [{"kind": "idea", "idea_id": idea["idea_id"], "family": idea["family"],
                                "thesis": idea["hypothesis"]["thesis"], "outcome": outcome,
                                "prereg_hash": idea["prereg_hash"]}]
        lessons = sorted(state.get("lessons", []), key=lambda l: (l["support"] - l["refute"], len(l["deltas"])),
                         reverse=True)
        fresh = [l for l in lessons if len(l["deltas"]) == 1][:2]        # trial seats for new lessons
        keep = [l for l in lessons if l not in fresh][: 12 - len(fresh)]
        out["lessons"] = keep + fresh
        return out

    # -------------------------------------------------------- endgame (no LLM from here on)
    def _signals_upto(code_ref: str, params: dict, segment: str) -> np.ndarray | None:
        frame = deps.snapshot.upto(segment)
        res = deps.sandbox.run(deps.store.get_text(code_ref), frame, [params])
        return res.signals[0] if res.ok else None

    def _segment_sharpe(code_ref: str, params: dict, segment: str) -> tuple[float, dict] | None:
        sig = _signals_upto(code_ref, params, segment)
        if sig is None:
            return None
        frame = deps.snapshot.upto(segment)
        r = engine.run(frame, sig, cfg.costs)
        sl = deps.snapshot.slice_of(segment)
        m = engine.metrics(r.pnl[sl], r.pos[sl], bpy)
        return m["sharpe"], m

    @node(deps, "finalize")
    def finalize(state):
        snap = deps.snapshot.snapshot_id
        srs = deps.ledger.trial_srs(snap)
        rows = []
        for e in state.get("eligible", []):
            d = e["diag"]
            final_dsr = stats.dsr(stats.per_bar(d["oof_sharpe"], bpy), d["n_obs"], d["skew"], d["kurt"],
                                  len(srs), stats.trial_variance(srs, d["n_obs"]))
            if final_dsr < cfg.is_gate.min_dsr:                   # re-deflate with the FINAL trial count
                continue
            shadow = _segment_sharpe(e["code_ref"], d["last_fold_params"], "shadow")
            if shadow is None or shadow[0] <= 0:
                continue
            rows.append({**e, "final_dsr": f(final_dsr), "shadow_sharpe": f(shadow[0])})
        rows.sort(key=lambda r: r["final_dsr"], reverse=True)
        picked = []
        for r in rows:                                            # prefer uncorrelated finalists
            oof = deps.store.get_array(r["oof_ref"])
            if all(abs(np.corrcoef(oof, deps.store.get_array(p["oof_ref"]))[0, 1]) < 0.7 for p in picked):
                picked.append(r)
            if len(picked) == L.max_finalists:
                break
        order = deps.ledger.register_finalists(state["run_id"], [p["variant_id"] for p in picked])
        by_id = {p["variant_id"]: p for p in picked}
        alpha_each = cfg.oos_gate.alpha / max(len(order), 1)          # Bonferroni over finalists
        oos_rows, champion = [], None
        for vid in order:
            p = by_id.get(vid)
            if p is None:                                             # frozen earlier in a previous attempt
                continue
            prev = deps.ledger.oos_result(snap, vid)
            if prev is None:
                seg = _segment_sharpe(p["code_ref"], p["diag"]["last_fold_params"], "oos")
                if seg is None:
                    continue
                sr_oos, mo = seg
                p_skill = stats.psr(stats.per_bar(sr_oos, bpy), 0.0, mo["n_obs"], mo["skew"], mo["kurt"])
                passed = p_skill >= 1.0 - alpha_each
                try:
                    deps.ledger.record_oos(snap, vid, state["run_id"], sr_oos, passed, L.max_oos_per_snapshot)
                except QuotaExceeded as e:
                    deps.events.emit("oos_quota", detail=str(e))
                    break
            else:
                sr_oos, passed = prev
            oos_rows.append({"variant_id": vid, "oos_sharpe": f(sr_oos), "passed": passed})
            if passed:
                champion = {**p, "oos_sharpe": f(sr_oos)}         # first in pre-registered order
                break                                             # do not spend OOS quota on the rest
        n_obs = int(np.mean([e["diag"]["n_obs"] for e in state.get("eligible", [])])) if state.get("eligible") \
            else len(research) - cfg.wf.initial_train_bars
        var = stats.trial_variance(srs, n_obs)
        final_curve = {"n_trials": len(srs), "required_is_sharpe": round(stats.required_annual_sr(
            max(len(srs), 1), n_obs, bpy, cfg.is_gate.min_dsr, var), 3),
            "trial_sr_dispersion_vs_null": round(var * (n_obs - 1), 2)}
        deps.events.emit("finalize", finalists=order, oos=oos_rows,
                         champion=champion and champion["variant_id"])
        stop = state.get("halt_reason") or budget_reason(deps, state) or "completed"
        return {"finalists": order, "oos": oos_rows, "champion": champion, "stop_reason": stop,
                "final_threshold": final_curve}

    @node(deps, "lockbox_eval")
    def lockbox_eval(state):
        snap, ch = deps.snapshot.snapshot_id, state["champion"]
        used = deps.ledger.lockbox_used(snap)
        if used is not None:
            if used["variant_id"] == ch["variant_id"]:
                return {"lockbox": used}
            return {"lockbox": {"status": "unavailable", "detail": "lockbox already spent on this snapshot "
                                                                   f"by {used['variant_id']}"}}
        seg = _segment_sharpe(ch["code_ref"], ch["diag"]["last_fold_params"], "lockbox")
        if seg is None:                                  # do not burn the one-shot quota on an infra error
            return {"lockbox": {"status": "error", "detail": "sandbox failed; lockbox not consumed"}}
        result = {"status": "evaluated", "sharpe": f(seg[0]), "max_dd": seg[1]["max_dd"],
                  "n_trades": seg[1]["n_trades"]}
        deps.ledger.record_lockbox(snap, ch["variant_id"], state["run_id"], result)
        return {"lockbox": {"variant_id": ch["variant_id"], **result}}

    # -------------------------------------------------------- HITL (no side effects before interrupt)
    def human_review(state):
        ch = state["champion"]
        packet = {
            "champion": ch["variant_id"], "code": deps.store.get_text(ch["code_ref"]),
            "params": ch["diag"]["last_fold_params"],
            "research_oof": context.visible_diagnostics(ch["diag"]),
            "final_dsr": ch["final_dsr"], "shadow_sharpe": ch["shadow_sharpe"],
            "oos_sharpe": ch["oos_sharpe"], "lockbox": state.get("lockbox"),
            "n_trials_on_snapshot": deps.ledger.n_trials(deps.snapshot.snapshot_id),
            "threshold_curve": state["threshold_curve"],
            "assumptions": {"fees": cfg.costs.taker_fee, "slippage": cfg.costs.slippage,
                            "execution": "next bar open", "funding": "charged on the position held at settlement"},
            "final_threshold": state.get("final_threshold"),
            "instructions": "Resume with {decision: approve|reject, reviewer, rationale(>=10 chars)}",
        }
        error = None
        while True:
            answer = interrupt({**packet, "previous_error": error} if error else packet)
            try:
                decision = HumanDecision.model_validate(answer)
                return {"review": decision.model_dump()}
            except ValidationError as e:
                error = str(e.errors()[:3])

    # -------------------------------------------------------- report
    @node(deps, "report")
    def report(state):
        variants = [a for a in state.get("attempts", []) if a.get("kind") == "variant"]
        arms = {}
        for arm in ("reflect", "control"):
            rows = [a for a in variants if a["arm"] == arm and a["status"] != "abandoned"]
            arms[arm] = {"n": len(rows), "improved": sum(bool(a["improved"]) for a in rows),
                         "prediction_hits": sum(bool(a["prediction_hit"]) for a in rows)}
        rate = {k: (v["improved"] / v["n"] if v["n"] else None) for k, v in arms.items()}
        lift = (rate["reflect"] - rate["control"]) if None not in rate.values() else None
        rep = {
            "run_id": state["run_id"], "snapshot": deps.snapshot.snapshot_id, "config": cfg.fingerprint(),
            "halt_reason": state.get("halt_reason"), "halt_detail": state.get("halt_detail"),
            "stop_reason": state.get("stop_reason"), "final_threshold": state.get("final_threshold"),
            "ideas": [a for a in state.get("attempts", []) if a.get("kind") == "idea"],
            "variants": variants, "threshold_curve": state.get("threshold_curve"),
            "n_trials_on_snapshot": deps.ledger.n_trials(deps.snapshot.snapshot_id),
            "eligible": [e["variant_id"] for e in state.get("eligible", [])],
            "finalists": state.get("finalists", []), "oos": state.get("oos", []),
            "champion": state.get("champion") and state["champion"]["variant_id"],
            "lockbox": state.get("lockbox"), "review": state.get("review"),
            "reflection_arms": arms, "reflection_lift": lift,
            "lessons": context.render_lessons(state.get("lessons", [])),
            "spent": state.get("spent"),
        }
        (deps.run_dir / f"report_{state['run_id']}.json").write_text(json.dumps(rep, indent=2, default=str))
        deps.events.emit("report", run_id=state["run_id"], champion=rep["champion"],
                         halt=rep["halt_reason"], usd=(state.get("spent") or {}).get("usd"))
        return {"report": rep}

    return {
        "init_run": init_run, "strategist": strategist, "coder": coder, "static_check": static_check,
        "sandbox_validate": sandbox_validate, "debugger": debugger, "abandon_variant": abandon_variant,
        "backtest_is": backtest_is, "gate_is": gate_is, "reflect": reflect, "control_change": control_change,
        "close_idea": close_idea, "finalize": finalize, "lockbox_eval": lockbox_eval,
        "human_review": human_review, "report": report,
    }


def validate_reflection(r: Reflection, visible: dict, previous_changes: list[str]) -> list[str]:
    problems = []
    for ev in r.evidence:
        if ev.ref not in visible or not isinstance(visible[ev.ref], (int, float)):
            problems.append(f"evidence ref '{ev.ref}' is not a numeric diagnostic")
            continue
        actual = float(visible[ev.ref])
        ok = {">": actual > ev.value, "<": actual < ev.value, ">=": actual >= ev.value - 1e-9,
              "<=": actual <= ev.value + 1e-9, "==": abs(actual - ev.value) <= 1e-6}[ev.op]
        if not ok:
            problems.append(f"evidence '{ev.ref} {ev.op} {ev.value}' is false (actual {actual})")
    key = r.change.description.split(":")[0].strip().lower()
    if key in {c.split(":")[0].strip().lower() for c in previous_changes}:
        problems.append(f"change '{key}' was already tried on this idea")
    return problems
