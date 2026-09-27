"""What each LLM role is allowed to see - the single place information
barriers are enforced. Nothing derived from shadow / OOS / lockbox data, and
no gate thresholds or DSR values, can reach a prompt through here.
"""
from __future__ import annotations

VISIBLE_DIAGNOSTICS = (
    "oof_sharpe", "max_dd", "n_trades", "exposure", "turnover_per_year",
    "long_sharpe", "short_sharpe", "fold_sharpes", "pos_fold_frac",
    "fee_x2_sharpe", "delay_sharpe", "plateau_ratio",
    "first_half_sharpe", "second_half_sharpe", "high_vol_sharpe", "low_vol_sharpe",
)
FORBIDDEN_SUBSTRINGS = ("oos", "lockbox", "shadow", "dsr", "gate", "threshold")


def visible_diagnostics(diag: dict) -> dict:
    return {k: diag[k] for k in VISIBLE_DIAGNOSTICS if k in diag}


def render_lessons(lessons: list[dict]) -> list[str]:
    out = []
    for l in lessons:
        n = l["support"] + l["refute"]
        med = sorted(l["deltas"])[len(l["deltas"]) // 2] if l["deltas"] else 0.0
        out.append(f"[{l['family']}] {l['change']} -> median dOOF_SR {med:+.2f} "
                   f"(confirmed {l['support']}/{n})")
    return out


def assert_clean(payload: object, path: str = "") -> None:
    """Defence in depth: fail loudly if a forbidden key ever reaches a prompt."""
    if isinstance(payload, dict):
        for k, v in payload.items():
            if any(s in str(k).lower() for s in FORBIDDEN_SUBSTRINGS):
                raise AssertionError(f"forbidden key in LLM payload: {path}.{k}")
            assert_clean(v, f"{path}.{k}")
    elif isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            assert_clean(v, f"{path}[{i}]")


def market_brief(cfg) -> dict:
    return {"symbol": cfg.symbol, "timeframe": cfg.timeframe,
            "costs": {"taker_fee": cfg.costs.taker_fee, "slippage": cfg.costs.slippage,
                      "funding_every_bars": cfg.costs.funding_every_bars},
            "execution": "signal at bar t is executed at bar t+1 open"}


def strategist_payload(cfg, attempts: list[dict], lessons: list[dict]) -> dict:
    tried = [{"family": a["family"], "thesis": a["thesis"], "outcome": a["outcome"]}
             for a in attempts if a.get("kind") == "idea"]
    p = {"market": market_brief(cfg), "tried": tried, "lessons": render_lessons(lessons)}
    assert_clean(p)
    return p


def coder_payload(cfg, idea: dict, idea_seq: int, parent_code: str | None, change: dict | None) -> dict:
    p = {"market": market_brief(cfg), "hypothesis": idea["hypothesis"], "idea_seq": idea_seq,
         "limits": {"max_params": cfg.limits.max_params, "max_grid": cfg.limits.max_param_grid}}
    if change:
        p["parent_code"] = parent_code
        p["change"] = change
    assert_clean(p)
    return p


def debugger_payload(code: str, error_class: str, error: str, attempt: int) -> dict:
    p = {"code": code, "error_class": error_class, "error": error, "attempt": attempt}
    assert_clean(p)
    return p


def reflect_payload(idea: dict, code: str, diag: dict, history: list[dict], lessons: list[dict]) -> dict:
    p = {"hypothesis": idea["hypothesis"], "code": code, "diagnostics": visible_diagnostics(diag),
         "history": history, "lessons": render_lessons(lessons)}
    assert_clean(p)
    return p
