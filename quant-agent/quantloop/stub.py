"""Deterministic offline backend so the whole graph runs without API keys.

It behaves like a (mediocre) researcher: rotates strategy families, writes
template code, and - when `inject_bugs` is on - deliberately makes the three
classic mistakes (a NameError, a negative shift, a whole-sample z-score) so
the self-debugging loop and both look-ahead guards are exercised.
"""
from __future__ import annotations

import json
import re

from .llm import (Change, CoderOut, DebugOut, Evidence, Hypothesis, Prediction, Reflection,
                  StrategistOut, Usage)

FAMILIES = ["trend", "breakout", "momentum", "mean_reversion", "funding_carry"]

BODIES = {
    "trend": (
        {"fast": [12, 24, 48], "slow": [96, 192, 336]},
        """    c = df["close"]
    fast = c.ewm(span=params["fast"], adjust=False).mean()
    slow = c.ewm(span=params["slow"], adjust=False).mean()
    raw = np.sign(fast - slow)""",
    ),
    "breakout": (
        {"n": [48, 96, 168, 336]},
        """    hi = df["high"].rolling(params["n"]).max().shift(1)
    lo = df["low"].rolling(params["n"]).min().shift(1)
    ev = pd.Series(np.where(df["close"] > hi, 1.0, np.where(df["close"] < lo, -1.0, np.nan)), index=df.index)
    raw = ev.ffill().fillna(0.0)""",
    ),
    "momentum": (
        {"n": [72, 168, 336, 720]},
        """    raw = np.sign(df["close"].pct_change(params["n"])).fillna(0.0)""",
    ),
    "mean_reversion": (
        {"n": [24, 48, 96], "k": [1.5, 2.0]},
        """    c = df["close"]
    z = (c - c.rolling(params["n"]).mean()) / c.rolling(params["n"]).std()
    raw = (-z / params["k"]).clip(-1, 1).fillna(0.0)""",
    ),
    "funding_carry": (
        {"n": [24, 72, 168]},
        """    f = df["funding_rate"].rolling(params["n"]).mean()
    raw = -np.sign(f).fillna(0.0)""",
    ),
}

FILTERS = {
    "vol_filter": """    vol = df["close"].pct_change().rolling(168).std()
    calm = vol <= vol.rolling(24 * 90, min_periods=24 * 30).median() * 1.5
    raw = raw.where(calm, 0.0)""",
    "long_only": """    raw = raw.clip(lower=0.0)""",
    "smooth": """    raw = raw.rolling(12, min_periods=1).mean()""",
}

BUGS = {
    "name_error": ("fast - slow", "fast - slw"),
    "neg_shift": ('df["close"]', 'df["close"].shift(-1)'),
}


def render(spec: dict, bug: str | None = None) -> str:
    space, body = BODIES[spec["family"]]
    body = "\n".join(line for line in body.splitlines())
    if bug == "whole_sample_z":
        body = """    c = df["close"]
    z = (c - c.mean()) / c.std()
    raw = (-z / params["k"]).clip(-1, 1).fillna(0.0)"""
    elif bug in BUGS:
        old, new = BUGS[bug]
        body = body.replace(old, new, 1)
    parts = [body, "    raw = pd.Series(np.asarray(raw, dtype=float), index=df.index)"]
    parts += [FILTERS[f] for f in spec.get("filters", [])]
    parts.append("    return raw.clip(-1, 1)")
    return (f"# SPEC: {json.dumps(spec, sort_keys=True)}\n"
            "import numpy as np\nimport pandas as pd\n\n"
            f"PARAM_SPACE = {json.dumps(space)}\n\n\n"
            "def generate_signals(df, params):\n" + "\n".join(parts) + "\n")


def spec_of(code: str) -> dict:
    m = re.search(r"# SPEC: (\{.*\})", code or "")
    return json.loads(m.group(1)) if m else {"family": "trend", "filters": []}


class StubBackend:
    def __init__(self, inject_bugs: bool = True):
        self.inject_bugs = inject_bugs
        self.payloads: list[tuple[str, dict]] = []     # for isolation tests

    def invoke(self, tier, schema, system, user, max_output_tokens):
        payload = json.loads(user.split("\n\nYour previous output")[0])
        self.payloads.append((schema.__name__, payload))
        out = getattr(self, "_" + schema.__name__)(payload)
        return out.model_dump(), Usage(len(user) // 4 + len(system) // 4, 400, f"stub-{tier}"), "end_turn"

    # -------------------------------------------------------------- roles
    def _StrategistOut(self, p):
        tried = {t["family"] for t in p.get("tried", [])}
        order = [f for f in FAMILIES if f not in tried] + FAMILIES
        return StrategistOut(candidates=[
            Hypothesis(family=f, thesis=f"{f}: behavioural under-reaction / forced flows",
                       signal_description=f"classic {f} signal on 1h bars",
                       expected_behavior="positive in both halves of the research sample")
            for f in order[:3]
        ])

    def _CoderOut(self, p):
        if p.get("change"):
            spec = spec_of(p["parent_code"])
            spec = _apply_change(spec, p["change"])
            return CoderOut(code=render(spec), notes=f"applied: {p['change']['description']}")
        spec = {"family": p["hypothesis"]["family"], "filters": []}
        bug = None
        if self.inject_bugs:
            bug = {1: "name_error", 2: "neg_shift"}.get(p["idea_seq"])
            if spec["family"] == "mean_reversion":
                bug = "whole_sample_z"
        return CoderOut(code=render(spec, bug))

    def _DebugOut(self, p):
        return DebugOut(explanation=f"fixing {p['error_class']}: {p['error'][:80]}",
                        code=render(spec_of(p["code"])))

    def _Reflection(self, p):
        d = p["diagnostics"]
        spec = spec_of(p["code"])
        if d["fee_x2_sharpe"] < 0.5 * d["oof_sharpe"] and "smooth" not in spec.get("filters", []):
            ch = Change(kind="exit_logic", description="smooth: average position over 12 bars to cut turnover")
            ev = [Evidence(ref="fee_x2_sharpe", op="<", value=round(0.5 * d["oof_sharpe"], 4) + 1e-4)]
        elif d["short_sharpe"] < 0 < d["long_sharpe"] and "long_only" not in spec.get("filters", []):
            ch = Change(kind="filter", description="long_only: drop the losing short side")
            ev = [Evidence(ref="short_sharpe", op="<", value=0.0), Evidence(ref="long_sharpe", op=">", value=0.0)]
        else:
            ch = Change(kind="filter", description="vol_filter: stand aside in abnormally high volatility")
            ev = [Evidence(ref="max_dd", op=">=", value=d["max_dd"])]
        return Reflection(diagnosis="see evidence", evidence=ev, change=ch,
                          prediction=Prediction(metric="oof_sharpe", direction="up", min_effect=0.1))


def _apply_change(spec: dict, change: dict) -> dict:
    spec = {**spec, "filters": list(spec.get("filters", []))}
    key = change["description"].split(":")[0].strip()
    if key in FILTERS and key not in spec["filters"]:
        spec["filters"].append(key)
    return spec
