"""Fixed, deterministic, human-reviewed backtest engine.

The LLM never writes any of this. It only supplies target positions in
[-1, 1]; timing, costs and metrics are decided here, identically for every
variant:

* the signal computed with data up to bar t's close is executed at bar t+1's
  open (``pos = signal.shift(1 + delay)``) and earns the open-to-open return;
* every unit of turnover pays taker fee + slippage;
* funding is charged on the position held at each funding bar
  (long pays positive funding).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Costs, WalkForward


@dataclass
class Result:
    pnl: np.ndarray        # per-bar net return
    pos: np.ndarray

    def metrics(self, bars_per_year: int) -> dict:
        return metrics(self.pnl, self.pos, bars_per_year)


def run(df: pd.DataFrame, signal: np.ndarray, costs: Costs,
        fee_mult: float = 1.0, delay: int = 0) -> Result:
    sig = np.clip(np.nan_to_num(np.asarray(signal, dtype=float)), -1.0, 1.0)
    pos = np.roll(sig, 1 + delay)
    pos[: 1 + delay] = 0.0
    o = df["open"].to_numpy()
    ret = np.zeros(len(o))
    ret[:-1] = o[1:] / o[:-1] - 1.0          # earned while holding bar t
    turnover = np.abs(np.diff(pos, prepend=0.0))
    funding = df["funding_rate"].to_numpy()
    pnl = pos * ret - turnover * (costs.taker_fee + costs.slippage) * fee_mult - pos * funding
    return Result(pnl=pnl, pos=pos)


def sharpe(pnl: np.ndarray, bars_per_year: int) -> float:
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    return 0.0 if sd == 0 or not np.isfinite(sd) else float(pnl.mean() / sd * math.sqrt(bars_per_year))


def metrics(pnl: np.ndarray, pos: np.ndarray, bars_per_year: int) -> dict:
    equity = np.cumprod(1.0 + pnl)
    peak = np.maximum.accumulate(equity)
    max_dd = float(((peak - equity) / peak).max()) if len(equity) else 0.0
    changes = np.abs(np.diff(pos, prepend=0.0)) > 1e-12
    long_mask, short_mask = pos > 0, pos < 0
    s = pd.Series(pnl)
    return {
        "sharpe": round(sharpe(pnl, bars_per_year), 4),
        "max_dd": round(max_dd, 4),
        "n_trades": int(changes.sum()),
        "exposure": round(float((pos != 0).mean()), 4),
        "turnover_per_year": round(float(np.abs(np.diff(pos, prepend=0.0)).sum() / max(len(pos), 1) * bars_per_year), 2),
        "long_sharpe": round(sharpe(pnl[long_mask], bars_per_year), 4) if long_mask.sum() > 2 else 0.0,
        "short_sharpe": round(sharpe(pnl[short_mask], bars_per_year), 4) if short_mask.sum() > 2 else 0.0,
        "skew": round(float(s.skew()) if len(s) > 2 else 0.0, 4),
        "kurt": round(float(s.kurt() + 3.0) if len(s) > 3 else 3.0, 4),
        "n_obs": int(len(pnl)),
    }


# --------------------------------------------------------------------------- walk-forward

@dataclass
class WFResult:
    oof_pnl: np.ndarray            # concatenated out-of-fold pnl
    oof_pos: np.ndarray
    oof_index: np.ndarray          # bar indices covered by OOF
    fold_sharpes: list[float]
    chosen: list[int]              # grid index picked per fold (inner selection)
    folds: list[tuple[int, int, int]]  # (test_start, test_end, chosen k)

    def apply(self, pnl_grid: np.ndarray) -> np.ndarray:
        """Same fold/param selection applied to a stressed grid (fees x2, delay...)."""
        return np.concatenate([pnl_grid[a:b, k] for a, b, k in self.folds])


def walk_forward(pnl_grid: np.ndarray, pos_grid: np.ndarray, wf: WalkForward,
                 bars_per_year: int) -> WFResult:
    """Expanding-window WF over a (bars x K) grid of per-combo pnl.

    Parameter selection happens *inside* each training window; the OOF test
    fold is never used to choose anything.
    """
    n, _ = pnl_grid.shape
    oof_pnl, oof_pos, oof_idx, folds, chosen, spans = [], [], [], [], [], []
    start = wf.initial_train_bars
    while start + wf.test_bars // 2 <= n:
        train_end = max(start - wf.embargo_bars, 1)
        train_sr = [sharpe(pnl_grid[:train_end, k], bars_per_year) for k in range(pnl_grid.shape[1])]
        k = int(np.argmax(train_sr))
        end = min(start + wf.test_bars, n)
        seg = pnl_grid[start:end, k]
        oof_pnl.append(seg)
        oof_pos.append(pos_grid[start:end, k])
        oof_idx.append(np.arange(start, end))
        folds.append(round(sharpe(seg, bars_per_year), 4))
        chosen.append(k)
        spans.append((start, end, k))
        start = end
    if not oof_pnl:
        raise ValueError("research segment too short for walk-forward")
    return WFResult(np.concatenate(oof_pnl), np.concatenate(oof_pos),
                    np.concatenate(oof_idx), folds, chosen, spans)


def plateau_ratio(pnl_grid: np.ndarray, grid: list[dict], bars_per_year: int) -> float:
    """median(sharpe of one-step neighbours of the best combo) / sharpe(best)."""
    if len(grid) <= 1:
        return 1.0
    srs = [sharpe(pnl_grid[:, k], bars_per_year) for k in range(len(grid))]
    best = int(np.argmax(srs))
    if srs[best] <= 0:
        return 0.0
    keys = list(grid[best].keys())
    values = {k: sorted({g[k] for g in grid}) for k in keys}

    def is_neighbour(g: dict) -> bool:
        diffs = [k for k in keys if g[k] != grid[best][k]]
        if len(diffs) != 1:
            return False
        k = diffs[0]
        return abs(values[k].index(g[k]) - values[k].index(grid[best][k])) == 1

    neigh = [srs[i] for i, g in enumerate(grid) if is_neighbour(g)]
    return round(float(np.median(neigh)) / srs[best], 4) if neigh else 1.0
