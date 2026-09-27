"""Market data snapshot + chronological split.

`synthetic_snapshot` exists so the loop can run end-to-end offline. For real
research, build a `Snapshot` from exchange OHLCV + historical funding rates and
keep the same split logic.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Splits

SEGMENTS = ("research", "shadow", "oos", "lockbox")


@dataclass(frozen=True)
class Snapshot:
    df: pd.DataFrame                 # open, high, low, close, volume, funding_rate
    snapshot_id: str                 # content hash; ledger quotas are per snapshot
    bounds: dict[str, tuple[int, int]]  # segment -> [start, end) bar index

    def upto(self, segment: str) -> pd.DataFrame:
        """All bars from the beginning through the end of `segment`.

        Signal functions are verified causal, so a finalist's signals on a
        later segment are computed on the full prefix and then sliced.
        """
        return self.df.iloc[: self.bounds[segment][1]]

    def slice_of(self, segment: str) -> slice:
        a, b = self.bounds[segment]
        return slice(a, b)


def _hash_frame(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def make_snapshot(df: pd.DataFrame, splits: Splits) -> Snapshot:
    required = {"open", "high", "low", "close", "volume", "funding_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"snapshot missing columns: {sorted(missing)}")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("snapshot index must be strictly increasing (UTC bar open time)")
    n = len(df)
    fracs = [splits.research, splits.shadow, splits.oos, splits.lockbox]
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError("splits must sum to 1")
    edges = np.cumsum([0] + [int(round(f * n)) for f in fracs])
    edges[-1] = n
    bounds = {s: (int(edges[i]), int(edges[i + 1])) for i, s in enumerate(SEGMENTS)}
    return Snapshot(df=df, snapshot_id=_hash_frame(df), bounds=bounds)


def synthetic_snapshot(splits: Splits, years: float = 5.0, seed: int = 7,
                       trend_strength: float = 0.000012) -> Snapshot:
    """Regime-switching random walk with a weak, persistent trend component.

    There is a small amount of genuine (in-sample and out-of-sample) trend
    persistence so that honest trend strategies have something to find,
    but most of what the loop sees is noise - like the real thing.
    """
    rng = np.random.default_rng(seed)
    n = int(years * 365 * 24)
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        regime[t] = regime[t - 1] if rng.random() > 1 / 500 else rng.integers(0, 3)
    vol = np.array([0.004, 0.009, 0.006])[regime]
    drift = np.zeros(n)
    ar = 0.0
    for t in range(n):
        ar = 0.995 * ar + rng.normal(0, trend_strength)
        drift[t] = ar
    rets = drift + vol * rng.standard_t(5, size=n) / np.sqrt(5 / 3)
    close = 20000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0, 1, n)) * vol * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.lognormal(10, 0.5, n)
    funding = np.where(np.arange(n) % 8 == 0, rng.normal(0.0001, 0.0002, n), 0.0)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume, "funding_rate": funding},
        index=idx,
    )
    return make_snapshot(df, splits)
