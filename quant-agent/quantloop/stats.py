"""Multiple-testing aware statistics (Bailey & Lopez de Prado).

All Sharpe ratios here are *per bar* unless the name says `annual`.
"""
from __future__ import annotations

import hashlib
import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def per_bar(sr_annual: float, bars_per_year: int) -> float:
    return sr_annual / math.sqrt(bars_per_year)


def _sr_denominator(sr: float, skew: float, kurt: float) -> float:
    v = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return math.sqrt(max(v, 1e-12))


def psr(sr: float, sr_star: float, n_obs: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_star)."""
    if n_obs < 3:
        return 0.0
    z = (sr - sr_star) * math.sqrt(n_obs - 1) / _sr_denominator(sr, skew, kurt)
    return _N.cdf(z)


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """E[max SR] among n_trials independent zero-skill trials (the DSR benchmark)."""
    if n_trials <= 1:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def trial_variance(trial_srs: list[float], n_obs: int) -> float:
    """Cross-trial variance of per-bar SR, floored at the null sampling variance.

    With only a handful of trials the sample variance is unreliable (and can be
    ~0 for near-duplicate variants), which would silently weaken the deflation.
    """
    null_var = 1.0 / max(n_obs - 1, 1)
    if len(trial_srs) < 2:
        return null_var
    return max(float(np.var(trial_srs, ddof=1)), null_var)


def dsr(sr: float, n_obs: int, skew: float, kurt: float,
        n_trials: int, var_sr: float) -> float:
    """Deflated Sharpe Ratio. `n_trials` and `var_sr` must describe the same trial set."""
    return psr(sr, expected_max_sr(n_trials, var_sr), n_obs, skew, kurt)


def required_annual_sr(n_trials: int, n_obs: int, bars_per_year: int,
                       min_dsr: float = 0.95) -> float:
    """Threshold curve: annual SR needed to pass DSR>=min_dsr at n_trials (normal returns).

    Published at run start so that humans know how strict the gate will get.
    """
    var = 1.0 / (n_obs - 1)
    sr0 = expected_max_sr(n_trials, var)
    sr = sr0 + _N.inv_cdf(min_dsr) / math.sqrt(n_obs - 1)
    return sr * math.sqrt(bars_per_year)


def seed_from(*parts: object) -> int:
    """Deterministic seed (Python's hash() is salted per process)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def paired_sharpe_delta(child: np.ndarray, parent: np.ndarray, bars_per_year: int,
                        seed: int, n_boot: int = 300, block: int = 24 * 7) -> tuple[float, float]:
    """(delta annual SR, block-bootstrap SE of the delta) on the same OOF bars."""
    from .engine import sharpe

    n = min(len(child), len(parent))
    child, parent = child[:n], parent[:n]
    delta = sharpe(child, bars_per_year) - sharpe(parent, bars_per_year)
    rng = np.random.default_rng(seed)
    n_blocks = max(n // block, 1)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()
        idx = idx[idx < n]
        deltas[i] = sharpe(child[idx], bars_per_year) - sharpe(parent[idx], bars_per_year)
    return float(delta), float(deltas.std(ddof=1))
