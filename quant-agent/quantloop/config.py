"""All tunable limits and gates live here.

Nothing in this file is ever shown to an LLM: gate thresholds are enforced by
routers (deterministic code), not by prompts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Splits:
    """Chronological data split, as fractions of the full history.

    research: walk-forward loop (the only numbers the LLM ever sees)
    shadow:   hidden tail of IS, used once per finalist by a router
    oos:      each pre-registered finalist sees it at most once (0 bits to LLM)
    lockbox:  champion only, once per data snapshot, result goes to humans only
    """
    research: float = 0.52
    shadow: float = 0.12
    oos: float = 0.24          # OOS power matters more than anything else: keep it long
    lockbox: float = 0.12


@dataclass(frozen=True)
class WalkForward:
    initial_train_bars: int = 24 * 180     # 6 months of 1h bars
    test_bars: int = 24 * 90               # 3-month OOF folds
    embargo_bars: int = 48                 # >= 2x typical holding period


@dataclass(frozen=True)
class Costs:
    taker_fee: float = 0.0005              # per unit turnover
    slippage: float = 0.0002               # per unit turnover
    funding_every_bars: int = 8            # 8h funding on 1h bars
    bars_per_year: int = 24 * 365


@dataclass(frozen=True)
class Limits:
    max_ideas: int = 6                     # strategy re-ideation loop (C)
    max_refines_per_idea: int = 3          # param/logic refine loop (B)
    max_debug_attempts: int = 4            # execution-error repair loop (A)
    max_same_error: int = 3                # same normalized error seen 3x -> give up (ladder fully tried)
    max_stalls: int = 2                    # consecutive non-improving refines
    max_finalists: int = 2
    max_oos_per_snapshot: int = 3          # lifetime, across runs (ledger-enforced)
    max_param_grid: int = 16
    max_params: int = 4
    max_trials_per_snapshot: int = 40      # raw variant count
    usd: float = 12.0
    active_seconds: float = 3 * 3600       # excludes human wait time
    sandbox_timeout_s: float = 60.0
    control_arm_every: int = 5             # every 5th refine is an unguided control change


@dataclass(frozen=True)
class ISGate:
    min_dsr: float = 0.95
    min_trades: int = 100
    min_pos_fold_frac: float = 0.6
    fee_x2_min_sharpe: float = 0.0
    delay_min_retention: float = 0.3       # sharpe(delay 1 bar) / sharpe
    plateau_min_ratio: float = 0.5         # median neighbour sharpe / best sharpe
    suspicious_sharpe: float = 4.0         # flag as probable bug, never pass


@dataclass(frozen=True)
class OOSGate:
    # One-sided test that the OOS Sharpe is > 0, family-wise alpha split over the
    # pre-registered finalists (Bonferroni). A fixed "SR >= 0.5" rule would pass
    # pure noise ~31% of the time on one year of data.
    alpha: float = 0.10


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Placeholder values; set real prices before use."""
    table: dict = field(default_factory=lambda: {
        "strong": {"in": 3.0, "out": 15.0},
        "cheap": {"in": 1.0, "out": 5.0},
    })


@dataclass(frozen=True)
class Config:
    symbol: str = "BTCUSDT-PERP"
    timeframe: str = "1h"
    splits: Splits = Splits()
    wf: WalkForward = WalkForward()
    costs: Costs = Costs()
    limits: Limits = Limits()
    is_gate: ISGate = ISGate()
    oos_gate: OOSGate = OOSGate()
    pricing: Pricing = Pricing()
    prompt_version: str = "v1"

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
