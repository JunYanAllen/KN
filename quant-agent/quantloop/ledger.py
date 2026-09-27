"""Persistent, cross-run trial ledger (SQLite).

It is the source of truth for multiple-testing corrections and for the
one-shot OOS / lockbox quotas. Quotas live here (not in graph state or in
memory) so they survive new threads, new processes and resumed runs.
All writes are idempotent on deterministic keys, so a retried node never
double counts.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class QuotaExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS trials(
            variant_id TEXT PRIMARY KEY, snapshot TEXT, run_id TEXT, idea_id TEXT,
            arm TEXT, sr_bar REAL, n_obs INTEGER, skew REAL, kurt REAL, code_hash TEXT,
            metrics TEXT, ts REAL);
        CREATE TABLE IF NOT EXISTS oos_used(
            snapshot TEXT, variant_id TEXT, run_id TEXT, sr_annual REAL, passed INTEGER, ts REAL,
            PRIMARY KEY(snapshot, variant_id));
        CREATE TABLE IF NOT EXISTS lockbox_used(
            snapshot TEXT PRIMARY KEY, variant_id TEXT, run_id TEXT, result TEXT, ts REAL);
        CREATE TABLE IF NOT EXISTS prereg(
            idea_id TEXT PRIMARY KEY, run_id TEXT, prereg_hash TEXT, body TEXT, ts REAL);
        CREATE TABLE IF NOT EXISTS finalists(
            run_id TEXT, rank INTEGER, variant_id TEXT, PRIMARY KEY(run_id, rank));
        """)
        self.db.commit()

    # ---------------------------------------------------------------- trials
    def record_trial(self, variant_id: str, snapshot: str, run_id: str, idea_id: str, arm: str,
                     sr_bar: float, n_obs: int, skew: float, kurt: float, code_hash: str,
                     metrics: dict) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO trials VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (variant_id, snapshot, run_id, idea_id, arm, sr_bar, n_obs, skew, kurt, code_hash,
             json.dumps(metrics), time.time()))
        self.db.commit()

    def trial_srs(self, snapshot: str) -> list[float]:
        """Per-bar SR of *every* performance trial on this snapshot, across all runs."""
        return [r[0] for r in self.db.execute("SELECT sr_bar FROM trials WHERE snapshot=?", (snapshot,))]

    def has_code(self, snapshot: str, code_hash: str) -> str | None:
        r = self.db.execute("SELECT variant_id FROM trials WHERE snapshot=? AND code_hash=?",
                            (snapshot, code_hash)).fetchone()
        return r[0] if r else None

    def n_trials(self, snapshot: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM trials WHERE snapshot=?", (snapshot,)).fetchone()[0]

    # ---------------------------------------------------------------- prereg
    def preregister(self, idea_id: str, run_id: str, prereg_hash: str, body: dict) -> None:
        self.db.execute("INSERT OR IGNORE INTO prereg VALUES(?,?,?,?,?)",
                        (idea_id, run_id, prereg_hash, json.dumps(body), time.time()))
        self.db.commit()

    def register_finalists(self, run_id: str, variant_ids: list[str]) -> list[str]:
        """Freeze the finalist order *before* any OOS is looked at. Idempotent."""
        existing = [r[0] for r in self.db.execute(
            "SELECT variant_id FROM finalists WHERE run_id=? ORDER BY rank", (run_id,))]
        if existing:
            return existing
        self.db.executemany("INSERT INTO finalists VALUES(?,?,?)",
                            [(run_id, i, v) for i, v in enumerate(variant_ids)])
        self.db.commit()
        return variant_ids

    # ---------------------------------------------------------------- OOS
    def oos_result(self, snapshot: str, variant_id: str) -> tuple[float, bool] | None:
        r = self.db.execute("SELECT sr_annual, passed FROM oos_used WHERE snapshot=? AND variant_id=?",
                            (snapshot, variant_id)).fetchone()
        return (r[0], bool(r[1])) if r else None

    def oos_count(self, snapshot: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM oos_used WHERE snapshot=?", (snapshot,)).fetchone()[0]

    def record_oos(self, snapshot: str, variant_id: str, run_id: str, sr_annual: float,
                   passed: bool, lifetime_cap: int) -> None:
        with self.db:
            if self.oos_result(snapshot, variant_id) is not None:
                return
            if self.oos_count(snapshot) >= lifetime_cap:
                raise QuotaExceeded(f"OOS lifetime cap {lifetime_cap} reached for snapshot {snapshot}")
            self.db.execute("INSERT INTO oos_used VALUES(?,?,?,?,?,?)",
                            (snapshot, variant_id, run_id, sr_annual, int(passed), time.time()))

    # ---------------------------------------------------------------- lockbox
    def lockbox_used(self, snapshot: str) -> dict | None:
        r = self.db.execute("SELECT variant_id, result FROM lockbox_used WHERE snapshot=?", (snapshot,)).fetchone()
        return {"variant_id": r[0], **json.loads(r[1])} if r else None

    def record_lockbox(self, snapshot: str, variant_id: str, run_id: str, result: dict) -> None:
        with self.db:
            if self.lockbox_used(snapshot) is not None:
                raise QuotaExceeded(f"lockbox already used for snapshot {snapshot}")
            self.db.execute("INSERT INTO lockbox_used VALUES(?,?,?,?,?)",
                            (snapshot, variant_id, run_id, json.dumps(result), time.time()))
