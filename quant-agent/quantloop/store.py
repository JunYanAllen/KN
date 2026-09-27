"""Content-addressed artifact store + JSONL event log.

Graph state only carries artifact refs (hashes); code, signal grids and
per-bar PnL live on disk here.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_text(self, text: str) -> str:
        ref = hashlib.sha256(text.encode()).hexdigest()[:20]
        path = self.root / f"{ref}.py"
        if not path.exists():
            path.write_text(text)
        return ref

    def get_text(self, ref: str) -> str:
        return (self.root / f"{ref}.py").read_text()

    def put_array(self, arr: np.ndarray) -> str:
        arr = np.ascontiguousarray(arr)
        ref = hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()[:20]
        path = self.root / f"{ref}.npy"
        if not path.exists():
            np.save(path, arr, allow_pickle=False)
        return ref

    def get_array(self, ref: str) -> np.ndarray:
        return np.load(self.root / f"{ref}.npy", allow_pickle=False)


class EventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **fields) -> None:
        rec = {"ts": round(time.time(), 3), "kind": kind, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str, sort_keys=True) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
