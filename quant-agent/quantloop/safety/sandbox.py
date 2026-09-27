"""Execute LLM-written signal code out of process.

`SubprocessSandbox` is the local/dev runner: a fresh interpreter per call,
empty environment (no API keys), rlimits and a hard wall-clock kill. It does
NOT isolate the network or filesystem - use `DockerSandbox` (same harness,
disposable container, --network none, read-only, non-root) for anything
beyond local experiments.

Inputs cross the boundary as .npz (no pickle) and outputs come back as .npy
loaded with allow_pickle=False, so the host never deserializes objects
produced by untrusted code.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HARNESS = r'''
import json, sys, traceback
import numpy as np, pandas as pd

def main():
    work = sys.argv[1]
    spec = json.load(open(f"{work}/spec.json"))
    z = np.load(f"{work}/data.npz", allow_pickle=False)
    idx = pd.to_datetime(z["index_ns"], utc=True)
    df = pd.DataFrame({c: z[c] for c in spec["columns"]}, index=idx)
    src = open(f"{work}/strategy.py").read()
    code = compile(src, "strategy.py", "exec")

    def call(frame, params):
        ns = {"__name__": "strategy"}          # fresh namespace per call: no cross-call cache
        exec(code, ns)
        out = ns["generate_signals"](frame.copy(), dict(params))
        if isinstance(out, pd.Series):
            if len(out) != len(frame) or not out.index.equals(frame.index):
                raise ValueError("E_SCHEMA: returned Series index must equal df.index")
            out = out.to_numpy()
        out = np.asarray(out, dtype=float)
        if out.shape != (len(frame),):
            raise ValueError(f"E_SCHEMA: expected shape ({len(frame)},), got {out.shape}")
        return out

    status = {"ok": True}
    try:
        # Truncated prefixes first, full series last.
        inv = []
        for k in spec["invariance_combos"]:
            for t in spec["cuts"]:
                inv.append((k, t, call(df.iloc[:t], spec["grid"][k])))
        signals = np.stack([call(df, p) for p in spec["grid"]])
        mismatches = []
        for k, t, part in inv:
            full = signals[k, :t]
            bad = ~(np.isclose(part, full, atol=1e-9, equal_nan=True))
            if bad.any():
                mismatches.append({"combo": k, "cut": t, "first_bad_bar": int(np.argmax(bad)), "n_bad": int(bad.sum())})
        status["invariance_mismatches"] = mismatches
        np.save(f"{work}/out/signals.npy", signals, allow_pickle=False)
    except Exception as e:
        tb = traceback.format_exc(limit=6)
        tb = tb[-2500:]
        status = {"ok": False, "error_type": type(e).__name__, "error": str(e)[:500], "traceback": tb}
    json.dump(status, open(f"{work}/out/status.json", "w"))

main()
'''


@dataclass
class SandboxResult:
    ok: bool
    error_class: str | None = None
    message: str = ""
    signals: np.ndarray | None = None          # (K, n)
    detail: dict = field(default_factory=dict)


def _prepare(work: Path, code: str, df: pd.DataFrame, grid: list[dict],
             cuts: list[int], invariance_combos: list[int]) -> None:
    (work / "out").mkdir()
    (work / "strategy.py").write_text(code)
    (work / "harness.py").write_text(HARNESS)
    cols = list(df.columns)
    np.savez(work / "data.npz", index_ns=df.index.asi8, **{c: df[c].to_numpy() for c in cols})
    (work / "spec.json").write_text(json.dumps(
        {"columns": cols, "grid": grid, "cuts": cuts, "invariance_combos": invariance_combos}))


def _collect(work: Path, n: int, k: int) -> SandboxResult:
    status_path = work / "out" / "status.json"
    if not status_path.exists():
        return SandboxResult(False, "E_RUNTIME", "sandbox produced no status (killed by rlimit?)")
    status = json.loads(status_path.read_text())
    if not status.get("ok"):
        msg = f"{status.get('error_type')}: {status.get('error')}\n{status.get('traceback', '')}"
        cls = "E_SCHEMA" if "E_SCHEMA" in status.get("error", "") else "E_RUNTIME"
        return SandboxResult(False, cls, msg)
    sig = np.load(work / "out" / "signals.npy", allow_pickle=False)
    if sig.shape != (k, n):
        return SandboxResult(False, "E_SCHEMA", f"signals shape {sig.shape} != {(k, n)}")
    if status["invariance_mismatches"]:
        m = status["invariance_mismatches"][0]
        return SandboxResult(False, "E_LOOKAHEAD_DYNAMIC",
                             f"truncation-invariance failed: signal at bar {m['first_bad_bar']} changes when "
                             f"data after bar {m['cut']} is removed ({m['n_bad']} bars differ). "
                             "The signal must only use data up to and including the current bar.",
                             detail={"mismatches": status["invariance_mismatches"]})
    return validate_signals(sig)


def validate_signals(sig: np.ndarray) -> SandboxResult:
    n = sig.shape[1]
    warmup = int(0.1 * n)
    if np.isnan(sig[:, warmup:]).any():
        return SandboxResult(False, "E_SCHEMA", f"NaN after warm-up (first {warmup} bars may be NaN)")
    finite = np.nan_to_num(sig)
    if np.abs(finite).max() > 1 + 1e-9:
        return SandboxResult(False, "E_SCHEMA", "signal outside [-1, 1]")
    active = (finite != 0).mean(axis=1)
    if (active < 0.01).all():
        return SandboxResult(False, "E_DEGENERATE", "signal is (almost) always flat for every param combo")
    return SandboxResult(True, signals=finite)


def _cuts(n: int) -> list[int]:
    return sorted({int(n * f) for f in (0.35, 0.6, 0.85)})


def _inv_combos(k: int) -> list[int]:
    return sorted({0, k // 2, k - 1})


class SubprocessSandbox:
    def __init__(self, timeout_s: float = 60.0, mem_bytes: int = 2 << 30):
        self.timeout_s = timeout_s
        self.mem_bytes = mem_bytes

    def _limits(self):
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (self.mem_bytes, self.mem_bytes))
        cpu = int(self.timeout_s) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (200 << 20, 200 << 20))
        os.setsid()

    def run(self, code: str, df: pd.DataFrame, grid: list[dict]) -> SandboxResult:
        work = Path(tempfile.mkdtemp(prefix="ql-sbx-"))
        try:
            _prepare(work, code, df, grid, _cuts(len(df)), _inv_combos(len(grid)))
            env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            try:
                subprocess.run([sys.executable, "-I", str(work / "harness.py"), str(work)],
                               cwd=work, env=env, preexec_fn=self._limits,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               timeout=self.timeout_s, check=False)
            except subprocess.TimeoutExpired:
                return SandboxResult(False, "E_TIMEOUT", f"exceeded {self.timeout_s:.0f}s")
            except OSError as e:
                return SandboxResult(False, "E_INFRA", f"could not start sandbox: {e}")
            return _collect(work, len(df), len(grid))
        finally:
            shutil.rmtree(work, ignore_errors=True)


class DockerSandbox(SubprocessSandbox):
    """Same harness in a disposable container. Requires Docker (+ gVisor recommended).

    The image must contain python, numpy and pandas only, pinned by digest.
    """

    def __init__(self, image: str, timeout_s: float = 60.0, runtime: str | None = "runsc"):
        super().__init__(timeout_s)
        self.image = image
        self.runtime = runtime

    def command(self, work: Path, name: str) -> list[str]:
        cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
               "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--user", "65534:65534",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
               "--pids-limit", "64", "--memory", "2g", "--memory-swap", "2g", "--cpus", "1",
               "-v", f"{work}:/work:ro", "-v", f"{work / 'out'}:/work/out:rw"]
        if self.runtime:
            cmd += ["--runtime", self.runtime]
        return cmd + [self.image, "python", "-I", "/work/harness.py", "/work"]

    def run(self, code: str, df: pd.DataFrame, grid: list[dict]) -> SandboxResult:
        work = Path(tempfile.mkdtemp(prefix="ql-sbx-"))
        try:
            _prepare(work, code, df, grid, _cuts(len(df)), _inv_combos(len(grid)))
            os.chmod(work / "out", 0o777)
            name = f"ql-sbx-{work.name}"
            try:
                subprocess.run(self.command(work, name), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               timeout=self.timeout_s, check=False)
            except subprocess.TimeoutExpired:
                # Killing the docker client does not stop the container.
                subprocess.run(["docker", "kill", name], capture_output=True, check=False)
                return SandboxResult(False, "E_TIMEOUT", f"exceeded {self.timeout_s:.0f}s")
            except OSError as e:
                return SandboxResult(False, "E_INFRA", f"docker unavailable: {e}")
            return _collect(work, len(df), len(grid))
        finally:
            shutil.rmtree(work, ignore_errors=True)
