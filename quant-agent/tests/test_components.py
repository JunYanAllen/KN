import dataclasses
import os

import numpy as np
import pytest

from quantloop import engine, stats
from quantloop.config import Config
from quantloop.data import synthetic_snapshot
from quantloop.ledger import Ledger, QuotaExceeded
from quantloop.safety import ast_rules
from quantloop.safety.sandbox import SubprocessSandbox
from quantloop.stub import render

CFG = Config()
GOOD = render({"family": "trend", "filters": []})


@pytest.fixture(scope="module")
def frame():
    return synthetic_snapshot(CFG.splits, years=1.0).df


# ------------------------------------------------------------------ static checks
@pytest.mark.parametrize("snippet,cls", [
    ('df["close"].shift(-1)', "E_LOOKAHEAD_STATIC"),
    ('df["close"].rolling(5, center=True).mean()', "E_LOOKAHEAD_STATIC"),
    ('df["close"].bfill()', "E_LOOKAHEAD_STATIC"),
    ('df["close"].to_csv("x")', "E_FORBIDDEN"),
    ('df.__class__', "E_FORBIDDEN"),
    ('eval("1")', "E_FORBIDDEN"),
    ('df.loc["2021-05-01":]', "E_FORBIDDEN"),
])
def test_static_rejects(snippet, cls):
    code = GOOD.replace("    return raw.clip(-1, 1)", f"    _x = {snippet}\n    return raw.clip(-1, 1)")
    assert ast_rules.check(code, 4, 16).error_class == cls


@pytest.mark.parametrize("snippet", [
    'pd.read_table("/etc/passwd")', 'df["close"].to_numpy().tofile("x")', 'np.fromfile("x")',
    'np.savetxt("x", [1])', 'pd.io.common', 'np.lib', 'np.random.rand(3)', 'df.query("close > 0")',
])
def test_static_blocks_io_and_module_escapes(snippet):
    code = GOOD.replace("    return raw.clip(-1, 1)", f"    _x = {snippet}\n    return raw.clip(-1, 1)")
    assert ast_rules.check(code, 4, 16).error_class == "E_FORBIDDEN"


def test_static_blocks_monkey_patching_and_submodule_imports():
    patched = GOOD.replace("PARAM_SPACE", "np.isclose = abs\nPARAM_SPACE", 1)
    assert ast_rules.check(patched, 4, 16).error_class == "E_FORBIDDEN"
    assert ast_rules.check("import numpy.lib\n" + GOOD, 4, 16).error_class == "E_IMPORT"
    assert ast_rules.check("from os import system\n" + GOOD, 4, 16).error_class == "E_IMPORT"
    mixed = GOOD.replace('"fast": [12, 24, 48]', '"fast": [12, None, [1]]')
    assert ast_rules.check(mixed, 4, 16).error_class == "E_INTERFACE"


def test_static_rejects_imports_and_big_grids():
    assert ast_rules.check("import os\n" + GOOD, 4, 16).error_class == "E_IMPORT"
    big = GOOD.replace('"fast": [12, 24, 48]', '"fast": [1, 2, 3, 4, 5, 6]')
    assert ast_rules.check(big, 4, 16).error_class == "E_INTERFACE"


def test_static_accepts_template():
    rep = ast_rules.check(GOOD, 4, 16)
    assert rep.ok and len(ast_rules.expand_grid(rep.param_space)) == 9


# ------------------------------------------------------------------ sandbox
def test_sandbox_runs_and_returns_grid(frame):
    grid = ast_rules.expand_grid(ast_rules.check(GOOD, 4, 16).param_space)
    res = SubprocessSandbox(60).run(GOOD, frame, grid)
    assert res.ok and res.signals.shape == (9, len(frame))


def test_truncation_invariance_catches_whole_sample_stats(frame):
    code = render({"family": "mean_reversion", "filters": []}, bug="whole_sample_z")
    assert ast_rules.check(code, 4, 16).ok                 # static filter cannot see it...
    grid = ast_rules.expand_grid(ast_rules.check(code, 4, 16).param_space)
    res = SubprocessSandbox(60).run(code, frame, grid)
    assert res.error_class == "E_LOOKAHEAD_DYNAMIC"       # ...the invariance test does


def test_invariance_checks_every_param_combo(frame):
    # only one of four combos peeks ahead, via a computed negative period the AST cannot see
    code = GOOD.replace('{"fast": [12, 24, 48], "slow": [96, 192, 336]}', '{"lag": [1, 2, 3, -1]}').replace(
        """    c = df["close"]
    fast = c.ewm(span=params["fast"], adjust=False).mean()
    slow = c.ewm(span=params["slow"], adjust=False).mean()
    raw = np.sign(fast - slow)""", """    r = df["close"].pct_change()
    raw = np.sign(r.shift(0 + params["lag"])).fillna(0.0)""")
    rep = ast_rules.check(code, 4, 16)
    assert rep.ok
    grid = ast_rules.expand_grid(rep.param_space)
    res = SubprocessSandbox(60).run(code, frame, grid)
    assert res.error_class == "E_LOOKAHEAD_DYNAMIC"
    assert grid[res.detail["combo"]] == {"lag": -1}


def test_sandbox_files_are_readable_by_unprivileged_container_user(tmp_path, frame):
    from quantloop.safety.sandbox import DockerSandbox, _prepare
    work = tmp_path / "w"
    work.mkdir(mode=0o700)
    _prepare(work, GOOD, frame.iloc[:100], [{"fast": 12, "slow": 96}], [50, 99])
    assert oct(work.stat().st_mode)[-3:] == "755"
    assert oct((work / "harness.py").stat().st_mode)[-3:] == "644"
    assert oct((work / "out").stat().st_mode)[-3:] == "777"
    cmd = DockerSandbox("img@sha256:abc").command(work, "n1")
    assert {"--network", "none", "--read-only", "--cap-drop", "ALL"} <= set(cmd)


def test_sandbox_classifies_runtime_errors_and_timeouts(frame):
    bad = render({"family": "trend", "filters": []}, bug="name_error")
    res = SubprocessSandbox(60).run(bad, frame, [{"fast": 12, "slow": 96}])
    assert res.error_class == "E_RUNTIME" and "NameError" in res.message
    loop = GOOD.replace("    c = df[\"close\"]", "    while True:\n        pass")
    res = SubprocessSandbox(3).run(loop, frame, [{"fast": 12, "slow": 96}])
    assert res.error_class in ("E_TIMEOUT", "E_RUNTIME")


def test_sandbox_environment_has_no_secrets(frame, monkeypatch):
    monkeypatch.setenv("FAKE_EXCHANGE_API_KEY", "sk-live-123")
    code = GOOD.replace("import numpy as np", "import os\nimport numpy as np\nLEAK = os.environ.get("
                        "'FAKE_EXCHANGE_API_KEY')").replace(
        "    c = df[\"close\"]", "    assert LEAK is None, 'secret visible'\n    c = df[\"close\"]")
    res = SubprocessSandbox(60).run(code, frame, [{"fast": 12, "slow": 96}])   # AST bypassed on purpose
    assert res.ok, res.message


def test_signals_out_of_range_rejected(frame):
    code = GOOD.replace("    return raw.clip(-1, 1)", "    return raw * 3")
    res = SubprocessSandbox(60).run(code, frame, [{"fast": 12, "slow": 96}])
    assert res.error_class == "E_SCHEMA"


# ------------------------------------------------------------------ engine
def test_engine_executes_next_bar_and_charges_costs(frame):
    sig = np.ones(len(frame))
    r = engine.run(frame, sig, CFG.costs)
    assert r.pos[0] == 0 and (r.pos[1:] == 1).all()        # executed one bar later
    o = frame["open"].to_numpy()
    expected_bar1 = (o[2] / o[1] - 1) - (CFG.costs.taker_fee + CFG.costs.slippage) - frame["funding_rate"].iloc[1]
    assert np.isclose(r.pnl[1], expected_bar1)


def test_same_bar_peeking_signal_earns_nothing_special(frame):
    # A signal equal to *this* bar's open->next open move would be perfect foresight if executed now;
    # the engine executes it one bar later, so it is just noise.
    o = frame["open"].to_numpy()
    fut = np.sign(np.append(o[1:] / o[:-1] - 1, 0))
    free = dataclasses.replace(CFG.costs, taker_fee=0.0, slippage=0.0)   # isolate timing from costs
    sr = engine.sharpe(engine.run(frame, fut, free).pnl, CFG.costs.bars_per_year)
    assert abs(sr) < 3


# ------------------------------------------------------------------ stats
def test_dsr_penalises_more_trials():
    sr = stats.per_bar(2.0, 8760)
    var = stats.trial_variance([], 20000)
    vals = [stats.dsr(sr, 20000, 0, 3, n, var) for n in (1, 10, 100)]
    assert vals[0] > vals[1] > vals[2]
    curve = [stats.required_annual_sr(n, 20000, 8760) for n in (1, 5, 20, 40)]
    assert curve == sorted(curve)
    assert stats.expected_max_sr(1, 1.0) == 0.0


def test_oos_gate_needs_real_evidence():
    # 1.2 years of hourly OOS: SR 0.5 is indistinguishable from noise, SR 2.5 is not
    n = int(1.2 * 8760)
    alpha_each = Config().oos_gate.alpha / 2
    weak = stats.psr(stats.per_bar(0.5, 8760), 0.0, n, 0, 3)
    strong = stats.psr(stats.per_bar(2.5, 8760), 0.0, n, 0, 3)
    assert weak < 1 - alpha_each < strong


def test_fold_boundary_param_switch_pays_turnover():
    pnl = np.zeros((10, 2))
    pos = np.array([[1.0] * 10, [-1.0] * 10]).T
    out = engine._stitch(pnl, pos, [(0, 5, 0), (5, 10, 1)], rate=0.001)
    assert np.isclose(out[5], -2 * 0.001)            # flipping +1 -> -1 at the boundary


def test_seed_is_deterministic_across_processes():
    assert stats.seed_from("run", 3) == stats.seed_from("run", 3) != stats.seed_from("run", 4)


# ------------------------------------------------------------------ ledger
def test_ledger_quotas_and_idempotency(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    for _ in range(2):
        led.record_trial("v1", "snap", "r", "i", "initial", 0.01, 100, 0, 3, "h", {})
    assert led.n_trials("snap") == 1
    for i in range(3):
        led.record_oos("snap", f"v{i}", "r", 0.5, True, lifetime_cap=3)
    led.record_oos("snap", "v0", "r", 0.5, True, lifetime_cap=3)   # re-record: no-op
    with pytest.raises(QuotaExceeded):
        led.record_oos("snap", "v9", "r", 0.5, True, lifetime_cap=3)
    led.record_lockbox("snap", "v0", "r", {"sharpe": 1})
    with pytest.raises(QuotaExceeded):
        led.record_lockbox("snap", "v1", "r2", {"sharpe": 2})
    assert led.register_finalists("r", ["a", "b"]) == ["a", "b"]
    assert led.register_finalists("r", ["b", "a"]) == ["a", "b"]    # order frozen
    assert os.path.exists(tmp_path / "l.sqlite")
