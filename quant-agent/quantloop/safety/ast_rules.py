"""Static pre-filter for LLM-written strategy code.

This is *not* the security boundary (the sandbox is). It exists to reject
obvious interface violations and common look-ahead patterns cheaply, with an
error message specific enough for the debugger to fix.
"""
from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass, field

ALLOWED_IMPORTS = {"numpy", "pandas", "math"}          # exactly these modules, no submodules

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit",
    "memoryview", "__builtins__",
}
# Attributes that read the future, touch I/O or reach other modules.
LOOKAHEAD_METHODS = {"bfill", "backfill", "interpolate"}
FORBIDDEN_ATTRS = {
    "load", "save", "savez", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile", "fromregex",
    "memmap", "system", "popen", "spawn", "os", "sys", "subprocess", "ctypes", "ctypeslib", "builtins",
    "importlib", "import_module", "io", "lib", "core", "api", "util", "compat", "testing", "plotting",
    "options", "f2py", "distutils", "random", "eval", "query", "pipe",
}
ALLOWED_TO_METHODS = {"to_numpy", "to_list"}
DATE_LITERAL = re.compile(r"\b(19|20)\d{2}-\d{2}")


@dataclass
class StaticReport:
    ok: bool
    error_class: str | None = None
    message: str = ""
    param_space: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _fail(cls: str, msg: str) -> StaticReport:
    return StaticReport(ok=False, error_class=cls, message=msg)


def check(code: str, max_params: int, max_grid: int) -> StaticReport:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return _fail("E_SYNTAX", f"line {e.lineno}: {e.msg}")

    warnings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            return _fail("E_IMPORT", f"line {node.lineno}: use 'import numpy as np' / 'import pandas as pd' / "
                                     "'import math' only")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    return _fail("E_IMPORT", f"line {node.lineno}: import of '{a.name}' is not allowed "
                                             f"(allowed: {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Attribute):     # no monkey-patching of modules/objects
                        return _fail("E_FORBIDDEN", f"line {node.lineno}: assigning to attribute "
                                                    f"'{sub.attr}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return _fail("E_FORBIDDEN", f"line {node.lineno}: global/nonlocal state is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return _fail("E_FORBIDDEN", f"line {node.lineno}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_"):
                return _fail("E_FORBIDDEN", f"line {node.lineno}: private/dunder attribute '{a}'")
            if a in LOOKAHEAD_METHODS:
                return _fail("E_LOOKAHEAD_STATIC", f"line {node.lineno}: '.{a}' uses future bars")
            if a in FORBIDDEN_ATTRS or a.startswith("read_"):
                return _fail("E_FORBIDDEN", f"line {node.lineno}: '.{a}' is not allowed")
            if a.startswith("to_") and a not in ALLOWED_TO_METHODS:
                return _fail("E_FORBIDDEN", f"line {node.lineno}: '.{a}' (I/O) is not allowed")
        elif isinstance(node, ast.Call):
            msg = _lookahead_call(node)
            if msg:
                return _fail("E_LOOKAHEAD_STATIC", f"line {node.lineno}: {msg}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and DATE_LITERAL.search(node.value):
            # Date-conditioned logic is a classic overfit; reject rather than warn.
            return _fail("E_FORBIDDEN", f"line {node.lineno}: date literal '{node.value}' "
                                        "(calendar-specific rules are not allowed)")

    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "generate_signals" not in funcs:
        return _fail("E_INTERFACE", "missing top-level function generate_signals(df, params)")
    if [a.arg for a in funcs["generate_signals"].args.args] != ["df", "params"]:
        return _fail("E_INTERFACE", "signature must be exactly generate_signals(df, params)")

    space = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARAM_SPACE" for t in n.targets):
            try:
                space = ast.literal_eval(n.value)
            except ValueError:
                return _fail("E_INTERFACE", "PARAM_SPACE must be a literal dict of lists")
        elif not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.Expr)):
            return _fail("E_INTERFACE", f"line {n.lineno}: only imports, constants and functions "
                                        "are allowed at module level")
    if not isinstance(space, dict) or not space or not all(isinstance(v, list) and v for v in space.values()):
        return _fail("E_INTERFACE", "PARAM_SPACE must be a non-empty dict of non-empty lists")
    for k, vals in space.items():
        if not isinstance(k, str) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals):
            return _fail("E_INTERFACE", f"PARAM_SPACE['{k}'] must be a list of numbers")
        if len(set(vals)) != len(vals):
            return _fail("E_INTERFACE", f"PARAM_SPACE['{k}'] has duplicate values")
    if len(space) > max_params:
        return _fail("E_INTERFACE", f"PARAM_SPACE has {len(space)} params (max {max_params})")
    grid_size = 1
    for v in space.values():
        grid_size *= len(v)
    if grid_size > max_grid:
        return _fail("E_INTERFACE", f"PARAM_SPACE grid has {grid_size} combos (max {max_grid})")
    return StaticReport(ok=True, param_space=space, warnings=warnings)


def _lookahead_call(node: ast.Call) -> str | None:
    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    if name == "shift":
        arg = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "periods"), None)
        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
            return "shift() with a negative period reads future bars"
        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
            return "shift() with a negative period reads future bars"
        # computed periods (e.g. shift(0 - n)) are left to the dynamic invariance test, which
        # checks every PARAM_SPACE combination
    if name == "rolling":
        for k in node.keywords:
            if k.arg == "center" and isinstance(k.value, ast.Constant) and k.value.value is True:
                return "rolling(center=True) uses future bars"
    if name == "fillna":
        for k in node.keywords:
            if k.arg == "method" and isinstance(k.value, ast.Constant) and k.value.value in ("bfill", "backfill"):
                return "fillna(method='bfill') uses future bars"
    return None


def expand_grid(space: dict) -> list[dict]:
    keys = sorted(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]
