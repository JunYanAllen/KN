"""Graph state: small, JSON-serializable values and artifact refs only."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


def add_spent(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = round(out.get(k, 0) + v, 6)
    return out


class ResearchState(TypedDict, total=False):
    # run identity (fixed at init)
    run_id: str
    snapshot_id: str
    config_fp: str
    threshold_curve: dict              # {n_trials: annual SR needed to pass DSR} - published up front

    # loop C: ideas
    idea_seq: int
    idea: dict | None                  # current IdeaCard (pre-registered hypothesis + refine counters)
    # loops A/B: the variant being built / debugged / evaluated
    variant: dict | None
    reflection: dict | None            # the ONE change the next coder call must apply

    # memory
    attempts: Annotated[list[dict], operator.add]     # append-only research log
    lessons: list[dict]                                # scored, compacted (overwritten by close_idea)
    eligible: Annotated[list[dict], operator.add]     # variants that passed the IS gate

    # control
    spent: Annotated[dict, add_spent]  # usd, tokens_in, tokens_out, llm_calls, backtests, active_s
    halt_reason: str | None
    halt_detail: str | None

    # endgame (deterministic)
    finalists: list[str]
    oos: list[dict]
    champion: dict | None
    lockbox: dict | None
    review: dict | None
    report: dict | None
