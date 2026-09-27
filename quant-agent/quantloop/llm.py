"""Structured-output LLM layer: schemas, budget pre-check, refusal handling.

Providers plug in behind `LLMBackend`. `LangChainBackend` wraps any LangChain
chat model (Anthropic, OpenAI, ...) through `with_structured_output`; the
offline `StubBackend` lives in `stub.py`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError


# ------------------------------------------------------------------ errors
class HaltError(Exception):
    """Errors that must end the research loop gracefully (never crash the run)."""
    reason = "halt"


class BudgetExceeded(HaltError):
    reason = "budget"


class LLMRefusal(HaltError):
    reason = "refusal"


class SchemaFailure(HaltError):
    reason = "schema_failure"


class LLMUnavailable(HaltError):
    reason = "llm_unavailable"


class TransientLLMError(Exception):
    """Raise from a backend for retryable failures (timeouts, 429, 5xx)."""


# ------------------------------------------------------------------ schemas
Family = Literal["trend", "breakout", "mean_reversion", "momentum", "funding_carry"]


class Hypothesis(BaseModel):
    family: Family
    thesis: str = Field(description="Economic rationale: who is on the other side and why the edge persists")
    signal_description: str
    expected_behavior: str = Field(description="Falsifiable pre-registered prediction, e.g. 'works in trending "
                                               "high-vol regimes, long and short both positive'")


class StrategistOut(BaseModel):
    candidates: list[Hypothesis] = Field(min_length=1, max_length=3)


class CoderOut(BaseModel):
    code: str = Field(description="Python module defining PARAM_SPACE and generate_signals(df, params)")
    notes: str = ""


class DebugOut(BaseModel):
    explanation: str = Field(description="What the error means and which line causes it")
    code: str


class Evidence(BaseModel):
    ref: str = Field(description="Key from the diagnostics payload, e.g. 'fee_x2_sharpe'")
    op: Literal[">", "<", ">=", "<=", "=="]
    value: float


class Change(BaseModel):
    kind: Literal["param_space", "entry_logic", "exit_logic", "filter", "position_sizing"]
    description: str


class Prediction(BaseModel):
    metric: Literal["oof_sharpe", "max_dd", "n_trades", "fee_x2_sharpe"]
    direction: Literal["up", "down"]
    min_effect: float = Field(gt=0)


class Reflection(BaseModel):
    diagnosis: str
    evidence: list[Evidence] = Field(min_length=1, max_length=5)
    change: Change
    prediction: Prediction


class HumanDecision(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1)
    rationale: str = Field(min_length=10)


# ------------------------------------------------------------------ backend protocol
@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    model: str


class LLMBackend(Protocol):
    def invoke(self, tier: str, schema: type[BaseModel], system: str, user: str,
               max_output_tokens: int) -> tuple[BaseModel | None, Usage, str | None]:
        """Return (parsed or None, usage, stop_reason)."""


class LangChainBackend:
    """Adapter for LangChain chat models, one per tier ('strong' / 'cheap')."""

    def __init__(self, models: dict[str, Any]):
        self.models = models

    def invoke(self, tier, schema, system, user, max_output_tokens):
        model = self.models[tier].bind(max_tokens=max_output_tokens)
        runnable = model.with_structured_output(schema, include_raw=True)
        try:
            out = runnable.invoke([("system", system), ("user", user)])
        except Exception as e:  # noqa: BLE001 - classify provider errors
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            if status in (408, 409, 429) or (isinstance(status, int) and status >= 500) or isinstance(e, TimeoutError):
                raise TransientLLMError(str(e)) from e
            raise
        raw = out["raw"]
        meta = getattr(raw, "usage_metadata", None) or {}
        stop = (getattr(raw, "response_metadata", None) or {}).get("stop_reason")
        usage = Usage(meta.get("input_tokens", 0), meta.get("output_tokens", 0),
                      (getattr(raw, "response_metadata", None) or {}).get("model_name", tier))
        return out.get("parsed"), usage, stop


# ------------------------------------------------------------------ client
def _with_spent(exc: HaltError, spent: dict) -> HaltError:
    exc.spent = spent          # tokens already paid for are still charged to the run
    return exc


class LLMClient:
    def __init__(self, backend: LLMBackend, pricing: dict, prompt_version: str,
                 on_call=None, transient_retries: int = 3):
        self.backend = backend
        self.pricing = pricing
        self.prompt_version = prompt_version
        self.on_call = on_call               # observability hook
        self.transient_retries = transient_retries

    def _cost(self, tier: str, tin: int, tout: int) -> float:
        p = self.pricing[tier]
        return (tin * p["in"] + tout * p["out"]) / 1e6

    def call(self, role: str, tier: str, schema: type[BaseModel], system: str, payload: dict,
             remaining_usd: float, max_output_tokens: int = 2000) -> tuple[BaseModel, dict]:
        """Returns (parsed, spent_delta). Raises HaltError subclasses only."""
        user = json.dumps(payload, sort_keys=True, default=str)
        est = self._cost(tier, int((len(system) + len(user)) / 3.5), max_output_tokens)
        if est > remaining_usd:
            raise BudgetExceeded(f"{role}: estimated ${est:.3f} > remaining ${remaining_usd:.3f}")
        spent = {"usd": 0.0, "tokens_in": 0, "tokens_out": 0, "llm_calls": 0}
        feedback = ""
        for schema_attempt in range(2):
            try:
                parsed, usage, stop = self._invoke_with_retry(tier, schema, system, user + feedback,
                                                              max_output_tokens)
            except HaltError as e:
                raise _with_spent(e, spent)
            cost = self._cost(tier, usage.input_tokens, usage.output_tokens)
            spent["usd"] += cost
            spent["tokens_in"] += usage.input_tokens
            spent["tokens_out"] += usage.output_tokens
            spent["llm_calls"] += 1
            if self.on_call:
                self.on_call(role=role, tier=tier, model=usage.model, prompt_version=self.prompt_version,
                             payload=payload, usd=cost, stop_reason=stop)
            if stop == "refusal":
                raise _with_spent(LLMRefusal(f"{role}: model refused"), spent)
            if parsed is not None:
                try:
                    return schema.model_validate(parsed), spent
                except ValidationError as e:
                    feedback = f"\n\nYour previous output failed validation: {e.errors()[:3]}"
            else:
                feedback = "\n\nYour previous output could not be parsed into the required schema."
            if spent["usd"] + est > remaining_usd:
                break
        raise _with_spent(SchemaFailure(f"{role}: no valid structured output after retry"), spent)

    def _invoke_with_retry(self, tier, schema, system, user, max_output_tokens):
        delay = 1.0
        for i in range(self.transient_retries):
            try:
                return self.backend.invoke(tier, schema, system, user, max_output_tokens)
            except TransientLLMError:
                if i == self.transient_retries - 1:
                    raise LLMUnavailable("provider unavailable after retries")
                time.sleep(delay)
                delay *= 2
        raise LLMUnavailable("unreachable")
