# 候選架構 A：強化版 Reflexion 單線迴圈

> 設計者：架構師 A　｜　對應需求：`requirements.md`（硬性約束 1–11 全數納入）
> 模型等級對照：T1 = `claude-opus-5`（策略發想、反思）、T2 = `claude-sonnet-5`（寫碼、多數除錯）、T3 = `claude-haiku-4-5`（語法修復、記憶壓縮）。價格以 2026-06 快取表為準：T1 $5 / $25、T2 $2 / $10、T3 $1 / $5（每百萬 input / output token），快取讀取約 0.1 倍。

---

## 1. 一句話定位與適用情境

**一句話定位**：一次只追一個假說、一條路徑走到底的研究迴圈。每次失敗都會轉成一條「可證偽、帶證據、之後會被驗證」的語言化教訓，寫進情節記憶（Reflexion 式 verbal reinforcement），下一輪注入 prompt。達標與否只由程式判定，LLM 只負責提出假說和批判。

**適用情境**
- MVP 或第一版：團隊 1–2 人，要在 1–2 週內跑通「LLM 產生策略 → 固定引擎回測 → 過擬合把關 → 人工審閱」的完整閉環。
- 預算敏感：每個 run 的預算上限約 USD 15，也不需要平行叢集。
- 策略空間偏窄、假說家族明確（趨勢、均值回歸、突破、資金費率 carry、波動率），主要瓶頸在「把一個想法做對、驗乾淨」，而不是「廣度搜尋」。
- 需要高可解釋性：單一執行緒、事件序列線性，審計與重現都最容易。

**不適用**：需要大量平行探索、組合型策略搜尋、跨資產 alpha 挖掘，或對 wall-clock 很敏感的情境（見第 11 節弱點）。

---

## 2. ASCII 架構圖

```
 START
   │
   ▼
 init_run ──(確定性) 資料快照hash / IS·OOS·lockbox 切分 / 開帳本 / 預算計時開始
   │
   ▼
┌──────────────────────────┐ ◄───────────────────────────────────────────────┐
│ strategist  [T1, LLM]    │  新想法 + 預登記假說（可證偽預測、kill criteria）  │
│ idea_iter += 1           │                                                  │ (C) 重新發想迴圈
└────────────┬─────────────┘                                                  │  idea_iter ≤ 8 / run
             ▼                                                                │  家族輪替 + 想法目錄去重
┌──────────────────────────┐ ◄──────────────────────────────────┐             │
│ coder       [T2, LLM]    │  generate_signals + PARAM_SPACE     │ (B) 改良迴圈 │
└────────────┬─────────────┘                                     │ refine ≤ 5 / idea
             ▼                                                   │ 停滯偵測 / 預測連錯 → 強制 (C)
   ┌──► static_check [確定性] AST 掃描·import 白名單·前視規則·      │             │
   │         │          PARAM_SPACE schema·正規化 code hash 去重    │             │
   │         │ ok                         fail ──┐               │             │
   │         ▼                                   │               │             │
   │   sandbox_validate [確定性,拋棄式容器]        │               │             │
   │     輸出 schema·值域·NaN·截斷不變性·逾時·     │               │             │
   │     行為指紋去重                              │               │             │
   │         │ ok                         fail ──┤               │             │
   │         │                                   ▼               │             │
   │ (A) 除錯短迴圈                    [route: 分類 + 計數]         │             │
   │  debug_iter ≤ 4 / 變體            ├─► debugger [T3/T2] ──┐   │             │
   │  同 error-hash ≤ 2               └─► abandon_variant ────┼──►[route_after_abandon]
   │  各類別上限（見 §7）                  (寫確定性 lesson)     │   │  └► coder / strategist / finalize
   └──────────────────────────────────────────────────────────┘   │             │
             │ ok (不經 Evaluator、不計試驗)                         │             │
             ▼                                                   │             │
 backtest_is [確定性,容器] nested walk-forward + purge/embargo, 只用 IS │             │
             ▼                                                   │             │
 record_trial [確定性] 全域試驗帳本寫入 · DSR(N=帳本試驗數) ·         │             │
                      上輪預測逐條驗證 · lessons 支持/反駁計數        │             │
             ▼                                                   │             │
      [route_after_is_gate]  ◄── 硬性門檻寫死在程式                  │             │
        pass │            │ fail                                 │             │
             ▼            │                                      │             │
 oos_check [確定性]        │                                      │             │
  每變體 1 次、每 run ≤ 6 次│                                      │             │
  只回傳 1 bit 給 LLM       │                                      │             │
        pass │   fail ────┤                                      │             │
             │            ▼                                      │             │
             │   reflect [T1, LLM] 結構化 critique + ≤2 lessons + 單一改動 + 可證偽預測
             │            ▼                                      │             │
             │   compact_memory [確定性 + 偶爾 T3] 教訓上限 12 條   │             │
             │            ▼                                      │             │
             │   [route_after_reflect] ──refine──────────────────┘             │
             │            └──────new idea─────────────────────────────────────┘
             ▼
 finalize [確定性] 選候選（DSR 最高且 OOS 通過）· 計算 PBO(CSCV) · 產報告
             ▼
 lockbox_eval [確定性] 只跑 1 次；結果只寫給人看，prompt 白名單排除 + canary 檢查
             ▼
 human_review ◆ interrupt()   ← 獨立節點，前面沒有副作用；SqliteSaver/PostgresSaver + thread_id
             ▼                   Command(resume={"decision": ...})
      [route_after_human] ── approve / reject ──► END
                          └─ continue（lockbox 已燒掉，後續候選標記 no-lockbox）─► strategist

 ※ 預算檢查：每個 router 的第一行都是 budget_exceeded(state)，超標就 → finalize
   （USD、token、wall-clock、回測次數、OOS 次數；LLM wrapper 另有 per-call 硬上限）
 ※ recursion_limit 設 1500，只當最後保險；正常終止靠計數器
```

三種迴圈與上限一覽：

| 迴圈 | 路徑 | 計數器 | 上限 | 會寫入試驗帳本嗎 | 經過 Evaluator/反思嗎 |
|---|---|---|---|---|---|
| (A) 執行錯誤修復 | static_check / sandbox_validate → debugger → static_check | `debug_iter`（每變體歸零） | 4；同 error-hash 2 次；各類別上限 | 不寫 | 不經過 |
| (B) 參數／邏輯改良 | reflect → coder → … → backtest_is | `refine_iter`（每想法歸零） | 5 | 寫（每個 PARAM_SPACE 組態都算試驗） | 經過 |
| (C) 策略重新發想 | reflect/abandon → strategist | `idea_iter` | 8 / run | 寫 | 經過 |
| 全域預算 | 所有 router | `budget` | USD 15、3M token、6h、60 次 IS 回測、6 次 OOS | — | — |

---

## 3. State schema

```python
from __future__ import annotations
import operator
from typing import Annotated, Literal, TypedDict, NotRequired

# ---------- 小型值物件（全部可 JSON 序列化） ----------

Family = Literal["trend", "mean_reversion", "breakout", "funding_carry",
                 "volatility", "seasonality", "microstructure_proxy"]

MetricKey = Literal[  # 預測與證據只能引用這些 key，程式負責取值與比較
    "median_fold_sharpe", "pos_fold_frac", "worst_fold_sharpe", "n_trades",
    "max_dd", "sharpe_2x_cost", "turnover_per_day", "avg_hold_bars",
    "win_rate", "top5_trade_pnl_share", "sharpe_high_vol", "sharpe_low_vol",
    "sharpe_trend_regime", "sharpe_range_regime", "long_sharpe", "short_sharpe",
    "funding_pnl_share", "plateau_ratio", "exposure_frac",
]

class Prediction(TypedDict):
    pred_id: str
    metric: MetricKey
    comparator: Literal["gt", "lt", "between", "delta_gt", "delta_lt"]  # delta_* = 相對上一變體
    value: float
    value_hi: float | None
    source: Literal["strategist", "reflection"]
    made_at_variant: str | None         # 預測在哪個變體提出
    lesson_ids: list[str]               # 這個預測依據哪些 lesson（用來回頭計分）

class PredictionResult(TypedDict):
    pred_id: str
    verdict: Literal["confirmed", "refuted", "inconclusive"]  # 由程式判定
    actual: float
    ci90: tuple[float, float]           # fold-bootstrap 信賴區間；區間跨門檻就判 inconclusive

class Hypothesis(TypedDict):
    hypothesis_id: str
    idea_id: int
    family: Family
    feature_tags: list[str]             # 例如 ["ema", "adx", "funding_z"]；用來算想法相似度
    mechanism: str                      # 市場機制敘述（誰虧錢給我們、為什麼）
    predictions: list[Prediction]       # 預登記：回測之前寫好，寫完就凍結
    kill_criteria: str
    prereg_hash: str                    # sha256(以上欄位的 canonical JSON)

class ChangeSpec(TypedDict):
    change_type: Literal["add_filter", "remove_filter", "change_entry", "change_exit",
                         "change_sizing", "narrow_param_space", "shift_param_space"]
    description: str
    fingerprint: str                    # sha256(change_type + 正規化描述 + 觸及元件)

class Attempt(TypedDict):               # 一次「績效變體」＝一次試驗批次
    variant_id: str                     # f"{idea_id}.{refine_iter}"
    idea_id: int
    code_hash: str                      # 正規化 AST hash
    param_space_hash: str
    behavior_fp: str                    # 驗證樣本上的訊號序列指紋
    n_configs: int                      # 本變體評估的參數組數（全部進帳本）
    is_summary_ref: str                 # artifact store ID（完整摘要放那裡）
    key_metrics: dict[str, float]       # 只放 ~8 個關鍵數字
    gate_failures: list[str]            # route_after_is_gate 的失敗原因代碼
    dsr: float
    oos_passed: bool | None             # None = 沒跑 OOS
    prediction_results: list[PredictionResult]
    change: ChangeSpec | None
    prompt_versions: dict[str, str]
    model_ids: dict[str, str]
    cost_usd: float

class Lesson(TypedDict):
    lesson_id: str
    statement: str                      # ≤ 200 字元
    scope: Literal["global", "family"]
    family: Family | None
    evidence_variants: list[str]
    support: int                        # 依此 lesson 做出的預測被證實的次數
    refute: int                         # 被推翻的次數
    status: Literal["active", "retired"]
    created_at_variant: str

class IdeaCard(TypedDict):              # 想法目錄（情節記憶的「長期」部分）
    idea_id: int
    family: Family
    feature_tags: list[str]
    best_median_sharpe: float
    closed_reason: Literal["refine_budget", "stalled", "predictions_refuted",
                           "reflection_new_idea", "impl_infeasible", "oos_fail", "winner"]
    one_line: str

class ErrorInfo(TypedDict):
    cls: Literal["E_SYNTAX", "E_POLICY", "E_LOOKAHEAD_STATIC", "E_SCHEMA", "E_RUNTIME",
                 "E_LOOKAHEAD_DYNAMIC", "E_TIMEOUT", "E_DEGENERATE", "E_DUPLICATE"]
    rule_id: str | None
    message: str                        # 已清洗、截斷到 ≤ 1500 字元
    hash: str                           # sha256(cls + rule_id + 去掉數字/位址/行號的訊息)

class Budget(TypedDict):
    usd: float
    tokens_in: int
    tokens_out: int
    backtests: int
    oos_runs: int
    debug_calls: int

# ---------- reducers ----------

def add_budget(a: Budget, b: Budget) -> Budget:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}  # type: ignore

LESSON_CAP = 12

def merge_lessons(old: list[Lesson], new: list[Lesson]) -> list[Lesson]:
    """以 lesson_id upsert；retired 的保留在 artifact store，不留在 state。"""
    by_id = {l["lesson_id"]: l for l in old}
    for l in new:
        by_id[l["lesson_id"]] = l
    active = [l for l in by_id.values() if l["status"] == "active"]
    active.sort(key=lambda l: (l["support"] - l["refute"], l["created_at_variant"]), reverse=True)
    return active[:LESSON_CAP]

# ---------- 主 State ----------

class State(TypedDict):
    # 可重現性錨點（init_run 寫入後不再改）
    run_id: str
    data_snapshot_hash: str
    engine_version: str
    split_spec: dict                    # {"is": [...], "oos": [...], "lockbox_ref": "vault://..."}
    wall_start_ts: float

    # 當前想法 / 變體
    idea_iter: int
    refine_iter: int
    hypothesis: Hypothesis | None
    pending_change: ChangeSpec | None
    pending_predictions: list[Prediction]   # 下一次回測要驗證的預測
    code_ref: str | None                # artifact ID（程式碼本體不放 state）
    code_hash: str | None
    param_space_hash: str | None
    behavior_fp: str | None
    last_good_variant: str | None       # 本想法中最後一個成功回測的變體

    # 除錯短迴圈（每變體 Overwrite 歸零；單線，不需要 reducer）
    debug_iter: int
    debug_error_hashes: list[str]
    debug_class_counts: dict[str, int]
    last_error: ErrorInfo | None
    static_ok: bool
    validate_ok: bool

    # 結果（大型資料只放 ref）
    is_summary_ref: str | None
    is_metrics: dict[str, float] | None
    dsr: float | None
    oos_passed: bool | None             # LLM 只看得到這 1 bit
    oos_result_ref: str | None          # 完整 OOS 數字：只給人與 finalize
    oos_tested_variants: list[str]

    # 累積型（reducer）
    attempts: Annotated[list[Attempt], operator.add]
    lessons: Annotated[list[Lesson], merge_lessons]
    idea_catalog: Annotated[list[IdeaCard], operator.add]
    seen_variant_hashes: Annotated[list[str], operator.add]   # code_hash+param_space_hash
    seen_behavior_fps: Annotated[list[str], operator.add]
    seen_change_fps: Annotated[list[str], operator.add]
    budget: Annotated[Budget, add_budget]

    # 反思輸出（當輪）
    reflection: dict | None             # Reflection schema 的 JSON
    # 收尾 / HITL
    final_candidate: str | None
    pbo: float | None
    lockbox_ref: str | None             # 只存 ID；不在任何 prompt 白名單裡
    final_report_ref: str | None
    human_decision: NotRequired[dict]
```

說明：
- 單線架構不會有平行寫入，所以只有「真的要累積」的欄位掛 reducer（attempts、lessons、目錄、去重集合、預算）；其餘欄位由節點直接覆寫。每條 Attempt 約 1–1.5 KB，60 個變體約 90 KB，checkpoint 大小可控。
- `lessons` 由 `merge_lessons` 控在 12 條以內，retired 的教訓搬到 artifact store。需要清空重建時用 `Overwrite(...)`。
- K 線、equity curve、交易明細、完整 IS/OOS 摘要、程式碼本體一律放 artifact store（內容定址：`sha256` → 路徑），state 只存 ID。

---

## 4. 節點清單

| # | 節點 | 類型 | 模型 | 輸入（讀 state） | 輸出（寫 state） | 副作用 |
|---|---|---|---|---|---|---|
| 1 | `init_run` | 確定性 | — | 設定檔 | run_id、data_snapshot_hash、engine_version、split_spec、預算歸零 | 建立 run 目錄與帳本 run 紀錄 |
| 2 | `strategist` | LLM | T1，effort high | idea_catalog、active lessons、已耗盡家族清單、人工 note | hypothesis（含預登記預測與 prereg_hash）、pending_predictions、idea_iter+1、refine_iter=0、last_good_variant=None | 預登記內容寫入帳本（在任何回測之前） |
| 3 | `coder` | LLM | T2，effort medium | hypothesis、pending_change、上一版程式碼（refine 時）、引擎介面規格 | code_ref、code_hash、param_space_hash；debug 狀態 Overwrite 歸零 | artifact 寫入 |
| 4 | `static_check` | 確定性 | — | code_ref | static_ok、last_error、debug_error_hashes（附加）、debug_class_counts | 無 |
| 5 | `sandbox_validate` | 確定性（容器） | — | code_ref、IS 驗證樣本（約 3 個月） | validate_ok、behavior_fp、last_error | 起拋棄式容器 |
| 6 | `debugger` | LLM | E_SYNTAX 用 T3；其餘 T2，effort medium | last_error、code_ref、hypothesis 的一行摘要（**不給績效**） | 新的 code_ref／code_hash、debug_iter+1、budget.debug_calls+1 | artifact 寫入 |
| 7 | `abandon_variant` | 確定性 | — | last_error、debug 歷史 | 一條確定性 lesson（「以 X 方式實作 Y 失敗：類別 Z」）；變體不算試驗 | 無 |
| 8 | `backtest_is` | 確定性（容器） | — | code_ref、PARAM_SPACE、IS 資料 | is_summary_ref、is_metrics（約 25 個指標）、budget.backtests+1 | 容器、artifact |
| 9 | `record_trial` | 確定性 | — | is_metrics、pending_predictions、全域帳本 | dsr、Attempt（附加）、prediction_results、lessons 支持/反駁更新、seen_* | **寫全域試驗帳本**（冪等鍵 = variant_id） |
| 10 | `oos_check` | 確定性（容器） | — | 凍結後的最佳參數、OOS 資料 | oos_passed、oos_result_ref、oos_tested_variants、budget.oos_runs+1 | 帳本標記「此變體 OOS 已用」 |
| 11 | `reflect` | LLM | T1，effort high | **白名單** context（見 §6） | reflection、pending_change、pending_predictions、新 lessons（≤ 2 條） | 無 |
| 12 | `compact_memory` | 確定性 + 偶爾 T3 | T3（只在 lessons > 12 且需要合併措辭時） | lessons、reflection | lessons（合併、去重、上限 12）；想法結束時附加 IdeaCard | 無 |
| 13 | `finalize` | 確定性 | — | attempts、帳本 | final_candidate、pbo、final_report_ref | 寫報告 artifact |
| 14 | `lockbox_eval` | 確定性（容器） | — | final_candidate、lockbox（另一組掛載/憑證） | lockbox_ref | 帳本標記「lockbox 已燒」 |
| 15 | `human_review` | interrupt | 人 | final_report_ref、lockbox_ref | human_decision | **無**（interrupt 前無副作用） |

Router（全部是純函式，不呼叫 LLM）：`route_after_static`、`route_after_validate`、`route_after_is_gate`、`route_after_oos`、`route_after_reflect`、`route_after_abandon`、`route_after_human`。

---

## 5. Routing functions（達標判定寫死在程式）

```python
import time

# ===== 寫死的門檻（改門檻 = 改程式 + code review + 引擎版本號遞增） =====
IS_GATE = dict(
    MIN_TRADES=150,              # walk-forward 測試段合計（OOF）
    MIN_TRADES_PER_FOLD=15,
    MIN_MEDIAN_FOLD_SHARPE=1.0,  # 淨成本後、年化 365 天
    MIN_POS_FOLD_FRAC=4 / 6,
    MIN_WORST_FOLD_SHARPE=-0.5,
    MAX_DD=0.25,
    MIN_SHARPE_2X_COST=0.6,      # 手續費 + 滑價加倍的壓力測試
    MIN_PLATEAU_RATIO=0.7,       # 最佳參數鄰域的 median Sharpe / 最佳 Sharpe
    MAX_TOP5_TRADE_PNL_SHARE=0.5,
    MIN_DSR=0.95,                # N = 全域帳本在此資料快照上的試驗數
)
OOS_GATE = dict(
    MIN_SHARPE=0.5,
    MIN_RATIO_TO_IS=0.4,         # OOS Sharpe / IS median fold Sharpe
    MIN_TRADES=40,
    MAX_DD_RATIO_TO_IS=1.5,
)
LIMITS = dict(
    DEBUG_MAX=4, SAME_ERR_MAX=2, DEBUG_CALLS_RUN_MAX=40,
    REFINE_MAX=5, IDEA_MAX=8, OOS_MAX=6, BACKTEST_MAX=60,
    USD_MAX=15.0, TOKENS_MAX=3_000_000, WALL_MAX_S=6 * 3600,
    STALL_ROUNDS=2, STALL_EPS=0.05, REFUTED_STREAK_MAX=2,
)
CLASS_CAP = {  # 每變體各錯誤類別最多送 debugger 幾次
    "E_SYNTAX": 2, "E_POLICY": 2, "E_LOOKAHEAD_STATIC": 2, "E_SCHEMA": 3,
    "E_RUNTIME": 3, "E_LOOKAHEAD_DYNAMIC": 2, "E_TIMEOUT": 2,
    "E_DEGENERATE": 1, "E_DUPLICATE": 1,
}


def budget_exceeded(s: State) -> str | None:
    b = s["budget"]
    if b.get("usd", 0) >= LIMITS["USD_MAX"]:                         return "usd"
    if b.get("tokens_in", 0) + b.get("tokens_out", 0) >= LIMITS["TOKENS_MAX"]: return "tokens"
    if time.time() - s["wall_start_ts"] >= LIMITS["WALL_MAX_S"]:     return "wall_clock"
    if b.get("backtests", 0) >= LIMITS["BACKTEST_MAX"]:              return "backtests"
    if b.get("debug_calls", 0) >= LIMITS["DEBUG_CALLS_RUN_MAX"]:     return "debug_calls"
    return None


def is_gate_failures(m: dict[str, float], dsr: float) -> list[str]:
    """純函式：回傳失敗原因代碼；空 list = 通過。record_trial 也呼叫它來記錄原因。"""
    g, f = IS_GATE, []
    if m["n_trades"] < g["MIN_TRADES"]:                          f.append("few_trades")
    if m["min_fold_trades"] < g["MIN_TRADES_PER_FOLD"]:          f.append("few_trades_fold")
    if m["median_fold_sharpe"] < g["MIN_MEDIAN_FOLD_SHARPE"]:    f.append("low_sharpe")
    if m["pos_fold_frac"] < g["MIN_POS_FOLD_FRAC"]:              f.append("inconsistent_folds")
    if m["worst_fold_sharpe"] < g["MIN_WORST_FOLD_SHARPE"]:      f.append("bad_worst_fold")
    if m["max_dd"] > g["MAX_DD"]:                                f.append("deep_dd")
    if m["sharpe_2x_cost"] < g["MIN_SHARPE_2X_COST"]:            f.append("cost_fragile")
    if m["plateau_ratio"] < g["MIN_PLATEAU_RATIO"]:              f.append("param_fragile")
    if m["top5_trade_pnl_share"] > g["MAX_TOP5_TRADE_PNL_SHARE"]: f.append("outlier_driven")
    if dsr < g["MIN_DSR"]:                                       f.append("dsr_low")
    return f


def oos_passed(oos: dict[str, float], is_m: dict[str, float]) -> bool:
    g = OOS_GATE
    return (oos["sharpe"] >= g["MIN_SHARPE"]
            and oos["sharpe"] >= g["MIN_RATIO_TO_IS"] * is_m["median_fold_sharpe"]
            and oos["n_trades"] >= g["MIN_TRADES"]
            and oos["max_dd"] <= g["MAX_DD_RATIO_TO_IS"] * max(is_m["max_dd"], 0.05))
    # 注意：oos_check 節點呼叫這個函式並把布林值寫進 state；
    # route_after_oos 只讀布林值，所以 OOS 數字不會流到 router 以外的地方。


# ---------------- 除錯短迴圈 ----------------
NON_DEBUGGABLE_AFTER_REPEAT = {"E_POLICY", "E_LOOKAHEAD_DYNAMIC"}  # 同規則重犯 → 直接放棄

def _debug_or_abandon(s: State) -> str:
    e = s["last_error"]
    assert e is not None
    same_hash = s["debug_error_hashes"].count(e["hash"])      # 含本次
    cls_count = s["debug_class_counts"].get(e["cls"], 0)       # 含本次
    if s["debug_iter"] >= LIMITS["DEBUG_MAX"]:                 return "abandon_variant"
    if same_hash >= LIMITS["SAME_ERR_MAX"]:                    return "abandon_variant"  # 原地打轉
    if cls_count > CLASS_CAP[e["cls"]]:                        return "abandon_variant"
    if e["cls"] in NON_DEBUGGABLE_AFTER_REPEAT and same_hash > 1: return "abandon_variant"
    return "debugger"

def route_after_static(s: State) -> str:
    if budget_exceeded(s):  return "finalize"
    return "sandbox_validate" if s["static_ok"] else _debug_or_abandon(s)

def route_after_validate(s: State) -> str:
    if budget_exceeded(s):  return "finalize"
    return "backtest_is" if s["validate_ok"] else _debug_or_abandon(s)


# ---------------- 績效閘門 ----------------
def route_after_is_gate(s: State) -> str:
    if budget_exceeded(s):  return "finalize"
    fails = is_gate_failures(s["is_metrics"], s["dsr"])
    vid = s["attempts"][-1]["variant_id"]
    if (not fails
            and vid not in s["oos_tested_variants"]                  # 每變體最多 1 次 OOS
            and s["budget"].get("oos_runs", 0) < LIMITS["OOS_MAX"]):  # 每 run 的 OOS 次數上限
        return "oos_check"
    return "reflect"

def route_after_oos(s: State) -> str:
    # 第一個通過 OOS 的變體直接收尾：不再繼續用 OOS 當篩子（降低 OOS 被「用舊」）
    return "finalize" if s["oos_passed"] else "reflect"


# ---------------- 改良 vs 重新發想（程式可以推翻 LLM 的建議） ----------------
def _idea_attempts(s: State) -> list[Attempt]:
    cur = s["hypothesis"]["idea_id"]
    return [a for a in s["attempts"] if a["idea_id"] == cur]

def _stalled(s: State) -> bool:
    xs = [a["key_metrics"]["median_fold_sharpe"] for a in _idea_attempts(s)]
    k = LIMITS["STALL_ROUNDS"]
    if len(xs) < k + 1:
        return False
    return all(abs(xs[-i] - xs[-i - 1]) < LIMITS["STALL_EPS"] for i in range(1, k + 1))

def _refuted_streak(s: State) -> int:
    n = 0
    for a in reversed(_idea_attempts(s)):
        vs = [r["verdict"] for r in a["prediction_results"]]
        if vs and all(v == "refuted" for v in vs):
            n += 1
        else:
            break
    return n

def _new_idea_or_finish(s: State) -> str:
    return "finalize" if s["idea_iter"] >= LIMITS["IDEA_MAX"] else "strategist"

def route_after_reflect(s: State) -> str:
    if budget_exceeded(s):                                    return "finalize"
    r = s["reflection"] or {}
    if s["refine_iter"] >= LIMITS["REFINE_MAX"]:              return _new_idea_or_finish(s)
    if _stalled(s):                                           return _new_idea_or_finish(s)
    if _refuted_streak(s) >= LIMITS["REFUTED_STREAK_MAX"]:    return _new_idea_or_finish(s)
    if s["attempts"][-1]["oos_passed"] is False:              return _new_idea_or_finish(s)  # OOS 失敗不做微調
    if r.get("next_action") == "new_idea" or r.get("change") is None:
        return _new_idea_or_finish(s)
    return "coder"

def route_after_abandon(s: State) -> str:
    if budget_exceeded(s):                                    return "finalize"
    # 變體寫不出來：若本想法有「最後一個可用版本」且改良額度還有，就讓 coder 從那版換個做法
    if s["last_good_variant"] and s["refine_iter"] < LIMITS["REFINE_MAX"]:
        return "coder"
    return _new_idea_or_finish(s)

def route_after_human(s: State) -> str:
    d = s["human_decision"]["decision"]
    if d == "continue" and s["idea_iter"] < LIMITS["IDEA_MAX"] + s["human_decision"].get("extra_ideas", 0):
        return "strategist"
    return "__end__"
```

關鍵點：
- `reflection.next_action` 只是**建議**，決定權在 `route_after_reflect`。程式碰到停滯、預測連錯、OOS 失敗、額度用完時，一律強制換想法。
- 改門檻就是改程式碼，要走 code review，門檻數值會進 `engine_version` 的 hash。prompt 裡**不揭露精確門檻**，只給質性目標，降低 LLM「對著門檻調參」的誘因。

---

## 6. 反思機制細節

### 6.1 反思的輸入（白名單 context builder）

`reflect` 的 prompt 由 `build_reflect_context(state, store)` 產生，只能讀白名單欄位。這是在程式層做資訊隔離：

| 給 | 內容 | 形式 |
|---|---|---|
| ✅ 預登記假說 | mechanism、當初的預測、kill criteria | 原文（凍結版，附 prereg_hash） |
| ✅ 預測驗證結果 | 每條預測的 verdict / actual / 90% CI | **由 record_trial 用程式算好**，LLM 不能改 |
| ✅ IS walk-forward 摘要 | fold 表（每 fold 的 Sharpe、報酬、DD、交易數）、regime 切片（高/中/低波動三分位、趨勢/盤整）、多空分拆、成本敏感度（0x/1x/2x）、資金費率貢獻、曝險比例、參數鄰域 plateau | JSON，約 1.5–2.5k token |
| ✅ 交易「統計」 | 交易數、勝率、平均盈虧比、持倉時間分位數、top-5 交易 PnL 佔比、連虧分布、進場時段分布 | 彙總統計，**不給逐筆明細** |
| ✅ 與上一變體的差異 | ChangeSpec + 程式碼 diff（截到 ≤ 80 行）+ 指標差值表 | 文字 |
| ✅ 閘門失敗原因代碼 | 例如 `["cost_fragile", "dsr_low"]` | 代碼（不附門檻數值） |
| ✅ 記憶 | active lessons（≤ 12）、本想法先前的 reflection 摘要（每條 ≤ 3 行） | 文字 |
| ⚠️ OOS | 只給 `oos_passed: false` **1 bit**，不給任何數字 | 布林 |
| ❌ lockbox | 完全不給；prompt builder 沒有讀取路徑，另外用 canary 字串掃描所有已記錄 prompt | — |
| ❌ 逐筆交易、equity 曲線 | 不給：token 太多，也最容易被拿來編故事（挑幾筆交易套敘事） | — |

「只給 IS 摘要還是給交易明細統計？」→ **兩者都給，但都是彙總層級**。逐筆明細會誘發「挑案例解釋」，統計層級才對得上可證偽預測的粒度。

### 6.2 反思輸出 schema（with_structured_output / strict tool）

```python
from pydantic import BaseModel, Field, conlist
from typing import Literal

class Finding(BaseModel):
    claim: str = Field(max_length=300)
    evidence_keys: conlist(str, min_length=1, max_length=4)   # 必須是摘要 JSON 裡存在的 key path
    kind: Literal["observed", "conjecture"]                    # 觀察到的事實 vs 推測，必須標清楚

class LessonDraft(BaseModel):
    statement: str = Field(max_length=200)
    scope: Literal["global", "family"]
    based_on_variants: conlist(str, min_length=1)

class PredictionDraft(BaseModel):
    metric: MetricKey
    comparator: Literal["gt", "lt", "between", "delta_gt", "delta_lt"]
    value: float
    value_hi: float | None = None
    rationale: str = Field(max_length=200)
    lesson_ids: list[str] = []

class ChangeDraft(BaseModel):
    change_type: Literal["add_filter", "remove_filter", "change_entry", "change_exit",
                         "change_sizing", "narrow_param_space", "shift_param_space"]
    description: str = Field(max_length=400)
    touched_components: conlist(str, min_length=1, max_length=2)

class Reflection(BaseModel):
    findings: conlist(Finding, min_length=1, max_length=5)
    failure_mode: Literal["no_edge", "cost_dominated", "regime_dependent", "overtrading",
                          "undertrading", "param_fragile", "outlier_driven",
                          "long_short_asymmetry", "oos_decay", "other"]
    prediction_review: list[str]                 # 對程式判定結果的解讀（不能改 verdict）
    lessons: conlist(LessonDraft, max_length=2)
    next_action: Literal["refine", "new_idea"]
    change: ChangeDraft | None                   # refine 時必填，且只能有 1 個改動
    predictions: conlist(PredictionDraft, max_length=2)   # refine 時至少 1 條
```

### 6.3 防止事後編故事（post-hoc rationalization）

1. **假說預登記**：strategist 產生的 mechanism + predictions 在任何回測之前就算好 `prereg_hash`、寫進帳本，之後不可變更。reflect 看到的是凍結版，無法「回頭改說法」。
2. **預測由程式驗證**：`record_trial` 用 fold-bootstrap（1000 次重抽 fold 測試段）算指標的 90% CI。CI 完全在門檻有利側 → `confirmed`，完全在不利側 → `refuted`，跨門檻 → `inconclusive`。LLM 只能在 `prediction_review` 裡解讀，改不了 verdict。
3. **證據必須可查**：`Finding.evidence_keys` 由 validator 核對是否真的存在於摘要 JSON。數值由程式回填到 log，LLM 不能自己寫數字，杜絕「幻覺數字」。核對失敗 → 重問 1 次，再失敗 → 視為 `new_idea`。
4. **observed 和 conjecture 分開**：只有 `observed` 的 finding 可以被 lesson 引用為依據。conjecture 必須轉成可證偽預測才能進記憶。
5. **可證偽且有風險的預測**：refine 時至少要有 1 條預測滿足最小效果量（例如 `delta_gt` 的 Sharpe 變化 ≥ 0.15、交易數變化 ≥ 20%、regime 切片的正負號翻轉），避免「Sharpe 不會變差」這種稀釋過的空洞預測。validator 用程式檢查。
6. **Lesson 計分與淘汰**：lesson 被預測引用後，預測 confirmed → `support+1`，refuted → `refute+1`。`refute ≥ 2 且 refute > support` → retired。每條 lesson 都要能被下一輪證據打臉，這是 Reflexion 原版沒有、但量化場景必須補上的一環（原論文的評分是二元的 heuristic/測試結果，無法校準反思本身）。
7. **反思校準分數**：全域追蹤 reflection 預測命中率。連續 5 條以上預測中命中率 < 30% → 在報告標記「反思 ≈ 雜訊」，route 端則更早換想法（`REFUTED_STREAK_MAX` 已涵蓋單一想法內的情況）。

### 6.4 防止原地打轉

| 機制 | 做法 | 在哪裡執行 |
|---|---|---|
| 正規化 code hash | `ast.parse` → 去 docstring/註解 → `ast.dump(annotate_fields=True, include_attributes=False)` → sha256；再加上 canonical PARAM_SPACE hash | static_check（`E_DUPLICATE`） |
| 行為指紋 | 在固定驗證樣本上跑出的訊號序列（round 到 1e-6）→ sha256，抓「改變數名的同一策略」 | sandbox_validate（`E_DUPLICATE`） |
| 改動指紋 | `ChangeSpec.fingerprint` 在同一想法中已出現 → validator 拒絕，要求換一個改動 | reflect validator |
| 單一改動規則 | 每輪只能有 1 個 ChangeSpec、觸及 ≤ 2 個元件；coder 產出的 AST diff 超過比例（> 40% 節點）→ 警告並記錄 | reflect validator / coder 後置檢查 |
| 停滯偵測 | 連 2 輪 |Δ median Sharpe| < 0.05 → 強制換想法 | route_after_reflect |
| 預測連錯 | 連 2 輪預測全被推翻 → 強制換想法 | route_after_reflect |
| 想法目錄去重 | 新假說的 `(family, feature_tags)` 與既有 IdeaCard 的 Jaccard ≥ 0.6 → 拒絕；某家族 ≥ 2 個想法都沒過 IS → 列為「已耗盡」，strategist 不得再選 | strategist validator |

### 6.5 Reflexion 記憶的兩層

- **短期（本想法內）**：本想法先前各輪 reflection 的 3 行摘要＋預測驗證結果。想法結束時清掉，改寫成 IdeaCard。
- **長期（跨想法）**：active lessons（≤ 12 條，帶 support/refute 分數）＋ IdeaCard 目錄（每張 1 行）。strategist 與 reflect 都會注入。
- Token 控制：lessons 約 1.2k token，IdeaCard 8 張約 0.6k token；總注入量固定在 2k token 以內，不隨迭代數線性增長。

---

## 7. 除錯機制細節

### 7.1 錯誤分類與處理

| 類別 | 偵測點 | 給 debugger 的內容 | 模型 | 每變體上限 | 放棄條件／特殊處理 |
|---|---|---|---|---|---|
| `E_SYNTAX` | `ast.parse` | 錯誤行 ±5 行、訊息 | T3 | 2 | 超過上限 → abandon |
| `E_POLICY` | AST：import 白名單外、`eval/exec/open/__import__/getattr` 動態呼叫、網路/檔案 IO、`os`/`sys` | 規則 ID + 違規節點 | T2 | 2 | **同規則重犯 → 立即 abandon**（代表模型堅持要用） |
| `E_LOOKAHEAD_STATIC` | AST 規則：`shift(-k)`、`rolling(center=True)`、`iloc[i+1]`、`bfill`、對整段序列取 `.max()/.mean()/.std()` 做正規化、`resample` 沒設 `closed/label='left'` | 規則 ID + 違規片段 + 正確寫法範例 | T2 | 2 | 超過 → abandon |
| `E_SCHEMA` | sandbox：回傳型別不是 `pd.Series`、index 未對齊、值不在 [-1,1]、NaN 比例 > 5%（暖機段除外）、PARAM_SPACE 格式錯誤或組數 > 48 | 期望 vs 實際 | T2 | 3 | 超過 → abandon |
| `E_RUNTIME` | sandbox 例外 | traceback 最後 15 行（清洗掉路徑、位址） | T2 | 3 | 同 hash 2 次 → abandon |
| `E_LOOKAHEAD_DYNAMIC` | 截斷不變性：隨機抽 20 個 t，比較 `signals(df[:t]).iloc[-1]` 與 `signals(df).iloc[t-1]`（容差 1e-9） | 第一個失敗的 t、兩個值、當時使用的欄位 | T2 | 2 | 同 hash 重犯 → abandon，並寫 lesson「此想法的某元件本質上需要未來資訊」 |
| `E_TIMEOUT` / OOM | 容器 60s / 2GB 上限 | 耗時、提示「改成向量化、避免逐列 loop」 | T2 | 2 | 超過 → abandon |
| `E_DEGENERATE` | 訊號恆定、驗證樣本上交易 < 5、曝險 < 1% | 統計值 | T2 | 1 | 只給一次機會（常是門檻單位寫錯）；再犯 → abandon |
| `E_DUPLICATE` | code hash、行為指紋與既有變體相同 | 對應的舊 variant_id + 本輪應實作的 ChangeSpec | T2 | 1 | 再犯 → abandon |
| `E_INFRA` | docker/容器層失敗 | **不給 LLM** | — | — | 節點 `RetryPolicy(max_attempts=2)`；不計入 debug_iter；run 內連續 3 次 → finalize 並通知人 |
| LLM 結構化輸出失敗 | 節點內 | — | — | — | 節點內重問 ≤ 2 次，不計入 debug 迴圈 |

### 7.2 除錯守則

- **不經 Evaluator、不看績效**：debugger 的輸入沒有任何指標，只修「能不能跑、合不合規」，防止它順手調參。
- **禁止改語意**：debugger 的輸出如果改了 PARAM_SPACE hash，直接拒絕、記一次迭代；AST diff 的節點比例超過 30% → 拒絕，並要求「最小修改」。
- **Error hash**：`sha256(cls + rule_id + normalize(message))`，其中 normalize 會去掉數字、記憶體位址、行號、檔案路徑，讓「同一個錯只是行號不同」能被抓到。
- **不算試驗**：只有 `backtest_is` 真的跑起來才寫帳本，修語法不會膨脹 DSR 的 N。
- **放棄之後**：`abandon_variant` 寫一條確定性 lesson（不呼叫 LLM）。如果本想法有 `last_good_variant`，coder 會以那版為基礎、依同一個 ChangeSpec 換一種寫法重試一次（算一次 refine）；否則換想法。
- **run 層級保險**：`DEBUG_CALLS_RUN_MAX=40`。超過代表 coder prompt 或引擎介面文件有系統性問題，繼續燒錢沒有意義，直接 finalize 並在報告標紅。

---

## 8. 過擬合控制如何嵌入圖中

### 8.1 資料切分（init_run 固定，寫進 split_spec 與 data_snapshot_hash）

以 2026-09 為例、BTCUSDT 永續 1h：
- **IS**：2020-01-01 ~ 2024-06-30。迴圈內只有這段資料，而且只以 walk-forward 方式使用。
- **OOS**：2024-07-01 ~ 2025-06-30。每變體最多 1 次、每 run 最多 6 次，LLM 只看得到 1 bit。
- **Lockbox**：2025-07-01 ~ 2026-06-30。存放在另一個 volume，只有 `lockbox_eval` 的容器 profile 有讀取權限；每個資料快照只用 1 次。
- 更新的資料（2026-07 之後）留給 paper trading 當真正的前向測試。

### 8.2 IS 內：nested walk-forward + purge/embargo（backtest_is）

- 滾動 6 個 fold：訓練 12 個月、測試 3 個月。
- **Purge** = 策略最大 lookback bar 數（由 PARAM_SPACE 上界推得）；**embargo** = 24 根 bar。
- 每個 fold 在訓練段從 PARAM_SPACE（≤ 48 組，固定 seed 的 grid/Sobol）挑參數，套到測試段，最後串成 OOF 報酬。**所有報告給閘門和反思的 IS 指標都來自 OOF 段**，不是全樣本擬合。
- Plateau：最佳參數在 PARAM_SPACE 格點上的鄰居 median Sharpe / 最佳 Sharpe。
- 引擎負責 shift 1 根、t+1 開盤成交、taker 手續費、滑價模型、8h 資金費率、年化 365 天。LLM 碰不到這些。

### 8.3 全域試驗帳本（record_trial）

SQLite（MVP）或 Postgres 的 `trials` 表，跨 run、以 `data_snapshot_hash` 分區：

```
trials(trial_id PK, run_id, variant_id, idea_id, code_hash, param_space_hash, params_json,
       returns_ref, sharpe, n_obs, skew, kurt, is_oof BOOLEAN, prereg_hash,
       oos_used BOOLEAN, lockbox_used BOOLEAN, engine_version, created_at)
```

- 每個被評估的參數組態都是一列（一個變體最多 48 列）。修語法不寫。
- **DSR**（Bailey & López de Prado 2014）：N 取該資料快照上的所有試驗數（MVP 用原始 N，偏保守）；V[SR] 用帳本中所有試驗的 Sharpe 變異數；再代入 OOF 的 skew/kurt。v2 可以改用相關性叢集估計有效 N（ONC），放寬過度保守的問題。
- **OOS 使用鎖**：`oos_check` 先檢查 `oos_used`，已用就拋 `OOSAlreadyUsed`，程式層保證「每變體 1 次」。
- **PBO**：`finalize` 用 CSCV（S=16 區塊）對本 run 所有變體的最佳組態 OOF 報酬矩陣計算。PBO > 0.5 → 不跑 lockbox，報告標記「搜尋過程本身過擬合」。

### 8.4 資訊洩漏預算

| 資料層 | 誰看得到 | 每 run 最多洩漏給 LLM 的資訊 |
|---|---|---|
| IS OOF | router、reflect、strategist（透過 lessons） | 不限（本來就是迴圈用） |
| OOS | router、finalize、人 | ≤ 6 bits（每次 1 bit） |
| Lockbox | 只有人 | 0（prompt 白名單 + canary 掃描 + 單元測試保證） |

---

## 9. 建圖程式碼骨架

```python
import sqlite3, time, uuid
import anthropic
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy, Overwrite
from langgraph.checkpoint.sqlite import SqliteSaver     # pip install langgraph-checkpoint-sqlite
from langchain_anthropic import ChatAnthropic

# ---------- 模型分級（結構化輸出一律 with_structured_output） ----------
T1 = ChatAnthropic(model="claude-opus-5", max_tokens=16000)
T2 = ChatAnthropic(model="claude-sonnet-5", max_tokens=16000)
T3 = ChatAnthropic(model="claude-haiku-4-5", max_tokens=8000)

def llm_structured(tier, schema, messages, *, node: str, state: State):
    """統一 wrapper：per-call 預算硬檢查、記錄 prompt 版本/模型 ID/usage/成本。"""
    if budget_exceeded(state):
        raise BudgetExceeded(node)
    runnable = tier.with_structured_output(schema, include_raw=True)
    out = runnable.invoke(messages)
    usage = out["raw"].usage_metadata
    cost = price(tier.model, usage)
    log_event(state["run_id"], node, model=tier.model, prompt_version=PROMPT_VERSIONS[node],
              usage=usage, cost=cost)
    return out["parsed"], {"usd": cost, "tokens_in": usage["input_tokens"],
                           "tokens_out": usage["output_tokens"]}

# ---------- 節點示意（只列關鍵節點） ----------
def strategist(state: State) -> dict:
    ctx = build_strategist_context(state)           # 白名單：idea_catalog、lessons、已耗盡家族、人工 note
    hyp, cost = llm_structured(T1, HypothesisOut, ctx, node="strategist", state=state)
    hyp = validate_hypothesis(hyp, state)           # 家族耗盡 / Jaccard 去重 / 預測有效量；失敗在節點內重問
    rec = freeze_and_preregister(hyp, state)        # 算 prereg_hash 並寫入帳本（冪等鍵 = hash）
    return {"hypothesis": rec, "pending_predictions": rec["predictions"], "pending_change": None,
            "idea_iter": state["idea_iter"] + 1, "refine_iter": 0, "last_good_variant": None,
            "budget": cost}

def coder(state: State) -> dict:
    out, cost = llm_structured(T2, CoderOut, build_coder_context(state), node="coder", state=state)
    ref = artifact_put(out.code)
    return {"code_ref": ref, "code_hash": norm_ast_hash(out.code),
            "param_space_hash": canonical_hash(out.param_space),
            "refine_iter": state["refine_iter"] + (1 if state["pending_change"] else 0),
            "debug_iter": 0, "debug_error_hashes": [], "debug_class_counts": {},
            "last_error": None, "budget": cost}

def static_check(state: State) -> dict:
    code = artifact_get(state["code_ref"])
    err = (check_syntax(code) or check_policy_ast(code) or check_lookahead_ast(code)
           or check_param_space(code) or check_duplicate(state))
    return _with_error(state, err, ok_key="static_ok")

def sandbox_validate(state: State) -> dict:
    res = run_in_sandbox("validate", code_ref=state["code_ref"], data="is_sample",
                         limits=dict(timeout_s=60, mem="2g", cpus=1, network="none"))
    err = res.error or check_duplicate_behavior(res.behavior_fp, state)
    upd = _with_error(state, err, ok_key="validate_ok")
    upd["behavior_fp"] = res.behavior_fp
    return upd

def _with_error(state, err, ok_key):
    if err is None:
        return {ok_key: True, "last_error": None}
    counts = dict(state["debug_class_counts"]); counts[err["cls"]] = counts.get(err["cls"], 0) + 1
    return {ok_key: False, "last_error": err,
            "debug_error_hashes": state["debug_error_hashes"] + [err["hash"]],
            "debug_class_counts": counts}

def record_trial(state: State) -> dict:
    m = state["is_metrics"]
    ledger_rows = ledger.append_variant(state, idempotency_key=variant_id(state))  # 冪等：重跑不重複寫
    dsr = ledger.deflated_sharpe(snapshot=state["data_snapshot_hash"], sr=m["oof_sharpe"],
                                 n_obs=m["n_obs"], skew=m["skew"], kurt=m["kurt"])
    preds = evaluate_predictions(state["pending_predictions"], m, prev=last_attempt(state))
    lesson_updates = score_lessons(state["lessons"], preds)
    att = make_attempt(state, m, dsr, preds, gate_failures=is_gate_failures(m, dsr))
    return {"dsr": dsr, "attempts": [att], "lessons": lesson_updates,
            "seen_variant_hashes": [state["code_hash"] + state["param_space_hash"]],
            "seen_behavior_fps": [state["behavior_fp"]],
            "last_good_variant": att["variant_id"], "pending_predictions": []}

def reflect(state: State) -> dict:
    ctx = build_reflect_context(state, store)          # §6.1 白名單；lockbox 根本沒有讀取路徑
    r, cost = llm_structured(T1, Reflection, ctx, node="reflect", state=state)
    r = validate_reflection(r, state)                   # 證據 key、單一改動、改動指紋、預測效果量
    return {"reflection": r.model_dump(), "pending_change": to_change_spec(r.change),
            "pending_predictions": to_predictions(r.predictions, state),
            "lessons": to_lessons(r.lessons, state),
            "seen_change_fps": [fp(r.change)] if r.change else [], "budget": cost}

def human_review(state: State) -> dict:
    # 這個節點在 resume 時會從頭重跑，所以 interrupt 前只能做純讀取
    decision = interrupt({
        "run_id": state["run_id"],
        "report_ref": state["final_report_ref"],     # 報告（含 lockbox）已由前面的節點寫好
        "lockbox_ref": state["lockbox_ref"],
        "options": ["approve", "reject", "continue"],
    })
    return {"human_decision": decision}

# ---------- RetryPolicy：只處理「基礎設施」層級的暫時性錯誤 ----------
llm_retry = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0,
                        retry_on=(anthropic.RateLimitError, anthropic.APIConnectionError,
                                  anthropic.InternalServerError))
infra_retry = RetryPolicy(max_attempts=2, initial_interval=5.0, retry_on=SandboxInfraError)

# ---------- 建圖 ----------
g = StateGraph(State)
g.add_node("init_run", init_run)
g.add_node("strategist", strategist, retry_policy=llm_retry)
g.add_node("coder", coder, retry_policy=llm_retry)
g.add_node("static_check", static_check)
g.add_node("sandbox_validate", sandbox_validate, retry_policy=infra_retry)
g.add_node("debugger", debugger, retry_policy=llm_retry)
g.add_node("abandon_variant", abandon_variant)
g.add_node("backtest_is", backtest_is, retry_policy=infra_retry)
g.add_node("record_trial", record_trial)
g.add_node("oos_check", oos_check, retry_policy=infra_retry)
g.add_node("reflect", reflect, retry_policy=llm_retry)
g.add_node("compact_memory", compact_memory)
g.add_node("finalize", finalize)
g.add_node("lockbox_eval", lockbox_eval, retry_policy=infra_retry)
g.add_node("human_review", human_review)

g.add_edge(START, "init_run")
g.add_edge("init_run", "strategist")
g.add_edge("strategist", "coder")
g.add_edge("coder", "static_check")
g.add_conditional_edges("static_check", route_after_static,
    {"sandbox_validate": "sandbox_validate", "debugger": "debugger",
     "abandon_variant": "abandon_variant", "finalize": "finalize"})
g.add_conditional_edges("sandbox_validate", route_after_validate,
    {"backtest_is": "backtest_is", "debugger": "debugger",
     "abandon_variant": "abandon_variant", "finalize": "finalize"})
g.add_edge("debugger", "static_check")                        # 除錯短迴圈
g.add_conditional_edges("abandon_variant", route_after_abandon,
    {"coder": "coder", "strategist": "strategist", "finalize": "finalize"})
g.add_edge("backtest_is", "record_trial")
g.add_conditional_edges("record_trial", route_after_is_gate,
    {"oos_check": "oos_check", "reflect": "reflect", "finalize": "finalize"})
g.add_conditional_edges("oos_check", route_after_oos,
    {"finalize": "finalize", "reflect": "reflect"})
g.add_edge("reflect", "compact_memory")
g.add_conditional_edges("compact_memory", route_after_reflect,  # 改良 / 重新發想 / 收尾
    {"coder": "coder", "strategist": "strategist", "finalize": "finalize"})
g.add_conditional_edges("finalize", lambda s: "lockbox_eval" if s["final_candidate"] and (s["pbo"] or 0) <= 0.5 else "human_review",
    {"lockbox_eval": "lockbox_eval", "human_review": "human_review"})
g.add_edge("lockbox_eval", "human_review")
g.add_conditional_edges("human_review", route_after_human,
    {"strategist": "strategist", "__end__": END})

conn = sqlite3.connect("runs/checkpoints.sqlite", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn))       # 正式環境改 PostgresSaver

# ---------- 執行與 HITL 恢復 ----------
run_id = f"run-{uuid.uuid4().hex[:8]}"
config = {"configurable": {"thread_id": run_id}, "recursion_limit": 1500}
result = app.invoke(initial_state(run_id), config)
if "__interrupt__" in result:                          # 停在 human_review
    show_to_human(result["__interrupt__"][0].value)
    result = app.invoke(Command(resume={"decision": "approve", "note": "送 paper trading 30 天"}),
                        config)
```

### 9.1 可觀測性、可重現性、安全落點

- **每個節點**都寫一筆事件（JSONL + 選配 LangSmith trace）：`run_id, node, variant_id, prompt_version, model_id, code_hash, data_snapshot_hash, engine_version, usage, cost_usd, duration_ms, route_decision`。router 的決策與原因（例如 `stalled`、`refuted_streak`）也記錄下來。
- **重現**：`(data_snapshot_hash, engine_version, code_hash, params)` 就能決定性地重跑任一試驗；LLM 部分透過 checkpoint + 已記錄的結構化輸出回放（`app.get_state_history(config)` 可以從任一步分叉）。
- **沙盒**：每次 `docker run --rm --network none --read-only --user 65534:65534 --cap-drop ALL --security-opt no-new-privileges --memory 2g --cpus 1 --pids-limit 128 --tmpfs /tmp:size=64m`。只掛載當次需要的資料片段（IS、OOS 或 lockbox，三者用不同 volume），不注入任何環境變數或秘密。import 白名單只當前置過濾，真正的防線是容器隔離。
- **Canary**：lockbox 報告裡嵌入隨機 canary 字串；CI 測試與 run 結束時掃描所有已記錄的 prompt，只要出現就讓 run 失敗。

---

## 10. 成本估算

### 10.1 每個策略想法（平均 5 個變體 = 1 個初版 + 4 次改良）

| 呼叫 | 模型 | 次數 | input tok/次 | output tok/次（含 thinking） | 小計 USD |
|---|---|---|---|---|---|
| strategist | T1 | 1 | 8k | 4k | 0.04 + 0.10 = **0.14** |
| coder | T2 | 5 | 6k | 3k | 5 × (0.012 + 0.03) = **0.21** |
| debugger | T2（部分 T3） | ~5（每變體平均 1 次） | 5k | 2.5k | 5 × 0.035 = **0.18** |
| reflect | T1 | 5 | 7k | 2.5k | 5 × (0.035 + 0.0625) = **0.49** |
| compact_memory | T3 | ~0.5 | 4k | 1k | **≈0.005** |
| **合計** | | **≈16.5 次** | **≈100k** | **≈44k** | **≈ USD 1.0** |

- 系統 prompt 與引擎介面規格（約 3k token）使用 prompt caching，input 成本約再降 20–25%；不過成本的大頭是 output（thinking），快取的影響有限。
- 最壞情況（每變體除錯到上限、6 個變體）：約 37 次呼叫，約 USD 2.5 / 想法。
- **每個 run**（8 個想法）：典型約 130 次呼叫、0.8M input + 0.35M output token，**約 USD 8**；硬上限 USD 15。
- **確定性運算**：每變體 1 次 IS walk-forward（≤ 48 組 × 6 fold，向量化引擎在 1h K 線上約 10–40 秒）加上截斷不變性（20 次 × 小樣本，約 10 秒）。每 run 約 40 個變體，約 30–50 分鐘 CPU。**wall-clock 主要花在 LLM 延遲**（T1 高 effort 每次 30–90 秒），一個 run 估計 2–4 小時。

### 10.2 與「帳本膨脹」的權衡

每 run 約 40 變體 × 48 組 ≈ 1900 個試驗，會讓 DSR 門檻變得很嚴（N=1000 時，虛無假設下的期望最大 SR 約為 3.2σ_SR）。這是**刻意的保守**：A 會用更高的「找不到策略」機率，換取更低的假陽性。調整手段有兩個：降低 PARAM_SPACE 上限（例如 24 組），或在 v2 改用有效 N。

---

## 11. 優點、弱點、失敗模式

### 11.1 優點
1. **最簡單**：15 個節點、7 個 router、一條主線；沒有 Send 扇出、沒有 subgraph、沒有並行 reducer 衝突。新人一天內就能讀懂整張圖。
2. **最便宜、最可預測**：每個想法約 USD 1，預算檢查集中在 router 第一行，行為一目了然。
3. **反思可以被校準**：預登記 + 程式驗證預測 + lesson 計分，讓「反思有沒有用」本身變成可以量測的數字（命中率），不只是感覺。
4. **資訊隔離清楚**：OOS 只洩漏 1 bit、lockbox 0 bit，有程式與測試保證。
5. **單執行緒 trace**：checkpoint 歷史就是研究日誌，審計和除錯都最直接。

### 11.2 弱點（誠實列出）
1. **局部最佳（最大弱點）**：改良迴圈本質上是在 IS OOF 上做「一次一個改動」的爬山。第一個想法的方向會強烈影響後續，lessons 也可能把 strategist 錨定在少數家族（記憶本身變成偏見）。
2. **序列探索慢**：一次只驗證一個假說，8 個想法就是 8 段序列時間。wall-clock 2–4 小時／run，無法利用平行回測資源。
3. **樣本數太小，學不到可靠的 lesson**：每個想法只有 5 個變體，lesson 的 support/refute 計數很小（多半 0–2），統計力很弱。計分機制能擋住明顯錯誤的 lesson，但無法證明 lesson 是對的。
4. **IS 被反覆使用**：所有想法都在同一段 IS 上做 walk-forward。walk-forward 能擋「參數過擬合」，卻擋不了「研究者（LLM）過擬合」。DSR 的 N 雖然反映了這點，但 N 的定義（組態數 vs 有效獨立試驗數）本身有爭議。
5. **OOS 當篩子**：即使每變體 1 次、每 run 6 次，「第一個過 OOS 就停」仍然是一種選擇，OOS 對最終候選已經不再完全乾淨。lockbox 才是唯一乾淨的評估，而且只有一次。
6. **同一模型家族的盲點相關**：strategist 和 reflect 都是 T1，批判和提案容易共享同樣的偏誤（例如偏好趨勢類、低估資金費率成本）。
7. **沒有組合／集成**：只產出單一策略，無法從多個弱訊號組合出穩健 alpha。

### 11.3 失敗模式與對策

| 失敗模式 | 症狀 | 對策（已內建 / 建議） |
|---|---|---|
| Coder 靠 PARAM_SPACE 偷搜尋 | 參數範圍越來越寬、組數頂到上限 | 組數上限 48；每組都算試驗（推高 DSR 門檻）；`narrow/shift_param_space` 只能由 reflect 提出 |
| Lesson 中毒 | 錯誤教訓讓後續想法全部繞開某個方向 | refute 計數淘汰；global lesson 需要 ≥ 2 個想法的證據才能升級（否則留在 family scope） |
| 反思變雜訊 | 預測命中率 ≈ 隨機 | 校準分數低於 30% 就在報告標紅，更早換想法；報告中列為「此 run 的反思無效」 |
| 除錯漂移 | debugger 順手改邏輯 | 禁改 PARAM_SPACE hash、AST diff 比例上限、不給績效資訊 |
| 門檻遊戲 | 指標剛好擦過門檻 | 不在 prompt 揭露門檻；DSR + plateau + 2x 成本 + PBO 多重把關；lockbox 最終確認 |
| 永遠找不到策略 | 預算耗盡，沒有候選 | 這是**正確的失敗**；finalize 仍然產出報告（最佳 IS 候選、失敗原因分布、IdeaCard 目錄）交給人 |
| Checkpoint 回放導致副作用重複 | 帳本重複寫入 | 帳本寫入使用冪等鍵（variant_id / prereg_hash）；human_review 在 interrupt 前無副作用 |
| 基礎設施抖動 | 容器反覆失敗 | RetryPolicy 只重試 infra 例外；連 3 次 → finalize 並通知 |

---

## 12. R1–R7 自評

| 評分項 | 權重 | 自評 | 理由 |
|---|---|---|---|
| R1 樣本外有效性 / 過擬合控制 | 25% | **7** | 硬性約束全部到位：引擎固定、三層資料、每變體 1 次 OOS、OOS 只洩 1 bit、lockbox 0 bit、全域帳本 DSR、nested WF + purge/embargo、plateau、2x 成本、PBO。扣分原因：序列爬山會對同一段 IS 造成累積的研究者過擬合；「第一個過 OOS 就停」讓 OOS 帶有選擇偏誤；DSR 的 N 定義偏粗糙。 |
| R2 反思品質 | 15% | **6** | 預登記、程式驗證預測、證據 key 核對、observed/conjecture 分離、lesson 計分、改動指紋，比原版 Reflexion 嚴謹很多。但每個想法只有約 5 個樣本，lesson 的統計力弱；單一反思者沒有對立觀點；「有資訊量的改進」只能事後透過命中率量測，無法事前保證。 |
| R3 除錯效率與穩健性 | 10% | **8** | 9 類錯誤分類、每類有專屬上下文與上限、error-hash 正規化、行為指紋去重、不經 Evaluator、不計試驗、禁止改語意、infra 與 LLM 錯誤分流。沒到 9 分是因為只有單一修復嘗試路徑（沒有多候選修補、沒有測試驅動的最小重現）。 |
| R4 探索能力 | 10% | **4** | 這是 A 的結構性弱點。家族輪替、Jaccard 去重、停滯與預測連錯時強制換想法，能避免完全卡死，但本質上仍是序列、單一假說的局部搜尋，沒有族群多樣性，也沒有平行比較。 |
| R5 成本與可控性 | 10% | **8** | 每想法約 USD 1、每 run 約 USD 8（上限 15）；五種預算集中檢查；per-call 硬上限；token 注入量固定在 2k 以內。沒給更高分是因為 T1 用在反思上，仍有降級空間；wall-clock 偏長。 |
| R6 實作複雜度與可維護性 | 15% | **8** | 圖本身非常簡單（無 Send、無 subgraph），1–2 週 MVP 可行。**但前提是回測引擎、沙盒、帳本、截斷測試這些共通基礎設施已經有了**（任何候選都要做這些）。反思 validator 與預測驗證約需 2–3 天。 |
| R7 可觀測性 / 可重現性 / HITL / 安全 | 15% | **8** | 單執行緒 trace、每節點事件含 prompt 版本/模型 ID/hash/成本、checkpoint 可分叉回放、冪等帳本、獨立無副作用的 human_review、canary 防洩漏、拋棄式容器。扣分：只有終點一個 HITL，中途（例如 OOS 前）沒有人工檢查點；LLM 輸出本身無法決定性重現，只能回放。 |

**加權總分 ≈ 7.05 / 10**（7×0.25 + 6×0.15 + 8×0.10 + 4×0.10 + 8×0.10 + 8×0.15 + 8×0.15）。

**定位建議**：A 最適合當第一個上線的基線（baseline）。它最便宜、最快做出來，而且裝上了所有防過擬合的安全帶。它預期的主要失敗不是「產出假策略」，而是「找不到策略」，這在量化研究裡是比較安全的失敗方向。等基線穩定之後，再把 strategist 換成平行扇出（Send），就能補上 R4 的缺口，而圖的其他部分幾乎不用改。
