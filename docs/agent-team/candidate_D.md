# 候選架構 D：Supervisor 多代理團隊 + 多視角批判（Multi-Critic Reflection）

> 驗證狀態：本文所有 LangGraph 程式碼取自可執行骨架 `candidate_D_skeleton.py`（同目錄）。LLM、沙盒、回測以 stub 代替，已在 **langgraph 1.2.12** 上跑過兩個情境：
> (a) 正常路徑：發想 → 寫碼 → 1 次自我除錯 → IS 回測 → 3 位批判者並行 → 聚合 → gate 通過 → OOS → lockbox → `interrupt()` → `Command(resume=...)` → finalize；
> (b) 壓力路徑：gate 一直不過、Supervisor 第一次輸出非法動作 `ACCEPT_AND_SHIP`、Quant Critic 一直給假證據 → 非法動作被擋下並改用預設動作（`violation=True` 有記錄）、假證據全部被丟掉、Quant Critic 在第 10 條發現後自動停用、Risk Critic 被採用 8 次都沒兌現後自動停用、第 40 次回測時預算強制 STOP 並轉人工。
>
> 文中描述、但骨架只以註解帶過的項目：researcher 的 novelty 檢查、除錯的 AST 語意 diff（`logic_changed`）、directive 相反方向互斥驗證、NEW_HYPOTHESIS 必須引用 lesson 的驗證、resume 值的 Pydantic 驗證、真實的 OOS / lockbox 計算。這些都是純函式，可以直接加進對應節點的外殼。
>
> 從 wheel 原始碼查到、會影響設計的 API 事實：
> 1. `add_node(..., destinations=...)` **只用來畫圖，runtime 不檢查**（`graph/state.py::_control_branch` 對 `Command.goto` 不做驗證）。所以不能拿 destinations 當 guardrail，要自己檢查（見 `guarded()`）。
> 2. reducer 必須剛好是 `(a, b) -> c` 兩個參數，多一個有預設值的參數也會 `ValueError`。
> 3. `Overwrite` 在同一個 superstep 只能寫一次，否則 `InvalidUpdateError`。
> 4. `interrupt()` 恢復時整個節點會**從頭重跑**，所以 human_review 節點裡不能有副作用。
> 5. 1.2.x 的 `add_node` 另外有 `timeout=TimeoutPolicy`（只能配 async 協作取消）和 `error_handler=`。MVP 沒用到，沙盒逾時交給容器自己處理。

---

## 1. 一句話定位 + 適用情境

**定位：** 一位 LLM 研究主管（Supervisor）負責把工作交給發想、寫碼、批判三類專家。它能選什麼由程式決定（「程式列選項，LLM 挑其一」）。三位專職批判者並行審同一份回測診斷，每條意見都要附上**可由程式驗證的證據斷言**，驗證不過的直接丟掉。策略達不達標、能不能看 OOS 或 lockbox，一律由確定性閘門判定。

**適用情境**
- 策略空間比較開放（趨勢、均值回歸、資金費率、期限結構、波動率 regime……），需要「換方向」而不只是調參，也就是一個能判斷「現在該精修還是該換題」的角色有價值的時候。
- 團隊想要「分工可見」：每位專家的輸出、被不被採用、事後有沒有兌現，都能分開統計。這適合用來做 prompt / 模型的 A/B 實驗和長期治理。
- 預算中等：每個變體大約 $0.2–0.3，一次 run 約 $8–12，可以接受多批判者帶來的 +17% 成本。

**不適用**
- 只想快速把單一想法調到最好，這時線性的 reflect-debug 迴圈更便宜也更簡單。
- 要大量探索（幾百個變體），這時族群式或 MAP-Elites 類的架構更合適。D 的 Supervisor 每一步只推進一個變體，吞吐量有限。

---

## 2. ASCII 架構圖

```
                               ┌───────────────────── 預算/計數（確定性，reducer 累加） ─────────────────────┐
                               │ usd · tokens · wall_s · backtests · trials(ledger) · oos_peeks · debug_total │
                               └──────────────────────────────────────┬───────────────────────────────────────┘
                                                                      │ 每次 supervisor 決策前讀取
 START → init_run ──(固定)──► researcher ──► coder ──► static_check ──► sandbox_exec ──route_after_exec──┐
            (hash 鎖定         (LLM:Opus)    (LLM:     (AST/schema/     (拋棄式容器 +     ┌──────────────┘
             gate config)        ▲  ▲         Sonnet)   複雜度)          截斷不變性)       │ error & 未超限
                                 │  │           ▲           ▲                               ▼
                                 │  │           │           └──────────── coder_debug ◄─────┤ (LLM:Sonnet；
                                 │  │           │                        (同錯 hash×2→Opus) │  同錯重複→升級)
                                 │  │           │                                           │ 超限/同錯已升級
                                 │  │           │                                           ▼
                                 │  │           │                                   abandon_variant ──┐
                                 │  │           │                                                     │
                                 │  │           │        ok ─► backtest_is (確定性引擎，只掛 IS 資料)  │
                                 │  │           │                  │ walk-forward purged CV；參數組合 → 全域試驗帳本
                                 │  │           │                  ▼
                                 │  │           │           dispatch_critics ──Send×k（依命中率停用無效批判者）
                                 │  │           │           ┌───────┼──────────────┐
                                 │  │           │           ▼       ▼              ▼
                                 │  │           │     quant_critic risk_critic stat_auditor   ← 只吃 Send payload（盲審）
                                 │  │           │      (Sonnet)    (Haiku)      (Sonnet)        看不到敘事/彼此/OOS
                                 │  │           │           └───────┼──────────────┘
                                 │  │           │                   ▼
                                 │  │           │           aggregate（確定性）：證據斷言驗證 → 去重 → 衝突 → 預測計分 → lessons
                                 │  │           │                   │ 靜態邊（不可繞過）
                                 │  │           │                   ▼
                                 │  │           │         ╔═ acceptance_gate（確定性，HMAC token）═╗
                                 │  │           │         ║ trades·CV Sharpe·fold一致·DSR·PBO·MDD   ║
                                 │  │           │         ║ plateau·2×成本·LLM 請求的額外檢查（只增不減）║
                                 │  │           │         ╚═══════╤═══════════════════════╤══════════╝
                                 │  │           │            fail │                  pass │ & oos_peeks < 上限
                                 │  │           │                 ▼                       ▼
                                 │  │  REFINE/  │   ┌──────► SUPERVISOR ◄──── fail ── oos_eval（驗 token；每變體 1 次）
                                 │  │  RESURRECT└───┤   (LLM:Sonnet，被夾在確定性外殼裡)      │ pass
                                 │  └─NEW_HYPOTHESIS┤   allowed_actions(state) → enum 限縮     ▼
                                 │                  │   → 事後驗證 → 越權改用預設            lockbox_eval（每 thread 1 次；
                                 │                  │   只剩一個選項時不呼叫 LLM              結果 LLM 永不可見）
                                 │                  │                                         │
                                 │     ESCALATE/STOP└──────────► prepare_review ◄─────────────┘
                                 │                               （寫報告 artifact；副作用放這裡）
                                 │                                        ▼
                                 └────────── continue(consult) ── human_review: interrupt()（零副作用）
                                                                          │ approve / reject / stop
                                                                          ▼
                                                                       finalize → END
 abandon_variant ──► SUPERVISOR
 checkpointer: SqliteSaver/PostgresSaver + thread_id；recursion_limit=1000 只當最後保險
```

**圖中三條硬性規則**
1. Supervisor 的出邊只有 `coder / researcher / prepare_review` 三個。它**沒有任何路徑**可以到 `backtest_is / acceptance_gate / oos_eval / lockbox_eval / finalize`，CI 裡有 `assert_topology()` 自動驗證。
2. `oos_eval` 的唯一入邊是 `acceptance_gate`，`lockbox_eval` 的唯一入邊是 `oos_eval`；另外 runtime 還會驗 HMAC gate token（第二道保險）。
3. 自我除錯迴圈（`sandbox_exec ↔ coder_debug`）完全**不經過** Supervisor 和批判者：修語法不算試驗，也不耗費 critic token。

---

## 3. State schema

### 3.1 圖的 State（TypedDict + Annotated reducer）

```python
# ───────────────────────── 0. 常數 / 預算（由人審定、hash 進 run_meta） ─────────────────────────
LIMITS = dict(
    max_debug_per_variant=3, max_same_error_hash=2, max_debug_total=30,
    max_refine_per_hyp=4, max_ideation=6, plateau_k=3, plateau_eps=0.05,
    max_backtests=40, max_oos_peeks=3, max_usd=25.0, max_tokens=3_000_000, max_wall_s=6 * 3600,
)
GATE_SECRET = b"held-by-orchestrator-not-in-state"      # 只有 gate 節點持有（真實版放 KMS / env）

Action = Literal["REFINE", "NEW_HYPOTHESIS", "RESURRECT", "ESCALATE_HUMAN", "STOP"]
ACTION_TO_NODE: dict[str, str] = {
    "REFINE": "coder", "NEW_HYPOTHESIS": "researcher", "RESURRECT": "coder",
    "ESCALATE_HUMAN": "prepare_review", "STOP": "prepare_review",
}
SUPERVISOR_DESTS = ("coder", "researcher", "prepare_review")


# ───────────────────────── 1. reducers ─────────────────────────
def add_counts(a: dict[str, float] | None, b: dict[str, float] | None) -> dict[str, float]:
    """並行分支可安全相加的計數 / 花費 reducer。"""
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + v
    return out


LESSON_CAP, ARCHIVE_K = 12, 5


def merge_lessons(a: list[dict] | None, b: list[dict] | None) -> list[dict]:
    """lessons 以 key 去重、保留最新，並截斷到 LESSON_CAP 條（控制 token）。reducer 必須是 (a, b) -> c。"""
    by_key = {x["key"]: x for x in (a or [])}
    for x in b or []:
        by_key[x["key"]] = x
    return sorted(by_key.values(), key=lambda x: (-x["weight"], -x["ts"]))[:LESSON_CAP]


def merge_topk(a: list[dict] | None, b: list[dict] | None) -> list[dict]:
    by_id = {x["variant_id"]: x for x in (a or []) + (b or [])}
    return sorted(by_id.values(), key=lambda x: -x["score"])[:ARCHIVE_K]


# ───────────────────────── 2. State ─────────────────────────
class State(TypedDict, total=False):
    run_meta: dict                                   # 不可變：engine/data/gate-config hash、prompt 版本
    phase: str
    spend: Annotated[dict[str, float], add_counts]   # usd / tokens_in / tokens_out / wall_s
    counters: Annotated[dict[str, float], add_counts]  # debug_total / backtests / ideation / oos_peeks ...
    hyp_counters: dict[str, dict[str, int]]          # 每個 hypothesis 的 refine / no_improve（整體覆寫）
    hypothesis: dict                                 # 當前假說（含 pre-registered prediction）
    variant: dict                                    # 當前變體：code_ref / code_hash / param_space / parent
    debug: dict                                      # {attempts, error_hashes:{h:n}, last_error, escalated}
    exec_result: dict
    diagnostics: dict                                # {pack_ref, keys:[...], headline:{...}}（小）
    round_id: str
    critiques: Annotated[list[dict], operator.add]   # 每輪以 Overwrite([]) 重置
    reflection: dict
    gate: dict                                       # 確定性 gate 結果 + HMAC token
    oos: dict                                        # 只存 pass/fail + 粗分類（LLM 可見），數字存 artifact
    lockbox_ref: str                                 # LLM 永不可見（context projection 白名單排除）
    attempts: Annotated[list[dict], operator.add]
    lessons: Annotated[list[dict], merge_lessons]
    archive: Annotated[list[dict], merge_topk]
    directive: dict                                  # supervisor → 專家的指示（有長度上限）
    supervisor_log: Annotated[list[dict], operator.add]
    critic_stats: Annotated[dict[str, float], add_counts]  # 每位批判者 valid/dropped/adopted/hit
    review: dict
    events: Annotated[list[dict], operator.add]      # 可觀測性：node, model, prompt_ver, hashes, cost
```

設計重點：
- **會被並行分支同時寫入的欄位一律用可交換的 reducer**（`spend`、`counters`、`critiques`、`critic_stats`）。三位批判者在同一個 superstep 回報花費，用 `add_counts` 相加就不會互相覆蓋。
- `critiques` 每一輪由 `dispatch_critics` 用 `Overwrite([])` 清空，每筆再帶 `round_id`，聚合時只取當輪的（兩層防呆）。
- `hyp_counters` 故意**不用** reducer，由 supervisor / gate 整體覆寫，因為它是巢狀 dict，語意是「目前值」而不是「增量」。
- 大型資料（K 線、equity、交易明細、完整診斷包、OOS 與 lockbox 數字）都放在 artifact store，state 只存 `*_ref` 和少量 headline 數字。
- `lockbox_ref` 在 state 裡，但 `LLM_VISIBLE` 白名單（§5.1）沒有它，所以任何 LLM 的 prompt 都組不到它。

### 3.2 LLM 結構化輸出 schema（Pydantic；用 with_structured_output / tool use strict）

```python
from typing import Literal
from pydantic import BaseModel, Field, conlist, constr

CheckId = Literal["fee_x2", "delay_1bar", "drop_top5_trades", "subperiod_split", "param_jitter",
                  "shuffle_returns_null", "long_short_split"]          # 確定性檢查登錄表（人審）

class Prediction(BaseModel):                       # 事前登錄：變更前就要寫下預期
    metric: Literal["cv_sharpe_median", "cv_calmar", "pos_fold_ratio", "max_dd", "turnover"]
    direction: Literal["up", "down"]
    min_delta: float = Field(ge=0)
    mechanism: constr(max_length=300)              # 為什麼（只存檔，不參與計分）

class EvidenceAssertion(BaseModel):                # 可被程式驗證的證據
    ref: str                                       # 必須是診斷包 keys 之一，如 "regime.high_vol.sharpe"
    op: Literal["<", ">", "<=", ">="]
    value: float

class Proposal(BaseModel):
    target: Literal["entry", "exit", "filter", "sizing", "holding_period", "param_space", "universe_tf"]
    direction: Literal["add", "remove", "tighten", "loosen", "replace"]
    text: constr(max_length=400)
    prediction: Prediction

class Finding(BaseModel):
    category: Literal["regime", "cost", "drawdown", "tail", "exposure", "overfit",
                      "lookahead_suspect", "magic_number", "complexity"]
    claim: constr(max_length=300)
    evidence: conlist(EvidenceAssertion, min_length=1, max_length=4)
    severity: Literal[1, 2, 3]
    proposal: Proposal | None = None
    requested_checks: conlist(CheckId, max_length=3) = []

class Critique(BaseModel):
    findings: conlist(Finding, max_length=5)       # 上限 5 條，逼批判者排優先

class SupervisorDecision(BaseModel):               # action 的 enum 在 runtime 依 allowed 動態產生
    action: str
    directive_ids: conlist(str, max_length=2) = [] # 只能引用本輪已驗證的 directive
    rationale: constr(max_length=500)

class Hypothesis(BaseModel):
    hyp_id: str
    family: Literal["trend", "mean_reversion", "carry_funding", "vol_regime", "breakout", "microstructure_lite"]
    statement: constr(max_length=500)
    economic_rationale: constr(max_length=500)     # 誰在付錢給這個策略？
    prediction: Prediction
    max_free_params: int = Field(le=4)

class SignalModule(BaseModel):
    code: constr(max_length=12000)                 # 只含 generate_signals(df, params) -> pd.Series
    param_space: dict[str, list[float | int]]      # 離散網格；總組合數 ≤ 64
    changelog: constr(max_length=300)
```

---

## 4. 節點清單

| # | 節點 | 類型 | 模型等級 | 輸入（讀 state / payload） | 輸出（寫 state） | 備註 |
|---|---|---|---|---|---|---|
| 1 | `init_run` | 確定性 | — | 設定檔 | `run_meta`（engine / data / gate-config hash、prompt 版本）、歸零計數 | gate 閾值在這裡 hash 鎖定，之後不可變 |
| 2 | `supervisor` | **LLM 包在確定性外殼內** | Sonnet 級（`claude-sonnet-5`）；只剩一個選項時不呼叫 | `allowed_actions()` 的結果＋投影後的 reflection / archive / counters / lessons | `supervisor_log`、`directive`、`hyp_counters`、`counters.ideation`、`critic_stats.adopted` → `Command(goto)` | 只能從 allowed 裡挑；越權就改用預設，並記 `violation` |
| 3 | `researcher` | LLM | Opus 級（`claude-opus-5`） | lessons、archive（只有 family 和成敗，不含 OOS 數字）、directive、人工備註 | `hypothesis`（含事前預測） | 確定性 novelty 檢查：family＋條件指紋不能和帳本裡最近 K 個重複，重複最多重試 1 次 |
| 4 | `coder` | LLM | Sonnet 級 | hypothesis、directive（被採用的 finding 原文）、lessons | `variant`（code_ref / hash、param_space） | 程式碼寫進 artifact，state 只存 hash |
| 5 | `static_check` | 確定性 | — | variant | `exec_result.static_ok` | AST：禁 `shift(-k)`、`center=True`、`iloc[t+1:]`、`.bfill()`、白名單外的 import、檔案 / 網路 I/O；參數 ≤ 4、網格 ≤ 64 |
| 6 | `sandbox_exec` | 確定性 | — | variant | `exec_result`（ok、error、invariance_ok） | 拋棄式容器 `--network none --read-only --user nobody`，限 CPU / 記憶體 / 時間；只掛 smoke 資料；做截斷不變性測試 |
| 7 | `coder_debug` | LLM | Sonnet 級；同錯 hash 第 2 次時升 Opus 級 | 錯誤分類、正規化 traceback、最小重現（違反不變性的時間點 t） | 新 code、`debug`、`counters.debug_total` | 不寫進試驗帳本 |
| 8 | `abandon_variant` | 確定性 | — | variant | `attempts`（`exec_failed`） | → supervisor |
| 9 | `backtest_is` | 確定性 | — | variant | `diagnostics`（pack_ref、keys、headline）、`counters.backtests / trials`、`round_id` | 只掛 IS 資料；每個參數組合寫進全域帳本；算 DSR / PBO / plateau / regime 表 |
| 10 | `dispatch_critics` | 確定性 | — | diagnostics、critic_stats | `Send×k`、`critiques=Overwrite([])` | 自動停用「證據造假率 > 60%」或「被採用 8 次以上但兌現率 < 25%」的批判者；Stat Auditor 永遠在場 |
| 11 | `quant_critic` | LLM | Sonnet 級 | payload：診斷 keys / headline、regime 表、code diff | `critiques`、`spend` | 績效歸因：regime、多空拆分、持有期、成本侵蝕 |
| 12 | `risk_critic` | LLM | Haiku 級（`claude-haiku-4-5`） | payload：DD 路徑摘要、尾部、曝險 | 同上 | 回撤、尾部（CVaR）、曝險集中、爆倉距離；硬門檻由 gate 判定，這裡只負責提改善方案 |
| 13 | `stat_auditor` | LLM | Sonnet 級 | payload：code 全文、帳本統計（N_trials、DSR、PBO）、參數鄰域 | 同上 | 抓語意型過擬合：magic number、日期條件、條件疊床架屋、可疑的前視寫法 → 以 `requested_checks` 轉成確定性檢驗 |
| 14 | `aggregate` | 確定性 | — | critiques、diagnostics、hypothesis.prediction、上一輪 directive | `reflection`、`lessons`、`critic_stats` | 反思核心（§6） |
| 15 | `acceptance_gate` | 確定性 | — | diagnostics、reflection.requested_checks、run_meta | `gate`（含 HMAC token）、`attempts`、`archive`、`hyp_counters` → `Command(goto)` | **唯一**能通往 OOS 的地方 |
| 16 | `oos_eval` | 確定性 | — | variant、gate | `oos`（只有 pass/fail＋粗分類）、`counters.oos_peeks` | 驗 token；每個變體只跑 1 次；整個 thread 最多 3 次 |
| 17 | `lockbox_eval` | 確定性 | — | variant | `lockbox_ref` | 每個 thread 只跑 1 次；數字另存在有 ACL 的 artifact |
| 18 | `prepare_review` | 確定性 | — | 全部 | `review`（report_ref） | 有副作用（寫報告），所以和 human_review 分開 |
| 19 | `human_review` | 人 | — | review | `review.decision` → `Command(goto)` | 第一行就是 `interrupt()`，前面沒有副作用 |
| 20 | `finalize` | 確定性 | — | 全部 | 可重現 manifest | → END |

LLM 節點只有 7 個（其中 3 個是批判者），其餘 13 個都是確定性節點。

---

## 5. Routing / guardrail：Supervisor 怎麼被程式限制

### 5.1 威脅模型：LLM Supervisor 可能怎麼破壞硬性約束

| 攻擊面 | 例子 | 防線 |
|---|---|---|
| 自稱達標 | 輸出 `ACCEPT_AND_SHIP`，或在 rationale 裡寫「已達標」 | 動作集合裡**沒有**「接受」這個選項；`ACTION_TO_NODE` 是靜態映射；事後驗證 `action ∈ allowed` |
| 繞過閘門 | goto `oos_eval` / `lockbox_eval` | Supervisor 的出邊不含這些節點（`guarded()` 在 runtime 檢查，`assert_topology()` 在 CI 檢查）；`oos_eval` 還要驗 HMAC token |
| 無限迴圈 / 原地打轉 | REFINE 和 RESURRECT 來回切換 | 每個 hypothesis 的 refine 上限、plateau（連 3 次沒進步）就移除 REFINE、ideation 上限、全域回測上限 → 只剩 STOP |
| 用 OOS 回饋調參 | 看到 OOS 失敗就指示「把門檻調緊一點」 | LLM 只看得到 OOS 的 pass/fail＋粗分類；OOS 失敗的變體會被**移除 REFINE 選項**，也不能 RESURRECT |
| 夾帶指令（prompt injection） | 在 rationale 裡叫 coder 去讀 OOS 檔 | rationale 截到 500 字；OOS / lockbox 資料根本不掛進沙盒；coder 的 context 由 `project()` 白名單產生 |
| 自創改動方向 | 不採用任何批判意見，自己亂改 | `directive_ids` 只能引用本輪**已通過證據驗證**的 finding id |
| 虛報花費 | — | 花費由節點外殼從 API usage 記帳，LLM 碰不到 |

### 5.2 程式碼：可見欄位投影、guarded Command、預算檢查

```python
# ───────────────────────── 4. 共用：context projection / guarded Command ─────────────────────────
LLM_VISIBLE = {  # 每個角色可見的 state 欄位白名單（lockbox_ref、gate token、oos 數字永遠不在內）
    "supervisor": ["phase", "reflection", "archive", "hyp_counters", "counters", "spend", "oos", "lessons"],
    "researcher": ["lessons", "archive", "directive", "review"],
    "coder": ["hypothesis", "directive", "lessons"],
    "coder_debug": ["variant", "debug"],
    "critic": [],  # 批判者只吃 Send payload，不讀 state
}


def project(role: str, state: State) -> dict:
    return {k: state.get(k) for k in LLM_VISIBLE[role]}


def guarded(goto: str | list, dests: tuple[str, ...], **kw) -> Command:
    """LangGraph 的 destinations 只用於畫圖、不在 runtime 強制；因此自己檢查。"""
    targets = [goto] if isinstance(goto, str) else goto
    for t in targets:
        name = t.node if isinstance(t, Send) else t
        assert name in dests, f"illegal goto {name} (allowed {dests})"
    return Command(goto=goto, **kw)


def over_budget(state: State) -> str | None:
    s, c = state.get("spend", {}), state.get("counters", {})
    if s.get("usd", 0) >= LIMITS["max_usd"]: return "usd"
    if s.get("tokens_in", 0) + s.get("tokens_out", 0) >= LIMITS["max_tokens"]: return "tokens"
    if s.get("wall_s", 0) >= LIMITS["max_wall_s"]: return "wall_clock"
    if c.get("backtests", 0) >= LIMITS["max_backtests"]: return "backtests"
    return None
```

### 5.3 程式碼：allowed_actions（純函式，可窮舉單元測試）

```python
# ───────────────────────── 5. Guardrail：Supervisor 只能在這些選項中挑 ─────────────────────────
def allowed_actions(state: State) -> tuple[list[str], str]:
    """純函式、可單元測試。回傳 (允許動作, 預設動作)。"""
    if (why := over_budget(state)) is not None:
        return ["STOP"], "STOP"
    c = state.get("counters", {})
    hyp = state.get("hypothesis") or {}
    hc = state.get("hyp_counters", {}).get(hyp.get("hyp_id", ""), {})
    allowed: list[str] = []
    variant_dead = state.get("oos", {}).get("variant_id") == state.get("variant", {}).get("variant_id") \
        and state.get("oos", {}).get("passed") is False                     # OOS 失敗的變體不可再往 OOS 方向微調
    if hyp and not variant_dead and hc.get("refine", 0) < LIMITS["max_refine_per_hyp"] \
            and hc.get("no_improve", 0) < LIMITS["plateau_k"]:
        allowed.append("REFINE")
    if c.get("ideation", 0) < LIMITS["max_ideation"]:
        allowed.append("NEW_HYPOTHESIS")
    if any(a["hyp_id"] != hyp.get("hyp_id") and not a.get("oos_failed") for a in state.get("archive", [])):
        allowed.append("RESURRECT")
    if state.get("reflection", {}).get("open_conflicts") and not state.get("review", {}).get("consulted"):
        allowed.append("ESCALATE_HUMAN")
    if not allowed:
        return ["STOP"], "STOP"
    allowed.append("STOP")   # 只有在還有其他選項時 STOP 才需要 LLM 判斷
    default = "REFINE" if "REFINE" in allowed and hc.get("no_improve", 0) == 0 else \
        ("NEW_HYPOTHESIS" if "NEW_HYPOTHESIS" in allowed else allowed[0])
    return allowed, default
```

### 5.4 程式碼：Supervisor 節點（三層：限縮 → 事後驗證 → 靜態映射）

```python
def supervisor(state: State) -> Command:
    allowed, default = allowed_actions(state)
    used_llm, violation, out = False, False, {}
    if len(allowed) == 1:
        action, rationale, cost = allowed[0], "forced_by_guardrail", {}
    else:
        # 真實版：schema 的 action enum 動態設為 allowed（constrained decoding）
        out, cost = llm_structured("supervisor", "SupervisorDecision", {"allowed": allowed, **project("supervisor", state)})
        used_llm = True
        action, rationale = out.get("action"), out.get("rationale", "")[:500]
        if action not in allowed:                     # 事後驗證：LLM 越權 → 用確定性預設
            action, violation = default, True
    # Supervisor 只能「引用」本輪已通過證據驗證的 directive id，不能自創改動方向
    valid_ids = {f["fid"]: f["critic"] for f in state.get("reflection", {}).get("directives", [])}
    picked = [i for i in (out.get("directive_ids", []) if used_llm else []) if i in valid_ids][:2]
    if action == "REFINE" and not picked and valid_ids:
        picked = [next(iter(valid_ids))]                     # 預設採用最高嚴重度的 directive
    upd: dict[str, Any] = {
        "supervisor_log": [{"allowed": allowed, "chosen": action, "used_llm": used_llm,
                            "violation": violation, "rationale": rationale, "directive_ids": picked}],
        "spend": cost,
        "directive": {"action": action, "text": rationale[:500], "adopted": [[i, valid_ids[i]] for i in picked]},
        "critic_stats": {f"{valid_ids[i]}.adopted": 1 for i in picked},
    }
    if action == "NEW_HYPOTHESIS":
        upd["counters"] = {"ideation": 1}
    elif action in ("REFINE", "RESURRECT"):
        if action == "RESURRECT":
            best = next(a for a in state["archive"] if a["hyp_id"] != state["hypothesis"]["hyp_id"])
            upd["hypothesis"] = best["hypothesis"]
        hid = (upd.get("hypothesis") or state["hypothesis"])["hyp_id"]
        hc = {k: dict(v) for k, v in state.get("hyp_counters", {}).items()}
        hc.setdefault(hid, {"refine": 0, "no_improve": 0})["refine"] += 1
        upd["hyp_counters"] = hc
    elif action in ("STOP", "ESCALATE_HUMAN"):
        upd["phase"] = "stopped" if action == "STOP" else "consult"
    return guarded(ACTION_TO_NODE[action], SUPERVISOR_DESTS, update=upd)
```

真實版的 `llm_structured("supervisor", ...)` 會在 runtime 產生 JSON schema，並把 `action` 設成 `{"enum": allowed}`（tool use `strict: true` 或 structured output），在解碼層就把選項限死。事後驗證是給 schema 失效或 fallback 模型用的第二道防線。

**Supervisor 不做什麼：** 不判定達標、不看 OOS / lockbox 數字、不決定要不要跑 OOS、不修改預算、不直接進除錯迴圈。它唯一的權力是：「在程式允許的 2–5 個選項中挑一個，再從本輪已驗證的 directive 裡挑 ≤2 條交給專家」。

**Shadow 評估（誠實面）：** 每次決策都會同時記下確定性的預設動作（`default`）。跑滿 3 個 run 後，如果 LLM 和預設的一致率超過 85%，而且兩者分歧時 LLM 的選擇沒有帶來更好的 IS-CV 結果，就把 Supervisor 降成純規則，省成本也省不確定性。這個退路讓架構 D 在最壞情況下退化成「規則路由＋多批判者」。

---

## 6. 反思機制細節（多批判者）

### 6.1 流程

```
backtest_is ─► 診斷包（確定性、一次算好）
               keys 例：cv.sharpe_median, cv.pos_fold_ratio, regime.{trend,range,high_vol,low_vol}.sharpe,
                     side.{long,short}.sharpe, dd.max, dd.duration_days, tail.cvar95_daily,
                     cost.share_of_gross, turnover.daily, plateau_ratio, dsr, pbo, ledger.n_trials,
                     hold.median_bars, trades, sharpe_fee_x2, funding.share_of_pnl
      │
      ├─Send─► quant_critic  ┐  盲審：只拿 payload（診斷＋code diff＋檢查登錄表）；
      ├─Send─► risk_critic   ├─ 看不到 researcher 的敘事理由、看不到彼此、看不到 OOS
      └─Send─► stat_auditor  ┘  每條 finding 必須附 1–4 條 EvidenceAssertion
      │
      ▼ aggregate（確定性）
      ① 證據斷言驗證：ref 必須存在於診斷包，而且 op(value) 對實際數字成立 → 否則丟棄（記 dropped）
      ② 去重：以 (category, proposal.target) 分組，保留 severity 最高的一條
      ③ 衝突偵測：同一個 target 出現相反的 direction（add vs remove、tighten vs loosen）→ open_conflicts
      ④ 預測計分：上一輪被採用的 directive 的 Prediction 對照本輪實際 → confirmed / refuted
      ⑤ 歸因：confirmed → 提出者的 critic.hit += 1
      ⑥ lessons：由 (變更, 預測, 實際 Δ, 結論) 模板產生，不經 LLM 改寫
      ⑦ requested_checks：取聯集（上限 4 個）→ 交給 gate 執行
      ▼
    reflection = {directives(top3, 有 fid), open_conflicts, requested_checks, prediction_outcome, dropped}
```

### 6.2 程式碼：扇出、批判者、確定性聚合

```python
CRITICS = ("quant_critic", "risk_critic", "stat_auditor")


def dispatch_critics(state: State) -> Command:
    """確定性挑選要跑的批判者（預算、歷史命中率）→ Send 扇出。Overwrite 重置本輪 critiques。"""
    stats = state.get("critic_stats", {})
    def useful(c: str) -> bool:
        n_f = stats.get(f"{c}.valid", 0) + stats.get(f"{c}.dropped", 0)
        if n_f >= 10 and stats.get(f"{c}.dropped", 0) / n_f > 0.6:      # 證據常造假 → 停用
            return False
        n_a = stats.get(f"{c}.adopted", 0)
        return n_a < 8 or stats.get(f"{c}.hit", 0) / n_a >= 0.25        # 被採用後很少兌現 → 停用
    chosen = [c for c in CRITICS if c == "stat_auditor" or useful(c)]   # Stat Auditor 永遠在場
    payload = {"round_id": state["round_id"], "variant": state["variant"],
               "diag_keys": state["diagnostics"]["keys"], "headline": state["diagnostics"]["values"],
               "check_registry": ["fee_x2", "delay_1bar", "drop_top5_trades", "subperiod_split", "param_jitter"]}
    return guarded([Send(c, payload) for c in chosen], CRITICS, update={"critiques": Overwrite([])})


def make_critic(role: str):
    def critic(payload: dict) -> dict:          # 只讀 payload：看不到 researcher 的敘事、彼此意見、OOS
        out, cost = llm_structured(role, "Critique", payload)
        return {"critiques": [{"critic": role, "round_id": payload["round_id"], **f} for f in out["findings"]],
                "spend": cost}
    critic.__name__ = role
    return critic


OPS = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}


def aggregate(state: State) -> dict:
    """確定性聚合：證據斷言驗證 → 去重 → 衝突偵測 → 預測計分 → lessons。"""
    vals, rid = state["diagnostics"]["values"], state["round_id"]
    kept, dropped, stats = [], [], {}
    for f in state.get("critiques", []):
        if f["round_id"] != rid:
            continue
        ok = f["evidence"] and all(e["ref"] in vals and OPS[e["op"]](vals[e["ref"]], e["value"]) for e in f["evidence"])
        (kept if ok else dropped).append(f)
        stats[f"{f['critic']}.{'valid' if ok else 'dropped'}"] = stats.get(f"{f['critic']}.{'valid' if ok else 'dropped'}", 0) + 1
    by_target: dict[str, list[dict]] = {}
    for f in kept:
        by_target.setdefault(f["proposal"]["target"], []).append(f)
    conflicts = [t for t, fs in by_target.items() if len({x["proposal"]["direction"] for x in fs}) > 1]
    directives = [{**f, "fid": f"{rid}:{f['critic']}:{i}"}
                  for i, f in enumerate(sorted(kept, key=lambda f: -f["severity"])[:3])]
    checks = sorted({c for f in kept for c in f.get("requested_checks", [])})[:4]
    pred = (state.get("hypothesis") or {}).get("prediction", {})
    hid0 = state["hypothesis"]["hyp_id"]
    prev = next((a["metric"] for a in reversed(state.get("attempts", [])) if a.get("hyp_id") == hid0), None)
    now = vals.get("cv.sharpe_median")
    outcome = "n/a" if prev is None else ("confirmed" if (now - prev) >= pred.get("min_delta", 0) else "refuted")
    if outcome == "confirmed":                       # 事後歸因：上一輪被採用的 directive 是否兌現預測
        for _, critic in (state.get("directive") or {}).get("adopted", []):
            stats[f"{critic}.hit"] = stats.get(f"{critic}.hit", 0) + 1
    hid = state["hypothesis"]["hyp_id"]
    lesson = {"key": f"{hid}:{state['variant']['code_hash']}", "weight": 2 if outcome == "refuted" else 1,
              "ts": len(state.get("attempts", [])),
              "text": f"{hid} 變更預測 {pred.get('direction')}≥{pred.get('min_delta')}，實際 Δ={None if prev is None else round(now - prev, 3)} → {outcome}"}
    return {"reflection": {"round_id": rid, "directives": directives, "open_conflicts": conflicts,
                           "requested_checks": checks, "prediction_outcome": outcome, "dropped": len(dropped)},
            "lessons": [lesson], "critic_stats": stats}
```

### 6.3 衝突處理

1. **可由資料仲裁的衝突優先交給確定性檢查。** 例：Quant 說「成本不重要，加快進出場」，Risk 說「換手率太高」。兩條都要附證據斷言；如果有人請求 `fee_x2`，gate 會實際跑一次，由數字決定。
2. **不能仲裁的衝突**進入 `open_conflicts`，Supervisor 的 allowed 會多出 `ESCALATE_HUMAN`（每個 thread 最多用一次）。它也可以兩條都不採用，改走 NEW_HYPOTHESIS。
3. **禁止「折衷兩邊」的 prompt 行為**：Supervisor 的 `directive_ids` 上限 2，而且兩條不能是同一個 target 的相反方向（外殼驗證）。避免同時做兩個互相抵消的改動，結果分不清是誰的功勞。
4. **嚴重度 3 的 `lookahead_suspect`**：不讓 LLM 自己判定真假，而是強制加入 `delay_1bar` 和 `shuffle_returns_null` 兩項檢查。
   - 延遲 1 根後 Sharpe 崩掉超過 70%，或打亂報酬的 null 分布 p > 0.05，就讓 gate 失敗，並記錄 `lookahead_suspect_confirmed`。
   - 這個變體之後不能再 REFINE。

### 6.4 如何避免「事後編故事」

| 機制 | 作用 |
|---|---|
| **事前登錄預測**：Hypothesis 和每條 Proposal 都必須先寫 `Prediction(metric, direction, min_delta)` | 回測之後才能驗證對錯；不能事後才說「其實這個改動是為了降低回撤」 |
| **證據斷言可驗證** | 批判者不能說「高波動期表現差」卻不給數字；給了錯的數字就被丟棄，造假率還會記進 `critic_stats` |
| **盲審** | 批判者看不到 researcher 的敘事，減少錨定和附和；也看不到彼此，讓分歧成為真正的不確定性訊號 |
| **lessons 模板化** | 教訓是「H3 加 ADX>25 過濾：預測 CV Sharpe +0.1，實際 −0.05 → refuted」這種事實句，不是 LLM 寫的故事；refuted 的權重較高，優先保留（reducer 截斷時） |
| **Supervisor 不能自創方向** | 只能引用已驗證的 finding |
| **兌現率歸因** | 長期下來，常「說得好聽但不兌現」的批判者會被自動停用 |

### 6.5 多批判者真的比單一 Evaluator 有資訊增益嗎？（誠實分析）

**可能有增益的地方**
- **覆蓋率**：單一 evaluator 的 prompt 要兼顧績效、風險、統計、程式碼審查，注意力分散，常漏掉程式碼層級的問題（magic number、日期條件）。Stat Auditor 讀的是 **code 全文＋帳本統計**，這個輸入和另兩位不同，是真正正交的資訊來源。
- **分歧即不確定性**：三條獨立意見出現衝突，本身就是「這個方向資料說不清楚」的訊號，可以觸發仲裁檢查或換題。
- **可歸因**：每位批判者的有效率和兌現率可以分開量測，prompt 治理有數據可依。

**可能沒有增益的地方**
- **Risk Critic 大部分和 gate 重複**：MDD、CVaR、曝險都是確定性指標，gate 已經在判。它的邊際價值只剩「提出改善方案」（波動率目標化、停損、減碼規則）。
- **同一個基礎模型的相關性**：三位批判者如果都是同一家、同一代的模型，錯誤高度相關，「多視角」可能只是換個措辭。用不同等級（Sonnet / Haiku）和不同輸入切片可以部分緩解，無法根除。
- **Quant Critic 和 Researcher 角色重疊**：regime 歸因之後提出「加過濾器」，本質上就是小型發想。

**我的先驗：** Stat Auditor 的增益最明確；Quant 次之；Risk 最低。預期「Stat Auditor＋一位合併的 Perf / Risk 批判者」就能拿到 85–90% 的價值。

**量測計畫（MVP 第 2 週）：** 固定 5 組種子假說和相同預算，比較 `critics ∈ {single, stat+perfrisk, 三位}`：
1. 有效 finding 率（證據驗證通過率）；
2. 被採用 directive 的兌現率；
3. 每個 gate-pass 變體的成本（$/gate-pass）；
4. gate 的 fail→pass 轉換所需迭代數。

OOS 通過率樣本太少，只當參考，不拿來決策。

**決策規則：** 某位批判者的兌現率沒有比 single 基線高 10pp 以上，就從預設組態移除。runtime 的自動停用（`dispatch_critics`）是這個規則的線上版本。

**成本：** 三位批判者每個變體約 $0.09；單一 Sonnet evaluator 約 $0.05；差額 $0.04，約佔總成本 +17%（§10）。延遲方面，因為是並行，只多出最慢那位的時間（約 20–40 秒），相對於 1–3 分鐘的回測可以忽略。

---

## 7. 除錯機制細節

```python
def norm_error_hash(err: str) -> str:
    import re
    core = re.sub(r"0x[0-9a-f]+|\d+|'[^']*'", "#", err.splitlines()[-1] if err else "")
    return hashlib.sha1(core.encode()).hexdigest()[:10]


def route_after_exec(state: State) -> str:
    r, d, c = state["exec_result"], state["debug"], state.get("counters", {})
    if r.get("static_ok") and r.get("ok") and r.get("invariance_ok"):
        return "backtest_is"
    h = norm_error_hash(r.get("error") or "invariance_violation")
    if d["attempts"] >= LIMITS["max_debug_per_variant"] or c.get("debug_total", 0) >= LIMITS["max_debug_total"]:
        return "abandon_variant"
    if d["error_hashes"].get(h, 0) >= LIMITS["max_same_error_hash"] and d["escalated"]:
        return "abandon_variant"
    return "coder_debug"


def coder_debug(state: State) -> dict:
    d = dict(state["debug"])
    h = norm_error_hash(state["exec_result"].get("error") or "invariance_violation")
    eh = dict(d["error_hashes"]); eh[h] = eh.get(h, 0) + 1
    escalate = eh[h] >= LIMITS["max_same_error_hash"]            # 同錯重複 → 換強模型 + 最小重現
    out, cost = llm_structured("coder_debug", "SignalModule", {**project("coder_debug", state), "escalate": escalate})
    code_hash = hashlib.sha256((out["code"] + str(d["attempts"])).encode()).hexdigest()[:12]
    v = {**state["variant"], "code_hash": code_hash, "code_ref": f"art://code/{code_hash}"}
    return {"variant": v, "debug": {"attempts": d["attempts"] + 1, "error_hashes": eh, "escalated": escalate or d["escalated"]},
            "counters": {"debug_total": 1}, "spend": cost}      # 修語法不記入試驗帳本


def abandon_variant(state: State) -> dict:
    return {"attempts": [{"variant_id": state["variant"]["variant_id"], "status": "exec_failed"}], "phase": "decide"}
```

- **與反思完全分離**：除錯迴圈是 `static_check → sandbox_exec → route_after_exec → coder_debug → static_check`，全部用靜態或條件邊，不經 Supervisor 和批判者，也不寫進試驗帳本（需求 4、6）。
- **錯誤分層**：
  - L0 靜態（AST / schema / 複雜度）：錯誤訊息是確定性的，直接指出行號和禁用樣式。
  - L1 執行期（例外、NaN 比例 > 5%、輸出超出 [-1, 1]、index 不對齊）。
  - L2 截斷不變性違反：回報第一個違反的 t，以及 `df[:t]` 和全量資料在 t 的訊號差值，當作最小重現。
  - L3 資源（逾時、記憶體）：提示向量化或降低 rolling 視窗。
- **錯誤 hash**：取最後一行 traceback，把數字、位址、字串字面值正規化後做 SHA1。
  - 同一個 hash 第 2 次出現 → 升級到 Opus 級模型，並附最小重現。
  - 升級後還是同一個 hash → 放棄這個變體。
  - 每個變體最多 3 次、全域最多 30 次除錯。
- **修補範圍限制**：coder_debug 的輸出 schema 和 coder 一樣，但 prompt 要求「只修錯，不改邏輯」。外殼另外做一個確定性的 **AST 語意 diff 檢查**：如果 debug 版本改了條件式結構（例如新增 / 刪除比較運算子），就標成 `logic_changed`，這次執行改記為一次試驗（避免拿除錯當改良的後門）。
- **重試 vs 除錯的區分**：LLM API 暫時性錯誤（429 / 5xx）交給 `RetryPolicy(max_attempts=3)`，不算進除錯次數；沙盒基礎設施錯誤（容器啟動失敗）由沙盒客戶端重試 2 次，也不算。

---

## 8. 過擬合控制如何嵌入

| 層 | 機制 | 在 D 的哪裡 |
|---|---|---|
| 資料紀律 | 迴圈內只用 IS（walk-forward、purged K-fold，embargo = 最長持有期 × 2） | `backtest_is` 只掛 IS 資料；沙盒容器根本看不到 OOS / lockbox 檔案 |
| 多重檢定 | 全域試驗帳本記錄**每一個參數組合**（不只變體）；DSR 的 N 用帳本累計值（保守起見用原始 N，另外報告相關聚類後的 N_eff 當參考） | `backtest_is` 寫入；gate 讀 DSR ≥ 0.95 |
| 選擇偏差 | 以變體自己的參數網格做 CSCV，得到 PBO | gate：PBO ≤ 0.3 |
| 參數穩健 | 鄰域高原：最佳點 ±1 格的中位數 Sharpe ≥ 0.7 × 最佳值 | gate：`plateau_ratio` |
| 時間穩健 | 正報酬 fold 比例 ≥ 70%；子期間分割（`subperiod_split`）由 LLM 請求或預設開啟 | gate |
| 成本穩健 | 2× 手續費＋滑價後 Sharpe ≥ 0.5；資金費率佔損益比例會被報告 | gate：`cost_x2` |
| 複雜度 | 自由參數 ≤ 4、網格 ≤ 64、條件節點數上限 | `static_check`（硬性）＋ Stat Auditor（語意） |
| 語意型過擬合 | magic number、日期條件、只對某段事件有效的規則 | **Stat Auditor**（多批判者在這裡最有價值）→ `requested_checks` 轉成確定性檢驗 |
| OOS 洩漏 | 每個變體最多 1 次 OOS、每個 thread 最多 3 次；LLM 只看 pass/fail＋粗分類；OOS 失敗的變體不能 REFINE、不能 RESURRECT | `oos_eval` ＋ `allowed_actions` ＋ `LLM_VISIBLE` |
| Lockbox | 每個 thread 1 次，只在人工審閱前用；數字只進人工報告 | `lockbox_eval` ＋ `lockbox_ref` 不在任何白名單 |
| Supervisor 帶來的過擬合 | Supervisor 看得到 archive 的 IS 分數，可能一直挑 IS 最好的去 RESURRECT → 選擇偏差 | RESURRECT 也會產生新試驗、計入帳本；archive 以 DSR 而非原始 Sharpe 排序 |
| 批判者帶來的過擬合 | 批判意見多 → 變體多 → 試驗數膨脹 | 這正是 DSR 用帳本 N 的目的：試驗越多門檻越高，會自然抑制「一直聽批判者微調」的行為 |

**gate 閾值由人審定，在 `init_run` 做 hash，寫進 `run_meta.gate_cfg_hash`。** 事後比對 hash 就能確認 run 中途沒有被改過。

---

## 9. 建圖程式碼骨架（真實 LangGraph API）

### 9.1 Gate、OOS、Lockbox（確定性、有 token）

```python
def acceptance_gate(state: State) -> Command:
    """唯一能讓變體前進到 OOS 的地方。純程式，閾值來自 run_meta 鎖定的 gate config。"""
    v, d = state["variant"], state["diagnostics"]["values"]
    checks = run_checks(state["reflection"]["requested_checks"], v)   # LLM 只能「加」檢查，不能減
    rules = {
        "trades": d["trades"] >= 150, "cv_sharpe": d["cv.sharpe_median"] >= 1.0,
        "fold_consistency": d["cv.pos_fold_ratio"] >= 0.7, "dsr": d["dsr"] >= 0.95, "pbo": d["pbo"] <= 0.3,
        "max_dd": d["dd.max"] <= 0.25, "plateau": d["plateau_ratio"] >= 0.7, "cost_x2": d["sharpe_fee_x2"] >= 0.5,
        "requested_checks": all(checks.values()),
    }
    passed = all(rules.values())
    token = hmac.new(GATE_SECRET, f"{v['code_hash']}|{state['run_meta']['data_hash']}|{passed}".encode(), "sha256").hexdigest()
    hid = v["hyp_id"]
    prev = next((a for a in reversed(state.get("attempts", [])) if a.get("hyp_id") == hid), None)
    improved = prev is None or d["cv.sharpe_median"] > prev["metric"] + LIMITS["plateau_eps"]   # 首個變體 = 基準
    hc = {k: dict(x) for k, x in state.get("hyp_counters", {}).items()}
    hc.setdefault(hid, {"refine": 0, "no_improve": 0})["no_improve"] = 0 if improved else hc[hid]["no_improve"] + 1
    upd = {"gate": {"variant_id": v["variant_id"], "passed": passed, "rules": rules, "token": token},
           "hyp_counters": hc,
           "attempts": [{"variant_id": v["variant_id"], "hyp_id": hid, "metric": d["cv.sharpe_median"],
                         "gate": passed, "failed_rules": [k for k, ok in rules.items() if not ok]}],
           "archive": [{"variant_id": v["variant_id"], "hyp_id": hid, "hypothesis": state["hypothesis"],
                        "score": d["dsr"], "oos_failed": False}]}
    if passed and state.get("counters", {}).get("oos_peeks", 0) < LIMITS["max_oos_peeks"]:
        return guarded("oos_eval", ("oos_eval", "supervisor"), update=upd)
    return guarded("supervisor", ("oos_eval", "supervisor"), update=upd)


def oos_eval(state: State) -> Command:
    g, v = state["gate"], state["variant"]
    expect = hmac.new(GATE_SECRET, f"{v['code_hash']}|{state['run_meta']['data_hash']}|True".encode(), "sha256").hexdigest()
    assert g["passed"] and hmac.compare_digest(g["token"], expect), "gate bypass detected"
    assert state.get("oos", {}).get("variant_id") != v["variant_id"], "OOS already used for this variant"
    passed = True                                   # 真實版：引擎在 OOS 區間跑一次，數字寫 artifact
    upd = {"oos": {"variant_id": v["variant_id"], "passed": passed, "bucket": "ok" if passed else "severe_decay"},
           "counters": {"oos_peeks": 1}}
    if not passed:
        upd["archive"] = [{"variant_id": v["variant_id"], "hyp_id": v["hyp_id"], "hypothesis": state["hypothesis"],
                           "score": -1, "oos_failed": True}]
        return guarded("supervisor", ("lockbox_eval", "supervisor"), update=upd)
    return guarded("lockbox_eval", ("lockbox_eval", "supervisor"), update=upd)


def lockbox_eval(state: State) -> dict:
    assert not state.get("lockbox_ref"), "lockbox already consumed in this thread"
    return {"lockbox_ref": f"art://lockbox/{state['variant']['variant_id']}", "phase": "final_review"}
```

### 9.2 HITL（副作用和 interrupt 分開）

```python
def prepare_review(state: State) -> dict:
    """副作用（寫報告 artifact）放在這裡，human_review 本身零副作用。"""
    kind = {"final_review": "final", "consult": "consult"}.get(state.get("phase"), "stopped")
    return {"review": {**state.get("review", {}), "kind": kind, "report_ref": f"art://report/{uuid.uuid4().hex[:6]}"}}


def human_review(state: State) -> Command:
    rv = state["review"]
    decision = interrupt({"kind": rv["kind"], "report_ref": rv["report_ref"],
                          "options": ["approve", "reject"] if rv["kind"] == "final" else ["continue", "stop"]})
    choice = decision.get("decision")
    if rv["kind"] == "consult" and choice == "continue":
        return guarded("supervisor", ("supervisor", "finalize"),
                       update={"review": {**rv, "consulted": True, "note": decision.get("note", "")[:1000]},
                               "phase": "decide"})
    return guarded("finalize", ("supervisor", "finalize"), update={"review": {**rv, "decision": choice}})
```

恢復方式：`app.invoke(Command(resume={"decision": "approve", "note": "..."}), {"configurable": {"thread_id": ...}})`。
- 正式版的 resume 值用 Pydantic 驗證。`decision` 必須在 `options` 裡，否則再 `interrupt()` 一次。
- 可選的保守模式：`compile(interrupt_before=["oos_eval"])`，每次看 OOS 前都要人點頭。

### 9.3 建圖與拓撲不變量測試

```python
# ───────────────────────── 7. 建圖 ─────────────────────────
def build_graph(checkpointer=None):
    llm_retry = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0)
    g = StateGraph(State)
    g.add_node("init_run", init_run)
    g.add_node("supervisor", supervisor, destinations=SUPERVISOR_DESTS, retry_policy=llm_retry)
    g.add_node("researcher", researcher, retry_policy=llm_retry)
    g.add_node("coder", coder, retry_policy=llm_retry)
    g.add_node("static_check", static_check)
    g.add_node("sandbox_exec", sandbox_exec)
    g.add_node("coder_debug", coder_debug, retry_policy=llm_retry)
    g.add_node("abandon_variant", abandon_variant)
    g.add_node("backtest_is", backtest_is)
    g.add_node("dispatch_critics", dispatch_critics, destinations=CRITICS)
    for c in CRITICS:
        g.add_node(c, make_critic(c), retry_policy=llm_retry)
    g.add_node("aggregate", aggregate)
    g.add_node("acceptance_gate", acceptance_gate, destinations=("oos_eval", "supervisor"))
    g.add_node("oos_eval", oos_eval, destinations=("lockbox_eval", "supervisor"))
    g.add_node("lockbox_eval", lockbox_eval)
    g.add_node("prepare_review", prepare_review)
    g.add_node("human_review", human_review, destinations=("supervisor", "finalize"))
    g.add_node("finalize", finalize)

    g.add_edge(START, "init_run")
    g.add_edge("init_run", "researcher")          # 第一步固定發想，不問 supervisor
    g.add_edge("researcher", "coder")
    g.add_edge("coder", "static_check")
    g.add_edge("static_check", "sandbox_exec")
    g.add_conditional_edges("sandbox_exec", route_after_exec,
                            {"backtest_is": "backtest_is", "coder_debug": "coder_debug",
                             "abandon_variant": "abandon_variant"})
    g.add_edge("coder_debug", "static_check")
    g.add_edge("abandon_variant", "supervisor")
    g.add_edge("backtest_is", "dispatch_critics")
    for c in CRITICS:
        g.add_edge(c, "aggregate")                 # 同一 superstep 的 Send 全部完成後 aggregate 只跑一次
    g.add_edge("aggregate", "acceptance_gate")     # 靜態邊：gate 不可被繞過
    g.add_edge("lockbox_eval", "prepare_review")
    g.add_edge("prepare_review", "human_review")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def assert_topology(app) -> None:
    """CI 測試：Supervisor 不得直達 gate/OOS/lockbox；OOS 只能由 gate 進入。"""
    edges = {(e.source, e.target) for e in app.get_graph().edges}
    for bad in ["acceptance_gate", "oos_eval", "lockbox_eval", "finalize", "backtest_is"]:
        assert ("supervisor", bad) not in edges, bad
    assert {s for s, t in edges if t == "oos_eval"} == {"acceptance_gate"}
    assert {s for s, t in edges if t == "lockbox_eval"} == {"oos_eval"}
    assert {s for s, t in edges if t == "acceptance_gate"} == {"aggregate"}


if __name__ == "__main__":
    app = build_graph()
    assert_topology(app)
    cfg = {"configurable": {"thread_id": "demo-1"}, "recursion_limit": 1000}
    for chunk in app.stream({}, cfg, stream_mode="updates"):
        print(list(chunk.keys()))
    snap = app.get_state(cfg)
    print("next:", snap.next, "| interrupt:", snap.tasks[0].interrupts[0].value if snap.tasks else None)
    for chunk in app.stream(Command(resume={"decision": "approve"}), cfg, stream_mode="updates"):
        print(list(chunk.keys()))
    final = app.get_state(cfg).values
    print("phase:", final["phase"], "| spend:", final["spend"], "| counters:", final["counters"])
    print("supervisor_log:", final.get("supervisor_log"))
    print("reflection:", {k: final["reflection"][k] for k in ("open_conflicts", "requested_checks", "dropped")})
```

`recursion_limit` 怎麼算：每個變體最多約 17 個 superstep（coder、3×(static+sandbox+debug)、backtest、dispatch、critics、aggregate、gate、supervisor），40 個變體約 680 個，再加上 OOS / HITL 餘裕，設 1000。真正的終止條件永遠是 `allowed_actions` 的預算和計數，`recursion_limit` 只是最後一道保險。

### 9.4 正式環境替換點

- `InMemorySaver` → `PostgresSaver`（`thread_id` = `run_id`），可以用 `get_state_history()` 做 time-travel 重放。
- `llm_structured` 的實作用 Anthropic SDK 的 structured outputs 或 strict tool use；每次呼叫都記下 model ID、prompt 版本、input / output hash、usage。
- `run_sandbox` / `run_backtest_is` 呼叫外部 job runner（容器），state 只收 ref。
- 試驗帳本放在 Postgres 的 append-only 表（`trial_id, run_id, dataset_hash, code_hash, params, is_metrics, ts`），不放在 state（state 只存計數）。這樣跨 thread 也能計算 DSR 的 N。

---

## 10. 成本估算

**單價假設**（Anthropic 官方價，每百萬 token）：Opus 級 `claude-opus-5` $5 / $25；Sonnet 級 `claude-sonnet-5` $2 / $10；Haiku 級 `claude-haiku-4-5` $1 / $5。輸出 token 已含 adaptive thinking。system prompt 做 prompt caching（讀取約 0.1×），估計可再省 10–15%，下表**未**扣除。

**每個變體（一次 IS 迭代）**

| 節點 | 模型 | 平均次數 / 變體 | in tokens | out tokens | $ / 變體 |
|---|---|---|---|---|---|
| supervisor | Sonnet | 0.8（強制時跳過） | 6k | 0.6k | 0.014 |
| coder | Sonnet | 1 | 10k | 4k | 0.060 |
| coder_debug | Sonnet | 0.7 | 8k | 3k | 0.032 |
| coder_debug 升級 | Opus | 0.1 | 10k | 4k | 0.015 |
| quant_critic | Sonnet | 1 | 8k | 2k | 0.036 |
| risk_critic | Haiku | 1 | 7k | 1.5k | 0.015 |
| stat_auditor | Sonnet | 1 | 10k | 2k | 0.040 |
| researcher（攤提：每個假說約 5 個變體） | Opus | 0.2 | 12k | 5k | 0.037 |
| **合計** | | | **≈ 49k** | **≈ 13.5k** | **≈ $0.25**（有快取約 $0.21） |

- 批判階段 $0.09，佔 36%。若換成單一 Sonnet evaluator（12k / 2.5k → $0.05），每個變體省 $0.04（−17%）。
- **一次 run**（預設上限 40 次回測、6 個假說）：約 2.0M input＋0.55M output → **$8–10**。硬上限 `max_usd = 25`、`max_tokens = 3M`。
- **Wall-clock**：瓶頸在回測。15m K 線 2 年約 70k 根，5 fold × ≤64 組參數向量化，約 1–3 分鐘；加上 LLM 呼叫（批判者並行，約 30 秒），每個變體約 3–6 分鐘 → 一次 run 2–4 小時。上限 `max_wall_s = 6h`。
- **成本控制**：
  - Supervisor 只剩單一選項時不呼叫；
  - 批判者依兌現率自動停用；
  - 除錯不經批判者；
  - researcher 只在 NEW_HYPOTHESIS 時才呼叫；
  - 所有預算檢查都在 `allowed_actions` 的**第一行**（超支就只剩 STOP）。

---

## 11. 優點、弱點、失敗模式

**優點**
1. **「LLM 選、程式限」的邊界清楚**：Supervisor 的權力可以一句話講完，而且有三層強制（拓撲、enum 限縮＋事後驗證、HMAC token＋CI 拓撲測試）。符合需求 5、6。
2. **反思有證據鏈**：可驗證的證據斷言、事前登錄預測、模板化 lessons、兌現率歸因，讓「反思是否有用」變成可量測的數字，而不是觀感。
3. **Stat Auditor 在過擬合控制上有實質增益**：把語意型過擬合轉成確定性檢驗，這是單一 evaluator 常漏掉的。
4. **可觀測性和治理**：每位專家的產出、被採用率、兌現率、造假率、Supervisor 越權次數，都是現成的儀表板欄位。
5. **退化路徑平滑**：Supervisor 可以降成規則，批判者可以降到 1 位，拓撲不變。

**弱點**
1. **節點數多**（20 個），Command / Send / Overwrite 混用，新人上手成本比線性迴圈高。
2. **Supervisor 的邊際價值未經證實**：它本質上是一個 2–5 類的分類器，很可能沒有比 `default` 規則好多少（所以有 shadow 評估）。
3. **多批判者的「多樣性」可能是假的**：同代模型的錯誤高度相關；Risk Critic 大部分和 gate 重複。
4. **探索仍是單線程**：每一步只推進一個變體，跳出局部最佳只能靠 NEW_HYPOTHESIS 和 plateau 強制；沒有族群多樣性壓力。
5. **證據斷言只能驗證「數字對不對」，不能驗證「因果推論對不對」**：批判者可以引用正確的數字，但得出錯誤的機制解釋。這部分只能靠預測兌現率在事後淘汰。

**失敗模式與對策**

| 失敗模式 | 徵兆 | 對策 |
|---|---|---|
| Supervisor 偏好 REFINE（局部打轉） | 同一假說 no_improve 一直累積 | plateau_k=3 強制移除 REFINE；refine 上限 4 |
| Supervisor 一直換題（淺嘗即止） | ideation 很快耗盡、每個假說只有 1 個變體 | 預設動作偏向 REFINE；NEW_HYPOTHESIS 的 rationale 必須引用 refuted 的 lesson（外殼驗證 lesson key 存在） |
| 批判者附和 / 模板化 | 三位批判者的 findings 高度相同、有效率高但兌現率低 | 盲審；兌現率自動停用；定期檢查 critic prompt |
| 證據造假 | dropped 率高 | 自動停用＋告警；診斷包 key 以列舉方式寫進 prompt |
| DSR 門檻過高、永遠過不了 | 試驗數累積後 gate 全失敗 | 這是**正確行為**（告訴人：這個資料集上的搜尋已經過度）；由人決定要換資料或結束，不由 LLM 放寬 |
| OOS 被間接當成訓練訊號 | 同一個 family 反覆 RESURRECT、OOS 反覆失敗 | 每個 thread 最多 3 次 OOS；OOS 失敗的變體不能 REFINE / RESURRECT；報告會標記 family 的 OOS 失敗次數 |
| 並行寫入衝突 | `InvalidUpdateError` | 所有並行寫入的欄位都用可交換 reducer；`Overwrite` 只由 dispatch 在單一節點寫 |
| checkpoint 膨脹 | state 越來越大 | `lessons` / `archive` reducer 有上限；`attempts` 只存小摘要；大物件放 artifact |
| 除錯偷改邏輯 | debug 後績效大變 | AST 語意 diff → `logic_changed` 改記一次試驗 |

---

## 12. 依評分標準 R1–R7 自評

| 項目 | 權重 | 分數 | 理由（誠實） |
|---|---|---|---|
| R1 樣本外有效性 / 過擬合控制 | 25% | **7** | 硬性約束全部落實在確定性節點（DSR 用帳本 N、PBO、plateau、OOS 次數上限、lockbox 隔離、OOS 失敗不可精修）。Stat Auditor 把語意型過擬合轉成可驗證檢查，這是真增益。扣分：這些大多是「共同需求」本來就要求的，D 的獨特機制（Supervisor＋多批判者）對 R1 的淨貢獻有限；批判意見多也可能讓試驗數膨脹（雖然 DSR 會懲罰）。 |
| R2 反思品質 | 15% | **7** | 證據斷言驗證、事前預測、盲審、模板化 lessons、兌現率歸因，讓「編故事」很難存活，而且反思品質可量測。扣分：同代模型的批判者錯誤相關，多視角的增益可能小於預期；證據斷言驗證不了因果解釋。 |
| R3 除錯效率與穩健性 | 10% | **7** | 除錯和反思完全分離、錯誤分層、正規化 hash、同錯升級、最小重現、AST 語意 diff 防後門。屬於標準但紮實的設計，沒有特別創新。 |
| R4 探索能力 | 10% | **6** | Supervisor 的 NEW_HYPOTHESIS / RESURRECT、plateau 強制換題、novelty 檢查、archive。但是單線程、沒有族群或多樣性壓力，跳出局部最佳的能力中等。 |
| R5 成本與可控性 | 10% | **6** | 預算在 guardrail 第一行、單一選項時跳過 LLM、批判者自動停用，一次 run 約 $8–10 且有硬上限。但每個變體有 5–6 次 LLM 呼叫，比單一 evaluator 架構貴約 17%，在同等預算下迭代數較少。 |
| R6 實作複雜度與可維護性 | 15% | **6** | 20 個節點，Command＋Send＋Overwrite＋HMAC。骨架已經驗證能跑，純函式（`allowed_actions`、`aggregate`、gate）容易單元測試。MVP 兩週可行的前提是：第一週先做 2 位批判者＋規則版 supervisor，第二週再加 LLM supervisor 和第三位批判者做 A/B。完整版比線性架構明顯複雜。 |
| R7 可觀測性 / 可重現 / HITL / 安全 | 15% | **8** | supervisor_log（allowed / chosen / violation）、critic_stats、events、gate HMAC token、`assert_topology` CI、LLM 可見欄位白名單、run_meta hash、checkpointer time-travel、HITL 副作用分離、沙盒不掛 OOS。這是 D 最強的一項。 |
| **加權總分** | | **≈ 6.8 / 10** | 0.25×7 + 0.15×7 + 0.10×7 + 0.10×6 + 0.10×6 + 0.15×6 + 0.15×8 = 6.80 |

**一句話結論：** D 的價值主要在「治理」和「可歸因」，而不在找到更好的策略。它把 LLM 主管關在程式劃好的籠子裡，並讓每位批判者的貢獻可被量測和淘汰。建議的落地方式是：**以 D 的 guardrail、證據斷言、Stat Auditor 為可移植元件**，嫁接到較簡單的主幹上；LLM Supervisor 和第三位批判者則以 feature flag 形式，經 A/B 證明有增益後才打開。
