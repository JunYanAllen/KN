# 候選架構 B：階層式巢狀子圖（Hierarchical Nested Subgraphs）

> 版本：v1（2026-09-27）｜目標框架：langgraph 1.2.12（已下載 wheel 讀原始碼，並在 venv 內實跑驗證）
> 可執行骨架：`candidate_B_skeleton.py`（同目錄，558 行。LLM、沙盒、引擎都用 stub 取代，三層拓撲、state 傳遞、checkpointer、interrupt、error_handler、預算路徑都已實跑通過）

---

## 0. 本設計依據的 LangGraph 事實（原始碼 + 實驗查證）

| # | 事實 | 證據 | 對設計的影響 |
|---|---|---|---|
| F1 | 編譯好的子圖直接 `add_node` 時，父圖會把**整份父 state** 傳給子圖；子圖只讀取自己 `input_schema` 內的 channel | `graph/state.py` add_node：`resolved_input_schema = input_schema or inferred or self.state_schema` | 子圖用 `input_schema` 只宣告自己需要的「請求信箱」鍵 |
| F2 | 子圖回傳時，父圖**只寫回父圖也有的鍵**（`_get_updates`：`if k in output_keys`），私有鍵會被默默丟掉 | `graph/state.py` attach_node / `_get_updates` | 子圖的私有計數器、錯誤歷史不會汙染上層 |
| F3 | **陷阱**：父子共用一個帶 reducer 的鍵（例如 `Annotated[list, operator.add]`）時，子圖結束後會把「整個累積值」回吐給父圖，父圖再 reduce 一次，結果重複 | 實驗 T1：`{'log': ['p0','p0','child']}` | **禁止**跨層共用 reducer 鍵。子圖一律用 `output_schema` 只輸出「結果信箱」鍵（實驗 T2 證實可解決） |
| F4 | `compile(checkpointer=None)`（預設）：子圖沿用父圖的 checkpointer，namespace 帶 task_id，所以**每次呼叫都是全新 state** | `pregel/main.py`（`checkpointer is True` 才 `recast_checkpoint_ns`）。實驗 C：`b` 兩次都是 `[1]` | Debug 與 Refine 用預設 None，每次進入時計數器自動歸零 |
| F5 | `checkpointer=True`：同一 thread 內跨呼叫累積子圖 state；根圖不能用（`RuntimeError`） | 實驗 C：`a` 第二次變成 `[2]` | **不用 True**，否則上一個 idea 的 debug 計數會延續到下一個 |
| F6 | `checkpointer=False`：子圖不做 checkpoint，`find_subgraph_pregel` 也不把它當成子圖 | `pregel/_utils.py` | 子圖內斷線時會整段重跑，`get_state(subgraphs=True)` 也看不到它。可以當成「省寫入量」的選項 |
| F7 | 子圖內呼叫 `interrupt()`，resume 時**父層 node 從頭重新執行**，子圖內已完成的 node 不會重跑（從子圖 checkpoint 接續） | 實驗 B：`wrapper=2, pre=1` | HITL 只放主圖的 `human_review`；子圖內不放 interrupt |
| F8 | 在 node 函式內手動 `sub.invoke()` 仍會被偵測為子圖（掃描 closure 與 global），checkpoint、stream、`get_state(subgraphs=True)` 都正常 | `get_function_nonlocals`；實驗 B 看得到子圖 state | 兩種寫法在可觀測性上等價，差別在耦合方式與重跑語意（見 §3.4） |
| F9 | 新版 `add_node(..., error_handler=fn)`：handler 收到 `NodeError(node, error)`。**只有回傳 `Command(goto=...)` 才會繼續往下走**，回傳一般 dict 會讓圖停在該處，而且不報錯。`GraphBubbleUp`（interrupt）不會被攔截 | `pregel/_runner.py`；實驗 A；原型 `test_crash.py` | 子圖崩潰時轉成 `status="crashed"` 結果，再 `goto` 到 absorb node，外層不會連帶崩潰 |
| F10 | `recursion_limit` 會透過 config **繼承**到子圖（每層各自計步，但上限共用） | 實驗：主圖設 20、子圖跑 50 步時觸發 `GraphRecursionError` | 業務上限寫在計數器裡，`recursion_limit=200` 只當最後保險 |
| F11 | 子圖內 node 可用 `get_config()["configurable"]["checkpoint_ns"]` 取得 `refine:<id>\|debug:<id>\|leaf:<id>` | `nstest.py` | 資料閘道可依 namespace 前綴強制「refine/debug 只能讀 IS」 |
| F12 | `interrupt(value, response_schema=TypedDict)` 會驗證 resume 值；Python < 3.12 必須用 `typing_extensions.TypedDict` | `types.py`；原型曾因此報 `PydanticUserError` | HumanDecision 的型別由框架驗證 |
| F13 | `stream(..., subgraphs=True)` 產出 `(namespace_tuple, update)` | `nstest.py` 輸出 | 儀表板可以分層顯示 |
| F14 | `timeout=` 只對 async node 生效（sync node 無法安全取消） | add_node docstring | 沙盒逾時由容器自己的 `--stop-timeout` 負責，不依賴 LangGraph timeout |

---

## 1. 一句話定位與適用情境

**定位**：用「三層巢狀子圖 + 信箱式 I/O 契約」把三種迴圈做成三個可獨立編譯、獨立測試、獨立計預算的 StateGraph。錯誤在哪一層發生，就在那一層消化；上層只拿到一個固定 schema 的結果物件。

**適用情境**
- 團隊要把「執行錯誤修復／參數邏輯改良／重新發想」三種迴圈的計數、預算、日誌**硬性分開**，而且希望各層可以分給不同人維護、分開寫單元測試。
- 每個策略 idea 依序精煉（先求深度、再求廣度），單次 run 的預算在 USD 10 上下、耗時約 1–2 小時。
- 需要嚴格稽核：每一層的 checkpoint namespace 天然就是稽核邊界（`refine:<task>|debug:<task>`）。

**不適用**：需要同時平行探索大量 idea（族群式搜尋）時，本架構的序列外迴圈會成為瓶頸（見 §11，可用 `Send` 擴充）。

---

## 2. ASCII 架構圖

```
╔══════════════════════════════════ L3 主圖 MainGraph (checkpointer=Postgres/Sqlite, thread_id=run_id) ══════════════════════════════════╗
║                                                                                                                                          ║
║  START → init_run ──► ideate ─────► prepare_refine ──► [ refine 子圖 ] ──► absorb_refine ──┬─(candidate & oos_uses<6 & 預算OK)─► oos_eval║
║          (資料切分    (LLM·強)       (切預算 slice)      error_handler=      (spent/lessons  │                                   (1次/變體) ║
║           hash清單)      ▲           IDEA_CAP=$1.6       refine_crash       回收)          └─(其餘)──┐                            │    ║
║                          │                                                                          ▼                            ▼    ║
║                          │ ideas<6 且預算OK                                                  distill_failure ◄──(未過)──── stat_gate  ║
║                          └────────────────────────────── route_next_idea ◄─────────────────  (只留類別型   (DSR/PBO/衰退/t值隨k升高) ║
║                                                              │ ideas≥6 或預算用完 或 crashed    lesson，不含數字)      │(過)        ║
║                                                              ▼                                                          ▼             ║
║                                                            report ◄── human_review ◄───────────────────────────── lockbox_eval        ║
║                                                              │       ★interrupt()★                                   (一次性、       ║
║                                                              ▼        response_schema=HumanDecision                   結果不進LLM)    ║
║                                                             END       （前面無副作用）                                               ║
║  上限：MAX_IDEAS=6、MAX_OOS_USES=6、run 預算 {usd 12, wall 2h, backtests 20k}、recursion_limit=200（保險）                             ║
╠══════════════════════ L2 Refine 子圖 (compile()，checkpointer=None 繼承；input={refine_request}  output={refine_result}) ════════════════╣
║                                                                                                                                          ║
║  START → init → prepare_debug ──► [ debug 子圖 ] ──► absorb_debug ─┬─(ok)──► backtest_is ──► score_prediction ──► gate ──┬─(過且進步)──┐ ║
║                  ▲ (帶 change/     error_handler=      (iter+1,    │         (WF 6折+purge    (預先登記的預測     (IS閘門  │            │ ║
║                  │  allowance)     debug_crash         spent回收)  │          +內層參數選擇     是否成立)          寫死)    │            ▼ ║
║                  │                                                 │          +鄰域穩定                                   │        finalize║
║                  │                                                 ├─(gave_up 且已有 champion)──► reflect                │   ▲       │ ║
║                  │                                                 └─(budget/crashed/首版就失敗)───────────────────────────┼───┘       │ ║
║                  │                                                                                                      ▼           END║
║                  └──(valid)── verify_reflection ◄── reflect ◄── diagnose(併入 backtest) ◄──(未過/未進步，且 iter<5、           │    ║
║                     (證據數字比對      (LLM·強)                                           plateau<2、refuted<3、預算OK)─────────────┘    ║
║                      + tabu 去重)                                                                                                        ║
║  上限：max_iters=5、PLATEAU_PATIENCE=2、REFUTED_STREAK_MAX=3、每 idea 預算 ≤ $1.6 / 30 分鐘；每個 param combo 都寫入 trial ledger        ║
╠══════════════════════ L1 Debug 子圖 (compile()，input={debug_request}  output={debug_result}) ═══════════════════════════════════════╣
║                                                                                                                                          ║
║  START → coder ──► validate ──(ok)──► sandbox_run ──(ok)──► finish → END                                                                 ║
║          ▲ (LLM·中/小)  (AST/schema/    (拋棄式容器：smoke +      ▲                                                                         ║
║          │              import白名單)   截斷不變性 + 輸出契約)    │                                                                         ║
║          │                 │(fail)            │(fail)               │                                                                    ║
║          │                 └──────► triage ◄──┘                     │                                                                    ║
║          │                   (確定性：分類、錯誤hash、升級策略)       │                                                                    ║
║          └──────(可修)───────────────┘ └──(attempt≥4 / 同hash≥3 / 前視≥2 / E_INFRA / 預算)──┘                                            ║
║  上限：MAX_DEBUG_ATTEMPTS=4、每次呼叫預算 ≤ $0.40 / 15 分鐘；修語法「不算」試驗、不寫 ledger                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

 橫切服務（放在 Runtime context，不進 state）：ArtifactStore｜TrialLedger（以 trial_id 冪等）｜DataGateway（依 checkpoint_ns 限制分區）｜
                                             SandboxClient｜LLMClient（每次呼叫前先依估算成本檢查預算）｜OOS/Lockbox 一次性許可
 預算檢查點：每層 router 都檢查「本層 spent ≥ allowance」；allowance = min(本層上限, 上層剩餘)，再逐層回收 spent
```

---

## 3. 各層 State schema 與層間轉換

### 3.1 共用小型型別與 reducer

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict, NotRequired   # py<3.12 時 interrupt(response_schema=...) 需要
import operator

class Cost(TypedDict, total=False):
    usd: float; in_tok: int; out_tok: int; backtests: int; wall_s: float

def add_cost(a: Cost | None, b: Cost | None) -> Cost:        # 初始 channel 值為 {}，必須容忍缺鍵
    a, b = a or {}, b or {}
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("usd","in_tok","out_tok","backtests","wall_s")}

class Lesson(TypedDict):
    key: str                                   # 去重鍵：family|mechanism-fingerprint 或 family|oos|categories
    scope: Literal["family","mechanism","implementation","oos_category"]
    family: str
    claim: str                                 # ≤200 字元、單句
    evidence: Literal["measured_is","oos_category","infeasible"]

LESSON_CAP = 12
def merge_lessons(old, new):                   # 同 key 以新蓋舊並移到尾端，只保留最近 12 條
    seen = {l["key"]: l for l in (old or [])}
    for l in new or []:
        seen.pop(l["key"], None); seen[l["key"]] = l
    return list(seen.values())[-LESSON_CAP:]

class StrategySpec(TypedDict):
    idea_id: str; family: str; thesis: str; signal_logic: str; param_space: dict

class VariantRecord(TypedDict):               # 只放摘要數字；完整 equity/交易明細在 artifact store
    variant_id: str; code_ref: str; code_hash: str; params: dict
    wf_sharpe: float; wf_sharpe_se: float; pos_fold_frac: float; min_fold_trades: int
    max_dd: float; cost_share: float; neighbor_ratio: float; n_params: int; diag_ref: str
```

### 3.2 層間契約（信箱鍵）：每條邊界只有一個請求鍵和一個結果鍵，兩者在父圖都是**沒有 reducer 的覆寫鍵**

```python
# ---- L2 → L1 ----
class DebugRequest(TypedDict):
    variant_id: str; spec: StrategySpec; mode: Literal["generate","patch"]
    base_code_ref: str | None; change: dict | None     # change = 反思產出的 ChangeSpec
    allowance: Cost; max_attempts: int
class DebugResult(TypedDict):
    variant_id: str; status: Literal["ok","gave_up","budget","crashed"]
    code_ref: str | None; code_hash: str | None
    error_class: str | None; error_digest: str | None   # ≤300 字元；完整 traceback 只存 artifact
    attempts: int; spent: Cost

# ---- L3 → L2 ----
class RefineRequest(TypedDict):
    idea_id: str; spec: StrategySpec; allowance: Cost; max_iters: int
class RefineResult(TypedDict):
    idea_id: str; status: Literal["candidate","no_candidate","infeasible","budget","crashed"]
    champion: VariantRecord | None; pbo: float | None; n_trials: int
    lessons: list[Lesson]          # ≤3 條
    iters: int; spent: Cost
```

### 3.3 三層 State

```python
# ================= L1 Debug（私有：attempt / errors / fix_plan / spent） =================
class DebugIn(TypedDict):  debug_request: DebugRequest
class DebugOut(TypedDict): debug_result: DebugResult
class ErrorRecord(TypedDict):
    attempt: int; stage: Literal["validate","sandbox"]; cls: str; hash: str; digest: str
class CheckReport(TypedDict):  ok: bool; stage: str; cls: str | None; raw: str | None
class FixPlan(TypedDict):
    tier: Literal["small","mid"]; strategy: Literal["patch","explain_then_patch","regenerate"]
class DebugState(DebugIn, DebugOut):
    attempt: NotRequired[int]
    code_ref: NotRequired[str | None]
    code_hash: NotRequired[str | None]
    last_check: NotRequired[CheckReport | None]
    fix_plan: NotRequired[FixPlan | None]
    errors: Annotated[list[ErrorRecord], operator.add]      # 只活在本次 debug 呼叫
    spent: Annotated[Cost, add_cost]

# ================= L2 Refine（私有：champion / 假說帳本 / tabu / 計數器） =================
class GateReport(TypedDict): passed: bool; failed_rules: list[str]; improved: bool
class Reflection(TypedDict):
    mechanism: str; change: dict; prediction: dict; evidence: list[dict]; valid: bool; fingerprint: str
class RefineIn(TypedDict):  refine_request: RefineRequest
class RefineOut(TypedDict): refine_result: RefineResult
class RefineState(RefineIn, RefineOut):
    debug_request: NotRequired[DebugRequest]     # 往 L1 的信箱
    debug_result: NotRequired[DebugResult]       # 從 L1 回來的信箱
    iter: NotRequired[int]
    current: NotRequired[VariantRecord | None]
    champion: NotRequired[VariantRecord | None]
    champion_passed: NotRequired[bool]
    gate: NotRequired[GateReport | None]
    reflection: NotRequired[Reflection | None]
    no_progress: NotRequired[int]
    refuted_streak: NotRequired[int]
    history: Annotated[list[dict], operator.add]      # 每個變體 3–5 個摘要數字
    verdicts: Annotated[list[dict], operator.add]     # 假說帳本：{fp, confirmed, delta}
    tabu: Annotated[list[str], operator.add]          # 被推翻的 change fingerprint
    spent: Annotated[Cost, add_cost]

# ================= L3 Main =================
class HumanDecision(TypedDict):
    decision: Literal["approve","reject"]; reviewer: str; note: str
class MainState(TypedDict):
    run_id: str
    budget: Cost                                   # 靜態上限
    data_manifest: NotRequired[dict]               # IS/OOS/lockbox 期間 + sha256 + 引擎版本
    idea_n: NotRequired[int]
    current_spec: NotRequired[StrategySpec | None]
    refine_request: NotRequired[RefineRequest]     # 往 L2 的信箱
    refine_result: NotRequired[RefineResult]       # 從 L2 回來的信箱
    oos_uses: NotRequired[int]
    oos_report: NotRequired[dict | None]
    stat_gate: NotRequired[dict | None]
    finalist: NotRequired[VariantRecord | None]
    lockbox_report: NotRequired[dict | None]       # 白名單上沒有它，永遠不會被渲染進任何 prompt
    human: NotRequired[HumanDecision | None]
    report_ref: NotRequired[str | None]
    lessons: Annotated[list[Lesson], merge_lessons]
    tried_families: Annotated[list[str], operator.add]
    spent: Annotated[Cost, add_cost]
```

### 3.4 層間 state 轉換：選擇「子圖直接當 node + 信箱鍵 + prepare/absorb 轉接 node」

| 面向 | A. 子圖直接 `add_node(name, compiled)`（本設計採用） | B. node 函式內 `sub.invoke(transform(state))` |
|---|---|---|
| 鍵名耦合 | 父子必須有同名鍵（信箱鍵） | 完全自由映射 |
| 私有鍵外洩 | 父圖自動過濾（F2） | 由包裝函式手動挑選 |
| reducer 重複陷阱 | 會發生（F3），必須用 `output_schema` 限制 | 不會發生 |
| interrupt / 崩潰後 resume | 從子圖 checkpoint 接續 | 包裝函式從頭重跑，`invoke` 前的程式碼會重複執行（F7） |
| 可視化 | `get_graph(xray=True)` 可以展開 | 看不到子圖內部邊 |
| error_handler | 可以直接掛在子圖 node 上（已驗證） | 也可以，或在函式內 try/except |
| 同一 node 內呼叫多次 | 不適用 | 可以（實驗：`[1, 11]`） |

**選 A 的原因**：resume 語意比較乾淨，可視化與分層 stream 是免費的，信箱鍵強迫每條邊界只有一個明確契約。A 的唯一陷阱（F3）用兩條規則封死：(1) 子圖 `output_schema` 只含結果信箱；(2) 信箱鍵在父圖不掛 reducer。轉換邏輯放在純函式 node `prepare_*` 與 `absorb_*`，可以直接做單元測試：

```python
def r_prepare_debug(s: RefineState) -> dict:        # L2 → L1：切預算、帶入 change
    req, ref, base = s["refine_request"], s.get("reflection"), s.get("champion")
    return {"debug_request": {
        "variant_id": f"{req['idea_id']}-v{s['iter']}", "spec": req["spec"],
        "mode": "patch" if ref else "generate",
        "base_code_ref": base["code_ref"] if base else None,
        "change": ref["change"] if ref else None,
        "allowance": min_cost({"usd": 0.40, "wall_s": 900}, remaining(req["allowance"], s.get("spent", {}))),
        "max_attempts": MAX_DEBUG_ATTEMPTS}}

def r_absorb_debug(s: RefineState) -> dict:         # L1 → L2：只回收 spent 並推進 iter
    return {"spent": s["debug_result"]["spent"], "iter": s["iter"] + 1}

def m_absorb_refine(s: MainState) -> dict:          # L2 → L3：spent / lessons / family 用 reducer 累積到上層
    r = s["refine_result"]
    return {"spent": r["spent"], "lessons": r["lessons"], "tried_families": [s["current_spec"]["family"]]}
```

**錯誤不汙染上層**有三道保證：(1) F2 過濾私有鍵（原型最後的 `assert "debug_request" not in main_state` 通過）；(2) `error_handler` 把子圖例外轉成 `status="crashed"` 結果物件（F9）；(3) 子圖每次呼叫都是新 namespace（F4），上一輪的錯誤歷史不會延續。

---

## 4. 節點清單

| 層 | Node | 類型 | 模型 / 實作 | 輸入（讀取的鍵） | 輸出（寫入的鍵） |
|---|---|---|---|---|---|
| L3 | `init_run` | 確定性 | — | run_id, budget | data_manifest（期間、sha256、引擎版本）, 計數器 |
| L3 | `ideate` | LLM | `claude-opus-5`，effort=high，結構化輸出 3 個候選 | lessons（≤12 行）, tried_families, 允許的 family 清單 | current_spec, idea_n, spent |
| L3 | （`ideate` 內的 novelty_guard） | 確定性 | Jaccard（family + 訊號特徵 token） | 3 個候選 + 過去的 specs | 選出最新穎且 family 未超配額者；全部太相似則重試 1 次 |
| L3 | `prepare_refine` | 確定性 | — | current_spec, budget, spent | refine_request |
| L3 | `refine` | **子圖 L2** | compiled，`error_handler=refine_crash_handler` | refine_request | refine_result |
| L3 | `absorb_refine` | 確定性 | — | refine_result | spent, lessons, tried_families |
| L3 | `oos_eval` | 確定性 | 沙盒 + 引擎；DataGateway 發 `is+oos` 一次性 handle | champion | oos_report, oos_uses |
| L3 | `stat_gate` | 確定性 | DSR（ledger 全部 N）、PBO、衰退比、t 門檻隨 k 升高 | oos_report, refine_result, ledger | stat_gate, finalist |
| L3 | `distill_failure` | 確定性 | 模板化，不用 LLM | stat_gate, refine_result | lessons（只含類別），清空 stat_gate |
| L3 | `lockbox_eval` | 確定性 | 一次性許可，結果只給人看 | finalist | lockbox_report |
| L3 | `human_review` | interrupt | `interrupt(payload, response_schema=HumanDecision)`，前面沒有副作用 | finalist, oos, stat_gate, lockbox | human |
| L3 | `report` | LLM + 確定性 | `claude-sonnet-5` 撰寫敘述；數字表格由程式直接填 | 全部 | report_ref |
| L2 | `init` | 確定性 | — | — | iter=0, champion=None, 計數器 |
| L2 | `prepare_debug` | 確定性 | — | refine_request, reflection, champion | debug_request |
| L2 | `debug` | **子圖 L1** | compiled，`error_handler=debug_crash_handler` | debug_request | debug_result |
| L2 | `absorb_debug` | 確定性 | — | debug_result | spent, iter |
| L2 | `backtest_is` | 確定性 | 固定引擎：錨定 walk-forward 6 折、purge（最大 lookback）+ embargo 1 天、每折內層網格選參（≤16 組）、鄰域 ±1 格、區塊 bootstrap SE；同時產出 DiagnosticPack artifact；每個 combo 寫一筆 ledger | debug_result | current, history, spent.backtests |
| L2 | `score_prediction` | 確定性 | 比對上一輪登記的預測 | reflection, current, champion | verdicts, refuted_streak, tabu |
| L2 | `gate` | 確定性 | IS_GATE 寫死 | current, champion | gate, champion, champion_passed, no_progress |
| L2 | `reflect` | LLM | `claude-opus-5`，effort=high，structured output | DiagnosticPack 精簡版（≤1.5k tokens）、假說帳本、tabu、失敗的規則名稱 | reflection, spent |
| L2 | `verify_reflection` | 確定性 | 證據數字比對、fingerprint 去重、單一變更檢查 | reflection, DiagnosticPack, tabu | reflection.valid |
| L2 | `finalize` | 確定性 | 模板化 lessons（由假說帳本生成） | champion, champion_passed, verdicts | refine_result |
| L1 | `coder` | LLM | 生成：`claude-sonnet-5`；修復：依 fix_plan 用 `claude-haiku-4-5`（small）或 `claude-sonnet-5`（mid）；`RetryPolicy(3)` 只處理 API 429/5xx | debug_request, fix_plan, errors[-1], 上一版程式 | code_ref, code_hash, attempt, spent |
| L1 | `validate` | 確定性 | AST 前視掃描、import 白名單、PARAM_SPACE schema、參數數量 ≤4 | code_ref | last_check |
| L1 | `sandbox_run` | 確定性 | 拋棄式容器（`--network none`、read-only、非 root、CPU/記憶體限制、無秘密）；smoke 跑 2k 根 K 線 + 截斷不變性（隨機 20 個 t）+ 輸出契約（index 對齊、[-1,1]、NaN 比例）；`RetryPolicy(2)` 只處理 InfraError | code_ref | last_check |
| L1 | `triage` | 確定性 | 分類 + 正規化 hash + 升級策略 | last_check, errors, attempt | errors, fix_plan |
| L1 | `finish` | 確定性 | — | last_check, errors, spent | debug_result |

模型 ID 取自 claude-api skill 的價目表（快取日期 2026-06-24），上線前請再確認：Opus 5 為 $5/$25、Sonnet 5 為 $2/$10、Haiku 4.5 為 $1/$5（每百萬 token，輸入/輸出）。Opus 5 不支援調 temperature，所以多樣性靠「一次產出 3 個候選 + 確定性挑選」，不靠取樣。

---

## 5. Routing functions（達標規則寫死在程式裡）

```python
# ---------------- 常數（改動需要 code review；run 時寫進 data_manifest 以利重現） ----------------
MAX_DEBUG_ATTEMPTS, SAME_HASH_GIVE_UP, LOOKAHEAD_GIVE_UP = 4, 3, 2
IS_GATE = dict(min_wf_sharpe=1.0, min_pos_fold_frac=0.67, min_fold_trades=15, max_dd=0.25,
               max_cost_share=0.5, min_neighbor_ratio=0.7, max_params=4)
PLATEAU_PATIENCE, REFUTED_STREAK_MAX, MIN_IMPROVE = 2, 3, 0.10
MAX_IDEAS, MAX_OOS_USES = 6, 6
STAT_GATE = dict(min_oos_sharpe=0.8, min_decay_ratio=0.4, min_dsr=0.95, max_pbo=0.30,
                 min_oos_trades=60, max_oos_dd=0.25)

def over(spent: Cost, cap: Cost) -> bool:
    return any(spent.get(k, 0) >= v for k, v in cap.items() if v is not None)

# ======================= L1 Debug =======================
def route_after_check(s: DebugState) -> Literal["sandbox_run", "finish", "triage"]:
    chk = s["last_check"]
    if chk["ok"]:
        return "sandbox_run" if chk["stage"] == "validate" else "finish"
    return "triage"

def route_after_triage(s: DebugState) -> Literal["coder", "finish"]:
    req, errs = s["debug_request"], s["errors"]; last = errs[-1]
    if over(s.get("spent", {}), req["allowance"]):                        return "finish"   # → status=budget
    if s["attempt"] >= req["max_attempts"]:                                return "finish"   # → gave_up
    if last["cls"] == "E_INFRA":                                           return "finish"   # RetryPolicy 已重試過
    if sum(e["hash"] == last["hash"] for e in errs) >= SAME_HASH_GIVE_UP:  return "finish"
    if sum(e["cls"] == "E_LOOKAHEAD" for e in errs) >= LOOKAHEAD_GIVE_UP:  return "finish"   # idea 本身可能需要未來資料
    return "coder"

# ======================= L2 Refine =======================
def r_gate(s: RefineState) -> dict:            # 確定性 node：計算閘門，寫進 state 給 router 讀
    c, champ, G = s["current"], s.get("champion"), IS_GATE
    failed = [n for n, bad in [
        ("wf_sharpe",     c["wf_sharpe"] < G["min_wf_sharpe"]),
        ("pos_fold_frac", c["pos_fold_frac"] < G["min_pos_fold_frac"]),
        ("fold_trades",   c["min_fold_trades"] < G["min_fold_trades"]),
        ("max_dd",        c["max_dd"] > G["max_dd"]),
        ("cost_share",    c["cost_share"] > G["max_cost_share"]),
        ("neighbor",      c["neighbor_ratio"] < G["min_neighbor_ratio"]),
        ("n_params",      c["n_params"] > G["max_params"])] if bad]
    # 「進步」必須超過雜訊帶：至少 +0.10 Sharpe，且超過 2 倍 bootstrap SE
    improved = champ is None or c["wf_sharpe"] > champ["wf_sharpe"] + max(MIN_IMPROVE, 2 * c["wf_sharpe_se"])
    upd = {"gate": {"passed": not failed, "failed_rules": failed, "improved": improved}}
    upd |= ({"champion": c, "champion_passed": not failed, "no_progress": 0} if improved
            else {"no_progress": s["no_progress"] + 1})
    return upd

def route_after_debug(s: RefineState) -> Literal["backtest_is", "reflect", "finalize"]:
    dr, req = s["debug_result"], s["refine_request"]
    if dr["status"] in ("budget", "crashed"):  return "finalize"
    if dr["status"] == "gave_up":
        if s.get("champion") is None:          return "finalize"   # 第一版就做不出來 → idea infeasible，升級到 L3
        if s["iter"] >= req["max_iters"]:      return "finalize"
        return "reflect"                                            # 這個 change 做不出來 → 回反思，改用別的 change
    return "backtest_is"

def route_after_gate(s: RefineState) -> Literal["reflect", "finalize"]:
    req, g = s["refine_request"], s["gate"]
    if g["passed"] and g["improved"]:                  return "finalize"   # 滿足即停（satisficing），不再追高 IS
    if over(s.get("spent", {}), req["allowance"]):     return "finalize"
    if s["iter"] >= req["max_iters"]:                  return "finalize"
    if s["no_progress"] >= PLATEAU_PATIENCE:           return "finalize"   # 原地打轉
    if s["refuted_streak"] >= REFUTED_STREAK_MAX:      return "finalize"   # 反思連續猜錯 → 反思沒有資訊量
    return "reflect"

def route_after_verify(s: RefineState) -> Literal["prepare_debug", "finalize"]:
    ref = s["reflection"]
    if ref["mechanism"] == "abandon_idea":                          return "finalize"
    if not ref["valid"] and s["no_progress"] + 1 >= PLATEAU_PATIENCE:  return "finalize"
    return "prepare_debug"

# finalize 內的狀態判定（同樣是確定性規則）：
#   champion is None → debug_result.status ∈ {budget, crashed}，否則為 infeasible
#   champion_passed  → "candidate"，否則 → "no_candidate"

# ======================= L3 Main =======================
def route_after_refine(s: MainState) -> Literal["oos_eval", "distill_failure"]:
    r = s["refine_result"]
    if r["status"] == "candidate" and s["oos_uses"] < MAX_OOS_USES and not over(s["spent"], s["budget"]):
        return "oos_eval"
    return "distill_failure"

def m_stat_gate(s: MainState, runtime) -> dict:
    o, r, G = s["oos_report"], s["refine_result"], STAT_GATE
    n_trials = runtime.context.ledger.count(data_hash=s["data_manifest"]["sha256"])  # 含歷史 run
    dsr = deflated_sharpe(r["champion"]["diag_ref"], n_trials=n_trials)             # Bailey & López de Prado 2014
    t_req = 1.96 + 0.35 * (s["oos_uses"] - 1)          # 每多看一次 OOS，門檻就提高一些
    checks = {"oos_sharpe": o["sharpe"] >= G["min_oos_sharpe"],
              "decay":      o["sharpe"] >= G["min_decay_ratio"] * r["champion"]["wf_sharpe"],
              "dsr":        dsr >= G["min_dsr"],
              "pbo":        (r["pbo"] if r["pbo"] is not None else 1.0) <= G["max_pbo"],
              "trades":     o["trades"] >= G["min_oos_trades"],
              "dd":         o["max_dd"] <= G["max_oos_dd"],
              "t_stat":     o["t_stat"] >= t_req}
    passed = all(checks.values())
    return {"stat_gate": {"passed": passed, "checks": checks, "dsr": dsr, "n_trials": n_trials},
            "finalist": r["champion"] if passed else None}

def route_after_stat_gate(s: MainState) -> Literal["lockbox_eval", "distill_failure"]:
    return "lockbox_eval" if s["stat_gate"]["passed"] else "distill_failure"

def route_next_idea(s: MainState) -> Literal["ideate", "report"]:
    if s["idea_n"] >= MAX_IDEAS or over(s["spent"], s["budget"]):  return "report"
    if s.get("refine_result", {}).get("status") == "crashed":       return "report"  # 基礎設施故障，不浪費 idea
    return "ideate"
# lockbox_eval → human_review → report → END：拓撲上不存在回到 ideate 的邊（lockbox 用過就作廢）
```

---

## 6. 反思機制

**目標**：每次反思都是一次「可被證偽的小實驗」，不是一段敘事。

### 6.1 輸入（全部來自確定性程式，只看 IS）
1. **DiagnosticPack**（由 `backtest_is` 產出，存成 artifact，prompt 只放精簡版 ≤1.5k tokens）：
   - 每折指標（Sharpe、報酬、DD、交易數、勝率、持有期中位數）
   - regime 切片：波動三分位 × 趨勢狀態（ADX 或 200MA 斜率），每格的 long/short PnL 與交易數
   - 成本歸因：gross 與 net、手續費、滑價、資金費率分別佔 PnL 多少；turnover
   - 參數鄰域熱圖摘要（中心值與 ±1 格）
   - 最差 3 段回撤的起訖時間與當時的 regime 標籤
   - 訊號統計：曝險比例、翻倉頻率、平均 |position|
2. **假說帳本**：過去每次反思的 `mechanism/change/prediction` 與 `confirmed/refuted/delta`
3. **tabu 清單**：已被推翻的 change fingerprint
4. **閘門失敗的規則名稱**（例如 `["cost_share","neighbor"]`）。**不給門檻數值**，降低「剛好卡過門檻」的博弈
5. **明確不給**：OOS 或 lockbox 的任何資訊、其他 idea 的程式碼、debug 的錯誤歷史（只給 L1 回傳的 `error_class`/`error_digest`，而且只在變體做不出來時給）

### 6.2 輸出 schema（structured output，`strict` JSON schema）

```python
class Evidence(TypedDict):
    metric_path: str          # 必須是 DiagnosticPack 內存在的路徑，例如 "by_regime.high_vol.short.pnl"
    observed: float           # 必須和 pack 內數值一致（|Δ| ≤ 1e-6）
    comparator: Literal["<", ">"]
    reference: float          # 相比較的數值或另一路徑的值

class ChangeSpec(TypedDict):
    kind: Literal["regime_filter","entry_logic","exit_logic","position_sizing",
                  "param_range","cost_reduction","revert"]
    target: str               # 修改的函式或區塊
    description: str          # ≤300 字元
    param_space_delta: dict   # 新增或刪除的參數；新增後總數必須 ≤4

class Prediction(TypedDict):
    metric_path: str          # 例如 "wf_sharpe" 或 "cost.share"
    direction: Literal["increase","decrease"]
    min_delta: float          # 事先承諾的最小效果
    guard_metrics: list[str]  # 不可惡化超過 10% 的指標

class ReflectionOut(TypedDict):
    diagnosis: str                    # ≤400 字元
    evidence: list[Evidence]          # 1–4 條
    hypothesis: str                   # 單一因果主張
    mechanism: Literal[..., "abandon_idea"]
    change: ChangeSpec                # 每輪只能有一個變更
    prediction: Prediction
    abandon_reason: str | None
```

### 6.3 如何避免事後編故事
- **證據必須對得上數字**：`verify_reflection` 逐條查 `metric_path` 是否存在、`observed` 是否等於 pack 值、比較方向是否成立。任何一條不符就標記 `valid=False`（不重問，避免在同一 prompt 上反覆磨），並計一次 `no_progress`。LLM 無法引用不存在或捏造的數字。
- **預先登記預測**：預測在跑回測**之前**就寫進 state（有 checkpoint 可稽核），由 `score_prediction` 用程式判定成立或被推翻。假說帳本累積的命中率就是反思品質的直接量測（R2 可觀測）。
- **一輪只改一處**：`ChangeSpec` 只有一個欄位，讓 IS 差異可以歸因到該變更。
- **勝負由程式定**：進步必須大於 `max(0.10, 2×SE)`，LLM 的 `confidence` 不參與任何路由。

### 6.4 如何避免原地打轉
- **fingerprint**：`sha1(kind + target + 正規化 description 關鍵詞 + param_space_delta)`。命中 tabu 就判無效。
- **三個停止計數器**：`no_progress ≥ 2`（連續兩輪未超過雜訊帶）、`refuted_streak ≥ 3`（連續三次預測被推翻，代表模型對這個 idea 的理解是錯的）、`iter ≥ 5`。任一觸發就結束 L2，把學到的東西升級到 L3 去換 idea。
- **champion 只在顯著進步時更新**，下一輪一律以 champion 為基底打補丁（`base_code_ref=champion.code_ref`）。失敗的變體不會變成新基底，不會越改越偏。
- **允許主動放棄**：`abandon_idea` 需要附上通過驗證的證據；由 router 決定結束，而不是讓 LLM 決定「達標」。

### 6.5 lessons 如何往上傳（控制 token）
L2 的 `finalize` 從假說帳本**模板化**生成最多 3 條 lesson（例如 `"trend|a1b2c3: regime_filter 在 high_vol 有效 (IS)"`），不經過 LLM，所以不會膨脹也不會美化。L3 用 `merge_lessons` 去重，只留最近 12 條，每條 ≤200 字元，進 ideate prompt 的總量約 600 tokens。

---

## 7. 除錯機制（L1，參考 Chen et al. 2023《Teaching LLMs to Self-Debug》）

### 7.1 錯誤分類（`triage`，確定性）

| 類別 | 偵測來源 | 給 coder 的回饋 | 處理方式 |
|---|---|---|---|
| `E_SYNTAX` | `validate`（ast.parse） | 行號與訊息 | small / patch |
| `E_IMPORT_FORBIDDEN` | `validate` 白名單 | 被禁的 import 與允許清單 | small / patch |
| `E_LOOKAHEAD_STATIC` | `validate` AST 規則（`shift(-k)`、`center=True`、全域 `.max()`/`.mean()` 未 rolling、`iloc[t+1]` 等） | 命中規則與程式片段 | 計入 E_LOOKAHEAD，mid / explain_then_patch |
| `E_LOOKAHEAD` | `sandbox_run` 截斷不變性測試 | 第一個不一致的 t，以及 full[t] 與 trunc[t] 的差異 | 同上；累計 2 次就放棄 |
| `E_RUNTIME` | 沙盒例外 | 截斷後的 traceback（≤600 字元）+ 出錯列前後 3 列 df 樣本 | small / explain_then_patch |
| `E_OUTPUT_SCHEMA` | 沙盒輸出契約（index 不齊、超出 [-1,1]、NaN>5%、dtype） | 違反的條款與統計 | small → 同 hash 第 2 次改 regenerate |
| `E_TIMEOUT` / `E_OOM` | 容器退出碼 | 資源使用量 | mid / explain_then_patch（提示向量化） |
| `E_INFRA` | 容器啟動失敗、docker 錯誤 | —（不給 LLM） | node 的 `RetryPolicy(max_attempts=2)` 重試；仍失敗就 `finish`，上報 status=gave_up，L2 route 到 finalize，L3 視為 crashed 並停止 run |

**語意退化**（全 0 訊號、交易數太少）**不在 L1 處理**。L1 只保證「程式能跑而且符合契約」；策略好不好是 L2 閘門的事。這條邊界避免 debug 迴圈偷偷改策略邏輯。

### 7.2 Self-Debug 流程
- **錯誤 hash**：traceback 或訊息去掉數字、位址、字串常值後取 sha1 前 10 碼（`normalize_error`）。
- **升級階梯**（依 `fix_plan`）：
  1. 首次遇到某個 hash：`claude-haiku-4-5` 做 patch（`E_RUNTIME` 用 explain_then_patch）
  2. 同一 hash 第 2 次：`claude-sonnet-5` 做 **explain_then_patch**（Chen et al. 的 rubber-duck 做法）：輸出 schema 強制先填 `code_explanation`（逐段說明程式在做什麼）與 `failure_trace`（依執行回饋推論錯在哪裡），再填 `patch`，讓模型先對齊「程式實際行為 vs 意圖」
  3. 只剩最後一次嘗試，或 `E_OUTPUT_SCHEMA`/`E_LOOKAHEAD` 重複出現：**regenerate**，依 spec 從頭重寫，不再沿用壞掉的程式
- **輸入最小化**：只給最近一次錯誤的 digest（不給全部歷史，避免 context 汙染），外加上一版程式與 spec。

### 7.3 何時放棄（全部寫在 `route_after_triage`）
`attempt ≥ 4`，或同一 hash 出現 3 次，或前視錯誤累計 2 次，或 `E_INFRA`，或本次呼叫預算（$0.40 / 15 分鐘）用完。

### 7.4 何時升級到上層、上層怎麼處理
- L1 只回傳 `DebugResult`（status + error_class + ≤300 字元 digest + attempts + spent）。完整錯誤歷史留在 L1 的 checkpoint namespace 供稽核，不進 L2 state。
- L2 的處理：
  - **第一版就失敗**（還沒有 champion）→ idea 判為 `infeasible`，L3 收到 `implementation` 類 lesson 後換 idea。
  - **某個變體失敗** → 回到 `reflect`，並把 `error_class` 給反思參考（例如「這個 regime_filter 需要未來資料」），同時計一次 iter。
  - **`crashed`**（框架或基礎設施例外，經 `error_handler` 轉換）→ L2 finalize → L3 直接去 `report`，由人修基礎設施，不浪費 idea 額度。
- 修語法的嘗試**不寫 trial ledger**；只有 `backtest_is` 真正評估的 (variant, combo) 才算試驗。

---

## 8. 過擬合控制如何嵌入圖中

| 控制 | 嵌入位置 | 機制 |
|---|---|---|
| 三層資料分區 | `init_run` 固定切分：IS 2020-01–2024-06、OOS 2024-07–2025-06、lockbox 2025-07–2026-06（最近 12 個月），存 sha256 | **DataGateway** 依 `checkpoint_ns` 前綴強制存取：`refine\|debug` 只能拿 IS handle（F11 已驗證 namespace 可取得）；OOS handle 由 `oos_eval` 以 variant_id 一次性發放；lockbox 整個 run 一次。這是基礎設施層的強制，不只靠圖拓撲 |
| 迴圈內只用 IS | L2 `backtest_is` | 錨定 walk-forward 6 折、purge = PARAM_SPACE 宣告的最大 lookback、embargo 1 天、每折內層選參；指標用拼接的 out-of-fold 報酬 |
| 參數穩定性 | L2 `gate` | 鄰域 ±1 格中位 Sharpe ≥ 0.7×中心值；參數數量 ≤4 |
| 成本敏感度 | L2 `gate` | cost_share ≤ 0.5；手續費、滑價、資金費率由引擎固定 |
| 滿足即停 | `route_after_gate` | IS 一過閘門就停，不再追高 IS（減少試驗數，也減少挑中運氣好的變體） |
| 多重檢定 | L3 `stat_gate` | DSR 的 N = ledger 內同一資料 hash 的所有試驗（含歷史 run）；PBO 以 CSCV（S=10 區塊）計算該 idea 所有 (variant×combo) 試驗矩陣 |
| OOS 重用懲罰 | L3 `stat_gate` | 每變體最多 1 次（一次性許可）；整個 run 最多 6 次；t 門檻 `1.96 + 0.35·(k−1)` 隨使用次數提高 |
| OOS 資訊不回流 | L3 `distill_failure` | OOS 失敗只產生**類別型** lesson（例如 `failed held-out checks: decay,t_stat`），不含任何數字，避免 ideate 對 OOS 做隱性擬合 |
| lockbox 隔離 | L3 拓撲 | `lockbox_eval → human_review → report → END`，沒有任何邊回到 LLM node；`lockbox_report` 不在任何 prompt 的渲染白名單上 |
| 試驗帳本冪等 | `backtest_is` | `trial_id = sha1(run_id\|code_hash\|params_hash\|data_hash\|engine_ver)` 做 upsert，崩潰重跑不會重複計數 |

---

## 9. 建圖程式碼骨架（節錄，完整版見 `candidate_B_skeleton.py`）

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver         # 正式環境：PostgresSaver / SqliteSaver
from langgraph.types import Command, RetryPolicy, interrupt
from langgraph.errors import NodeError
from langgraph.runtime import Runtime

@dataclass
class Deps:                       # Runtime context：不可序列化的依賴放這裡，不進 state、不進 checkpoint
    run_id: str; store: ...; ledger: ...; gateway: ...; sandbox: ...; llm: ...

# ------------------------------ L1 ------------------------------
def build_debug_graph():
    g = StateGraph(DebugState, context_schema=Deps, input_schema=DebugIn, output_schema=DebugOut)
    g.add_node("coder", d_coder, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("validate", d_validate)
    g.add_node("sandbox_run", d_sandbox_run, retry_policy=RetryPolicy(max_attempts=2, retry_on=InfraError))
    g.add_node("triage", d_triage)
    g.add_node("finish", d_finish)
    g.add_edge(START, "coder")
    g.add_edge("coder", "validate")
    g.add_conditional_edges("validate",    route_after_check,  ["sandbox_run", "finish", "triage"])
    g.add_conditional_edges("sandbox_run", route_after_check,  ["sandbox_run", "finish", "triage"])
    g.add_conditional_edges("triage",      route_after_triage, ["coder", "finish"])
    g.add_edge("finish", END)
    return g.compile()            # checkpointer=None：繼承父圖；每次呼叫是全新 namespace（F4）

# ------------------------------ L2 ------------------------------
def debug_crash_handler(s: RefineState, error: NodeError) -> Command:
    req = s["debug_request"]
    return Command(update={"debug_result": {"variant_id": req["variant_id"], "status": "crashed",
                   "code_ref": None, "code_hash": None, "error_class": "E_FRAMEWORK",
                   "error_digest": repr(error.error)[:300], "attempts": 0, "spent": {}}},
                   goto="absorb_debug")                     # 一定要 goto，否則圖會默默停止（F9）

def build_refine_graph(debug_graph):
    g = StateGraph(RefineState, context_schema=Deps, input_schema=RefineIn, output_schema=RefineOut)
    g.add_node("init", r_init)
    g.add_node("prepare_debug", r_prepare_debug)
    g.add_node("debug", debug_graph, error_handler=debug_crash_handler)      # ← 編譯好的子圖當 node
    g.add_node("absorb_debug", r_absorb_debug)
    g.add_node("backtest_is", r_backtest_is)
    g.add_node("score_prediction", r_score_prediction)
    g.add_node("gate", r_gate)
    g.add_node("reflect", r_reflect, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("verify_reflection", r_verify_reflection)
    g.add_node("finalize", r_finalize)
    g.add_edge(START, "init"); g.add_edge("init", "prepare_debug")
    g.add_edge("prepare_debug", "debug"); g.add_edge("debug", "absorb_debug")
    g.add_conditional_edges("absorb_debug", route_after_debug, ["backtest_is", "reflect", "finalize"])
    g.add_edge("backtest_is", "score_prediction"); g.add_edge("score_prediction", "gate")
    g.add_conditional_edges("gate", route_after_gate, ["reflect", "finalize"])
    g.add_edge("reflect", "verify_reflection")
    g.add_conditional_edges("verify_reflection", route_after_verify, ["prepare_debug", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()

# ------------------------------ L3 ------------------------------
def m_human_review(s: MainState) -> dict:
    # 前面沒有任何副作用：resume 時本 node 會從頭重跑
    decision = interrupt({"kind": "strategy_review", "run_id": s["run_id"], "finalist": s["finalist"],
                          "oos": s["oos_report"], "stat_gate": s["stat_gate"], "lockbox": s["lockbox_report"]},
                         response_schema=HumanDecision)       # 框架會驗證 resume 值（F12）
    return {"human": decision}

def build_main_graph(checkpointer):
    refine = build_refine_graph(build_debug_graph())
    g = StateGraph(MainState, context_schema=Deps)
    g.add_node("init_run", m_init)
    g.add_node("ideate", m_ideate, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("prepare_refine", m_prepare_refine)
    g.add_node("refine", refine, error_handler=refine_crash_handler)         # ← 編譯好的子圖當 node
    g.add_node("absorb_refine", m_absorb_refine)
    g.add_node("oos_eval", m_oos_eval)
    g.add_node("stat_gate", m_stat_gate)
    g.add_node("distill_failure", m_distill_failure)
    g.add_node("lockbox_eval", m_lockbox_eval)
    g.add_node("human_review", m_human_review)
    g.add_node("report", m_report)
    g.add_edge(START, "init_run"); g.add_edge("init_run", "ideate")
    g.add_edge("ideate", "prepare_refine"); g.add_edge("prepare_refine", "refine")
    g.add_edge("refine", "absorb_refine")
    g.add_conditional_edges("absorb_refine", route_after_refine, ["oos_eval", "distill_failure"])
    g.add_edge("oos_eval", "stat_gate")
    g.add_conditional_edges("stat_gate", route_after_stat_gate, ["lockbox_eval", "distill_failure"])
    g.add_conditional_edges("distill_failure", route_next_idea, ["ideate", "report"])
    g.add_edge("lockbox_eval", "human_review"); g.add_edge("human_review", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)

# ------------------------------ 執行 / HITL ------------------------------
app  = build_main_graph(PostgresSaver.from_conn_string(DSN))
cfg  = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
out  = app.invoke(init_state, cfg, context=Deps(run_id=run_id, ...), durability="sync")
# out["__interrupt__"] → 通知審閱者；分層觀察：
for ns, upd in app.stream(None, cfg, stream_mode="updates", subgraphs=True, context=deps): ...
snap = app.get_state(cfg, subgraphs=True)
app.invoke(Command(resume={"decision": "approve", "reviewer": "allen", "note": "..."}), cfg, context=deps)
```

**原型實跑結果**（stub 版）：
```
INTERRUPT: ['strategy_review']
next: ('human_review',) | 主 state 鍵不含 debug_request/history 等私有鍵（assert 通過）
spent so far: {'usd': 0.3, 'in_tok': 42000, 'out_tok': 15500, 'backtests': 192}
human: {'decision': 'approve', ...}  report: report:d3946cfd6f3e
namespaces: 'refine:<uuid>', 'refine:<uuid>|debug:<uuid>' ...
崩潰測試：sandbox 拋 RuntimeError → debug_crash_handler → refine 狀態 crashed → 主圖直接 report（ideas=1）
預算測試：usd 上限 0.12 → 實際 spent 0.13（超出一次呼叫）→ status=budget → report
```

**各層獨立測試**：`build_debug_graph().invoke({"debug_request": fake_req}, context=FakeDeps(...))` 不需要 checkpointer；router 是純函式，可以用 property test 窮舉（例如「attempt≥4 必定 finish」）；prepare/absorb 轉接 node 用 golden test；另有一條 **F3 回歸測試**：斷言子圖 `output_schema` 的鍵集合 ∩ 父圖 reducer 鍵集合 = ∅。

---

## 10. 成本估算

單價（claude-api skill 快取 2026-06-24，每百萬 token，輸入/輸出）：Opus 5 $5/$25；Sonnet 5 $2/$10；Haiku 4.5 $1/$5。輸出 token 已含 adaptive thinking 的估計量。**未計入 prompt caching**（系統提示與契約約 4k tokens 可快取，實際輸入成本約再降 20–30%）。

| 呼叫 | 模型 | 輸入/輸出 tokens | 單次成本 |
|---|---|---|---|
| ideate（3 候選） | Opus 5 | 8k / 3k | $0.115 |
| coder 生成/補丁 | Sonnet 5 | 6k / 4k | $0.052 |
| fix（small） | Haiku 4.5 | 8k / 3k | $0.023 |
| fix（mid） | Sonnet 5 | 8k / 4k | $0.056 |
| reflect | Opus 5 | 10k / 3k | $0.125 |
| report | Sonnet 5 | 12k / 4k | $0.064 |

| 層級 | 期望值 | 最壞情況（受上限約束） |
|---|---|---|
| 一次 debug 呼叫 | $0.067（平均 0.5 次 fix） | $0.19（1 small + 2 mid），上限 $0.40 |
| 一輪 refine（debug + reflect） | ≈ $0.19 | ≈ $0.31 |
| 一個 idea | ≈ $0.57（3 輪、2 次反思） | ≈ $1.67，被 IDEA_CAP $1.6 截斷 |
| 一次 run | ≈ $2.3（約 4 個 idea）+ report | 6 × $1.6 + $0.06 ≈ **$9.7**；run 硬上限 $12（最多超出一次呼叫） |
| Tokens / run | ≈ 32 萬 | ≈ 90 萬 |
| 回測次數 | 每輪 16 combo × 6 折 + 8 個鄰域 ≈ 100 次折回測；期望 12 輪 ≈ 1.2k | 30 輪 ≈ 3k（上限 20k） |
| 沙盒容器 | 每個變體 1 次（16 combo 在同一容器內批次生成訊號），約 12–30 次/run | 每次 debug 嘗試另加 1 次 smoke |
| Wall-clock | 每輪 2–3 分鐘（LLM 60–150 秒、沙盒 10–30 秒、回測 5–10 秒）→ 30–40 分鐘 | 約 90 分鐘；上限 2 小時；人工審閱另計（非同步等待） |
| DSR 試驗數 N | 約 190（12 變體 × 16 combo） | 約 480（盡量讓 N 保持小，因為 DSR 的懲罰隨 √(2 ln N) 增長） |

**預算控制方式**：由上而下切 allowance（run → idea $1.6 → debug $0.40），由下而上用 `spent` reducer 回收，每個 router 都檢查。`LLMClient` 在呼叫前依「輸入 token 數 + max_tokens × 輸出單價」估算，若會超過本層 allowance 就直接拋 `BudgetExceeded`，由 error_handler 轉成 `status=budget`，把超支上限從「一整次呼叫」壓到「一次估算誤差」。

---

## 11. 優點、弱點、失敗模式

### 優點
1. **關注點分離是結構性的**：三組計數器、三份預算、三個 checkpoint namespace，各自存在於各自的 state，不靠命名慣例維持。
2. **每層可以獨立測試**：L1 可以單獨用 100 個已知壞程式碼跑 regression；L2 可以餵固定的 DebugResult 測反思路由；L3 可以 mock 整個 refine 子圖。
3. **錯誤隔離**：F2 過濾 + error_handler + 每次呼叫全新 namespace，三道保證（原型已驗證）。
4. **稽核與重現**：`get_state(subgraphs=True)` 與 `stream(subgraphs=True)` 天然分層；每個 LLM 呼叫以 (run_id, checkpoint_ns, node, attempt) 為鍵記錄 prompt 版本、模型 ID、輸入 hash、輸出 JSON、usage、成本。
5. **過擬合控制有基礎設施層強制**：DataGateway 依 namespace 限制分區、一次性 OOS/lockbox 許可、冪等 ledger。
6. **成本可預測**：最壞情況有數學上限（6 × 1.6 + ε）。

### 弱點
1. **探索是序列的**：一次只精煉一個 idea，最多 6 個；容易被前幾個 idea 的 family 錨定（R4 弱項）。可擴充：在 L3 用 `Send` 平行跑 2–3 個 refine 子圖實例（每個都是獨立 namespace），代價是成本倍增，而且 OOS 使用次數會上升得更快。
2. **樣板程式較多**：三組 In/Out/State schema，外加 prepare/absorb 轉接 node。約 30% 的程式碼是「搬運」。
3. **上層資訊不足**：L2 只看到 debug 的 `error_class` 與 digest；如果某個 change 反覆以不同方式做不出來，反思未必能看出原因。
4. **滿足即停可能挑中剛好過線的變體**：設計上刻意犧牲 IS 峰值來換取較少的試驗數，因此 OOS 通過率可能偏低，需要多幾個 idea。
5. **OOS 在 6 個 idea 間仍會形成選擇偏誤**：OOS 實際上退化成一個「驗證集」。t 門檻遞增與 lockbox 只能緩解，無法消除。

### 失敗模式（與對策）

| 失敗模式 | 觸發原因 | 對策 |
|---|---|---|
| 共用 reducer 鍵重複累加 | 有人在子圖 output 加上 `lessons` 之類的 reducer 鍵 | F3 回歸測試（schema 交集 = ∅），CI 擋下 |
| 子圖第一個 node 取值時 KeyError | 新 namespace 的私有鍵尚未初始化 | 私有鍵一律標 `NotRequired`，並用 `init` node 設預設值，讀取一律用 `.get` |
| error_handler 默默讓圖結束 | handler 回傳 dict 而不是 `Command(goto=...)` | 規範 + 測試：handler 回傳型別必須是 `Command` |
| 誤設 `checkpointer=True` | 上一個 idea 的 attempt/errors 延續下來 | 在 build 函式內斷言 `sub.checkpointer is None` |
| 預算超出一次呼叫 | 路由只在呼叫之後檢查 | LLMClient 呼叫前估算（§10） |
| resume 後重複寫入副作用 | node 在崩潰前已寫 ledger | ledger/artifact 皆以內容 hash 冪等 upsert |
| 圖結構改版後舊 thread 無法 resume | namespace 或 node 名稱改變 | 圖版本寫進 data_manifest；版本不同就不 resume，改開新 run |
| 反思引用真實數字但因果錯誤 | 驗證器只能檢查數字，不能檢查因果 | 預先登記預測 + refuted_streak 停止條件（以結果檢驗因果） |
| ideate 反覆提相同 family | lessons 被 12 條上限擠掉 | tried_families 用 reducer 完整保留，不受上限影響；同一 family ≥2 次即禁用（確定性檢查） |
| recursion_limit 誤觸 | 子圖繼承上限（F10） | 設 200；業務上限遠低於此值（每層 < 60 步） |
| 沙盒逾時沒被 LangGraph 攔下 | sync node 不支援 timeout（F14） | 容器自己的 timeout + OOM kill → 視為 E_TIMEOUT/E_OOM |
| checkpoint 寫入量大 | 三層每個 superstep 都寫一次 | Postgres 可承受；如有需要，把 L1 改成 `checkpointer=False`（代價：看不到 L1 內部狀態，崩潰時整段 debug 重跑，成本 ≤ $0.40） |

---

## 12. 依評分標準 R1–R7 自評

| 項目 | 權重 | 分數 | 理由（含扣分點） |
|---|---|---|---|
| R1 樣本外有效性 / 過擬合控制 | 25% | **8** | 分區存取由 DataGateway 依 namespace 在基礎設施層強制；OOS/lockbox 一次性許可；lockbox 在拓撲上無回流邊；滿足即停減少試驗數；DSR 的 N 取自完整 ledger；PBO；t 門檻隨 OOS 使用次數提高；OOS 失敗只回傳類別。扣分：6 個 idea 共用同一段 OOS 仍有選擇偏誤；lockbox 只有 12 個月、單一 regime 的風險無法在圖內解決。 |
| R2 反思品質 | 15% | **7** | 證據數字對帳、預先登記可證偽的預測、一輪只改一處、tabu、refuted_streak，命中率可以量測。扣分：驗證器只能確認數字，無法確認因果；一次只改一處使收斂較慢；DiagnosticPack 的切片維度是人定的，反思看不到切片之外的結構。 |
| R3 除錯效率與穩健性 | 10% | **8** | 分類 + 正規化 hash + 三段升級（Haiku patch → Sonnet 先解釋再補丁 → 依 spec 重寫）；前視兩次即放棄；infra 錯誤用 RetryPolicy 且不浪費 LLM；錯誤歷史隔離在 L1。扣分：hash 正規化規則需要實際資料調校；截斷不變性只抽 20 個 t，有漏檢機率。 |
| R4 探索能力 | 10% | **5** | 有 3 候選 + novelty guard + family 配額 + lessons，但外層是序列的，而且最多 6 個 idea；沒有族群或平行搜尋，容易被早期 family 錨定。`Send` 擴充可以改善，但不在 MVP 範圍。 |
| R5 成本與可控性 | 10% | **8** | 三層 allowance 由上而下切、由下而上回收；最壞情況有上限（約 $9.7）；每層 router 都檢查預算；便宜的模型負責修錯。扣分：路由是事後檢查，要靠呼叫前估算才能把超支壓小；thinking token 的實際用量要上線後量測校準。 |
| R6 實作複雜度 / 可維護性 | 15% | **6** | 三個圖的骨架已實跑（558 行 stub）；層層可獨立測試，符合「可分工維護」。扣分：schema 與轉接 node 的樣板多；巢狀 checkpoint 與 resume 語意（F3/F4/F7/F9）有多個不直覺的坑，新手容易踩；1–2 週做出 MVP 可行的前提是回測引擎與沙盒已經存在，否則偏緊。 |
| R7 可觀測性 / 重現 / HITL / 安全 | 15% | **8** | namespace 天然分層稽核；`get_state(subgraphs=True)` 與分層 stream；每次 LLM 呼叫都有完整紀錄；data_manifest 記下 hash、引擎版本與常數；HITL 只在主圖、前面無副作用、resume 值由 `response_schema` 驗證；沙盒隔離；秘密不進 state/context 以外的地方。扣分：跨層追一條「某個 lesson 從哪裡來」需要串接三個 namespace 的查詢工具（需另外寫）。 |

**加權總分 ≈ 0.25×8 + 0.15×7 + 0.10×8 + 0.10×5 + 0.10×8 + 0.15×6 + 0.15×8 = 7.25 / 10**

---

### 附錄：驗證用腳本（scratchpad，不屬於交付物）
- `lgsrc/`：langgraph 1.2.12 wheel 解開後的原始碼
- `sgtest.py`、`sgtest2.py`、`sgtest3.py`、`nstest.py`：驗證 F1–F13 的實驗
- `proto/arch_b.py`（= `team/candidate_B_skeleton.py`）、`proto/test_crash.py`：三層原型、崩潰路徑與預算路徑
