# LangGraph 加密貨幣量化策略代理：多角度審查報告

> 審查方式：四位審查代理平行、獨立評估同一份提案，再整合結論。
> 1. 量化方法論（回測、統計檢定、過擬合）
> 2. LangGraph 圖結構與工程架構（API 以 langgraph 1.2.12 原始碼查證）
> 3. 沙盒與營運安全
> 4. LLMOps 成本／可靠度，兼魔鬼代言人
>
> 審查對象：「Strategist → Coder → Backtest → Evaluator → 未達標回 Coder（≤5 次）→ 報告」的 LangGraph 迴圈提案。

---

## 1. 總結判定

**工程上可行，但照目前的設計，方法論與安全風險都偏高，不宜直接實作。**

四份審查各自得出相同的核心結論：

> 這個迴圈目前優化的是「讓 In-Sample 指標過門檻」，而不是「找到樣本外仍有效的策略」。
> 在 IS 上「達標即停」，加上 LLM 每次先看 IS 結果再改策略，整套系統等於一台**自動化的過擬合產生器**：
> 很容易產出 Sharpe > 1.5 的報告，但那個數字不可信。

提案的骨架可以保留：節點分工、條件迴圈、interrupt 人工審閱、IS/OOS 切分的想法。需要重做的是以下四件事：

| 面向 | 提案現況 | 需要改成 |
|---|---|---|
| 回測引擎歸屬 | LLM 寫整份回測程式 | **引擎固定、由人審過**；LLM 只寫 `generate_signals()` |
| 停止條件 | Sharpe>1.5、MaxDD<15%（IS） | **統計閘門**：DSR／PBO、最低交易數、成本壓力、穩健性 |
| 資料紀律 | IS 過了才看 OOS，失敗可能回迴圈 | **三層切分**：walk-forward（迴圈用）／OOS／lockbox（只用一次） |
| 沙盒 | Docker **或** import 白名單 | **只能用容器／microVM 隔離**；白名單最多當前置過濾 |

---

## 2. 共識發現（依嚴重度）

### 🔴 高

**H1. 達標即停造成資料窺探與多重檢定**（四份審查都指出）
- 真實的試驗次數 N 遠大於 5：使用者會重跑整張圖、會跑多個想法，LLM 也會在腦中隱性試很多組。
- 「prompt 禁止暴力調參」無法強制執行。把 ATR 倍數從 2 改成 2.5 本身就是調參；「改邏輯」只是在自由度更大的空間裡調參（Goodhart 效應）。
- **對策**
  - 建一本**跨 run 的全域試驗帳本**，記下每個變體的報酬序列。語法修復不算試驗，績效改版一定要算。
  - 停止門檻改用 **Deflated Sharpe Ratio > 0.95**。累積 N ≥ 20 後，以 CSCV 計算 **PBO < 0.2–0.3**。策略家族之間的比較用 White's Reality Check 或 Hansen SPA。

**H2. OOS 會被污染**
- OOS 失敗後只要回到迴圈，OOS 就變成第二個 IS。
- **對策：三層資料紀律**
  1. 迴圈內只用 IS 內部的 walk-forward 或 CPCV，折與折之間要做 purging 和 embargo。
  2. 每個變體最多跑 1 次 OOS。
  3. **Lockbox**（最近 6–12 個月）放在 LLM 碰不到的地方，只在人工審閱前打開一次。失敗就淘汰整個假說家族，結果**絕不回饋給 LLM**。
  4. 過了 lockbox 之後，以 paper trading 當作真正的前推 OOS。

**H3. LLM 自己寫回測程式：前視偏差、偽造結果**
- 常見的前視錯誤型態：
  - 用第 t 根收盤價算出訊號，又在同一根收盤成交
  - `shift(-1)`、`rolling(center=True)`、`bfill`
  - 用全樣本做 z-score 或正規化
  - resample 時 label/closed 對齊錯，高週期 K 線還沒收完就拿來用
  - `merge_asof` 方向寫錯
  - 會重繪的指標（fractal、zigzag）
  - VectorBT 預設在收盤價成交
- 更嚴重的是，生成的程式可以**自己算、自己回報**漂亮的 Sharpe。
- **對策：框架層強制，不靠 prompt**
  - LLM 只能產生純函式 `generate_signals(df, params) -> pd.Series`（目標部位落在 [-1, 1]），外加一份 `PARAM_SPACE` 宣告。
  - 固定的引擎統一處理：訊號 shift 1 根、於 t+1 開盤成交、手續費、滑價、資金費率、年化（加密市場用 365 天）。
  - 三道驗證：
    - AST 靜態掃描，擋掉上面列的寫法；
    - Schema 檢查：index 必須對齊，值域正確，warmup 之後不能有 NaN；
    - **截斷不變性測試**：`df[:t]` 與完整資料在第 t 根算出的訊號必須相同。
  - Sharpe 大於 4 之類不合理的數字，自動標記為可疑。
- 架構審查判斷：這一項對可靠度的提升，**比換任何框架都大**。

**H4. 由 LLM 判定「達標」／Evaluator 自評偏誤**
- 出題、作答、改考卷都是同一個模型家族，判斷也可能前後不一致。
- **對策**
  - 門檻寫死在 routing function，由程式判定。Evaluator 只輸出結構化的 critique 和假說，不參與路由。
  - Evaluator 改用不同的模型，temperature 設 0。

**H5. import 白名單不是安全邊界**
- 可以繞過白名單的方式：
  - `().__class__.__base__.__subclasses__()`、`__import__`、`importlib`、`eval`/`exec`
  - pandas 能讀寫任意檔案或 URL：`read_csv('/proc/self/environ')`、`read_html('http://…')`
  - pickle 反序列化：`read_pickle`、`numpy.load(allow_pickle=True)`
  - `ctypes` 直接呼叫 libc
- LangChain 的 `PythonREPLTool` 在 agent **同一個 process** 裡 `exec`，讀得到 LLM API Key 等環境變數，等於沒有沙盒。
- **對策**：每次執行開一個新的拋棄式容器，最好搭配 gVisor 或 Firecracker；不接網路、檔案系統唯讀、非 root、限制資源、不注入任何秘密。範例見 §5。

**H6. 加密貨幣成本與執行模型過於簡化**
- **資金費率**：用歷史實際費率，按結算時點（多半是 UTC 00/08/16，部分合約是 1h 或 4h）判斷持倉並計算，正負方向要對。
- **手續費**：市價單與停損單一律按 taker 計；限價單不能假設一定成交。
- **滑價**：隨波動度與價差放大。15m 策略換手率高，須做 **2 倍成本壓力測試**。
- **K 線內順序不明**：同一根 K 線同時碰到停損與止盈時，一律假設先停損，或改用 1m 資料判斷。
- **強平與插針**：強平要用 mark price 和分級維持保證金計算。
- **資料品質**：處理缺漏和重複的 K 線與停機期間；統一時間戳是 open time 還是 close time；全部用 UTC；不能把現貨和永續資料混用；擴充到其他幣種時要把已下市的幣也放進樣本。

**H7. 交易所 API 與上線風控**
- 研發環境不存在任何交易所 Key。
- Key 設定：禁止提領、綁 IP 白名單、用子帳號、設資金上限，存在 Vault/KMS 並定期輪替。
- kill switch、最大虧損熔斷、倉位與槓桿上限，全部由**非 LLM 的確定性程式**實作，放在策略 process 之外。

**H8. State 沒有歷史記憶**
- State 只存當前這一次的結果，LLM 會繞圈子、重複嘗試已經試過的東西。
- **對策**：用 reducer 累積 `attempts`，加上摘要化的 `lessons`，並依 code hash 去重。

### 🟠 中

- **M1. 迴圈拓撲太粗**：三種失敗共用一條「回 Coder」的迴圈。
  - 語法錯誤應直接回 Coder，不必經過 Evaluator。
  - 策略本身不行時要能回 Strategist。
  - 三層迴圈各自設計數器，見 §3。
- **M2. `MAX=5` 把修復與改良混在一起算**：拆成 `fix_count ≤ 3`（用便宜模型）與 `refine_count ≤ 3`。
- **M3. 缺預算上限**：每個想法設 token、金額、wall-clock 上限，每天再設全域上限；超出時產出「預算耗盡」報告後結束。`recursion_limit`（1.2.12 預設為 10007）只能當最後的保險。
- **M4. 同一個錯誤打轉**：把錯誤訊息正規化後連同程式碼算 hash，連續 2 次相同就中止。
- **M5. 自由文字解析**：一律用 tool use／JSON schema。
  - Coder 回傳 `{code, entrypoint, params}`。
  - Evaluator 回傳 `{verdict: enum, failure_type: enum, suggestions[]}`。
- **M6. 大型資料放進 state**：K 線、equity curve、交易明細放 artifact store，state 只存 ID 或路徑，否則每個 checkpoint 都會暴增。
- **M7. 長時間回測與 interrupt 的恢復語意**
  - checkpoint 只在 superstep 邊界寫入，而恢復時整個 node 會從頭重跑。
  - 回測改成外部 job，以 `(code_hash, params, data_version)` 當 key 快取，達到冪等。
  - `interrupt()` 放在獨立的 `human_review` node，前面不能有副作用。
- **M8. 單線序列迭代容易卡在局部最佳**：用 `Send` 一次展開 N 個變體，最後保留 top-k（beam search）。代價是試驗數變多，所以 H1 的多重檢定修正是必要的。
- **M9. 評估指標太單薄**
  - 交易次數：IS 至少 200 筆，每一折至少 30 筆。
  - PSR 要考慮自相關。
  - 分行情狀態、分年度檢驗。
  - 參數在 ±20% 範圍內要落在高原，不能是尖峰。
  - 跨資產、跨週期都要有效，多空要分開看。
  - 要對 BTC 做 beta 分解。
  - 要和 buy & hold、隨機訊號的基準比。
  - 用 bootstrap 看 MaxDD 的分布。
- **M10. 橡皮圖章式人工核准**
  - 審閱包必須包含：程式碼 diff、回測假設、交易明細抽樣、IS/OOS 分開呈現、總嘗試次數與失敗版本摘要。
  - 審閱者不能是發起人；核准要寫理由；高風險變更需要兩人核准。
- **M11. 可重現性與可觀測性**
  - 每次 run 記錄：prompt 版本、模型 ID、temperature/seed、程式碼 hash、資料快照 hash、映像檔 digest、引擎版本、指標、成本。
  - 工具：LangSmith 或 Langfuse 追 trace，MLflow 做實驗追蹤。
- **M12. 供應鏈**：執行期禁止 `pip install`，映像檔預先建好、版本鎖定並以 digest 引用。
- **M13. 外部資料的 prompt injection**：若會抓取新聞或社群內容，只以「資料」角色傳入；Strategist 沒有任何工具權限；抓取器與沙盒分開。

---

## 3. 修正後架構

```
START -> load_data（確定性：切 IS / OOS / lockbox，產生 data_version）
      -> strategist ──Send×N──┐             <── 迴圈 C：策略重新發想（round ≤ 3）
                              v
   ┌──────────── variant 子圖（每個變體各一份） ─────────────────┐
   │ coder -> validate(AST / schema / 截斷不變性) -> backtest_IS  │
   │   ^            │ fail                        │ 執行錯誤      │
   │   └────────────┴─────────────────────────────┘               │
   │        迴圈 A：執行錯誤修復（fix ≤ 3，不經 Evaluator）         │
   │                          │ 取得指標                           │
   │                   gate_IS（程式規則判斷）                     │
   │     接近門檻 -> evaluator(critique) -> coder                  │
   │        迴圈 B：參數／邏輯改良（refine ≤ 3，參數空間已宣告）    │
   │     通過 -> backtest_OOS（每個變體 1 次）                     │
   └──────────────────────────┬──────────────────────────────────┘
                              v （reducer 合併 attempts，更新試驗帳本）
                       select_topk（規則判斷）
        全部失敗 / 遠離門檻 / 次數用盡 ──> strategist（迴圈 C）
        有變體通過 -> robustness（walk-forward、2× 成本、參數高原、跨資產）
                   -> 統計閘門（DSR / PBO）
                   -> lockbox（只跑 1 次，結果不回饋 LLM）
                   -> human_review [interrupt()] -> 核准 -> report -> END
                                                -> 駁回 -> strategist
   全域預算：回測總數、USD、wall-clock 上限；達上限 -> 失敗報告 -> END
```

### State schema 範例

```python
import operator
from typing import Annotated, Literal, TypedDict

class Metrics(TypedDict):
    sharpe: float; max_dd: float; win_rate: float; pf: float; n_trades: int

class Attempt(TypedDict):
    variant_id: str; round: int; stage: Literal["fix", "refine", "new"]
    idea: str; code_hash: str; params: dict
    status: Literal["invalid", "runtime_error", "is_fail", "is_pass", "oos_fail", "oos_pass"]
    is_metrics: Metrics | None; oos_metrics: Metrics | None
    error_summary: str | None      # 截短後的 traceback（完整版存 artifact）
    critique: str | None
    artifacts: dict[str, str]      # {"code": "s3://…", "equity": "…", "trades": "…"}

class ResearchState(TypedDict):
    thread_meta: dict              # symbol, timeframe, data_version
    data_refs: dict[str, str]      # IS / OOS / lockbox 路徑，不放 DataFrame
    attempts: Annotated[list[Attempt], operator.add]  # 只增不改，Send 扇入時合併
    lessons: list[str]             # 摘要化歷史，由 summarizer 覆寫
    seen_hashes: Annotated[set[str], operator.or_]    # 去重
    round: int                     # 迴圈 C 計數
    budget: dict                   # {"backtests_left": 60, "usd_left": 20.0}
    n_trials_total: int            # 給 Deflated Sharpe 用
    candidates: list[str]          # top-k variant_id
    review: dict | None            # interrupt 的人工回覆
```

variant 子圖用私有 state 保存 `fix_count`、`refine_count` 與當前的程式碼參照。Prompt 只注入最近 k 筆 attempts 加上 `lessons`，控制 token 用量。

### 框架選擇

- **外層用 LangGraph** 管流程，值得的理由：
  - checkpointer 可恢復執行；
  - `interrupt()` 做人工審閱；
  - `Send` 平行扇出，搭配 reducer 合併結果；
  - time travel；
  - LangSmith 追蹤；
  - 1.2 版 `add_node` 的 `retry_policy`、`timeout`、`cache_policy`。
- **Coder 節點內層**可以用 agent，例如 Claude Agent SDK，讓它自己寫、自己跑驗證。
- **MVP 也可以先用純 Python while-loop**，約 150 行；需要 HITL 和恢復執行時再遷移。
- 注意：time travel 重放會重新呼叫 LLM，結果不會一樣。比較不同分支要看試驗帳本，不要靠重放。

---

## 4. 成本量級（假設值，請換成實際報價）

**單價假設**：強模型輸入 $3、輸出 $15／百萬 token；便宜模型輸入 $0.8、輸出 $4；不開 extended thinking。

| 項目 | 估算 |
|---|---|
| 每輪（Strategist 7k + Coder 10k + Evaluator 11k token） | 約 $0.13 |
| 每個想法（5 輪，歷史 context 逐輪增加） | 約 $0.75（開 thinking 約 $1.5–2.5） |
| 每日 50 個想法 | 約 $40–120 |
| 每日 500 個想法 | 約 $400–1,200 |
| 套用下列節省手段後 | 可省 50–70%，每個想法約 $0.25–0.35 |

**節省手段**：
- 對系統提示、資料描述、引擎 API 說明做 prompt caching；
- 語法修復改用便宜模型，而且不經過 Evaluator；
- 歷史摘要化，固定在 1–2k token；
- 先用短期資料做 smoke test，把明顯不行的想法提早剔除。

---

## 5. 沙盒參考設定

```bash
RUN=/runs/$RUN_ID
timeout 300 docker run --rm --name bt-$RUN_ID \
  --runtime=runsc \
  --network none \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --user 65534:65534 \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 64 --memory 1g --memory-swap 1g --cpus 1 \
  --ulimit nofile=128:128 --ulimit fsize=52428800 \
  -v /data/ohlcv_snapshot_2026Q3:/data:ro \
  -v $RUN/strategy.py:/work/strategy.py:ro \
  -v $RUN/out:/out:rw \
  backtest-sandbox@sha256:<digest> \
  python -I /work/strategy.py
# 容器只輸出 signals.parquet；PnL 與各項指標由沙盒外、受信任的引擎計算。
# 逾時由 host 端 docker kill 強制結束，不要只依賴 timeout 指令。
```

---

## 6. 魔鬼代言人：結構性質疑與替代設計

1. **昂貴的隨機搜尋**：LLM 產出的多半是 RSI、MA、布林通道、ATR 停損這類公開指標的排列組合。BTC 15m 純 OHLCV 策略扣完成本後，剩下的 alpha 很薄。同樣的結果，「隨機模板 + Optuna」可能只要 1% 的成本。
2. **alpha 來源錯位**：加密市場的優勢多半來自資料（資金費率極端值、基差、OI、強平流、訂單簿、鏈上）和執行微結構。只餵 K 線的架構碰不到這些來源。
3. **事後編故事**：LLM 先看了 IS 結果才做「反思」，這會放大過擬合。對策：假說要先登記，包含經濟邏輯與預期的效果方向，之後才能看資料。

**建議的混合設計**
- LLM 負責產出**可證偽的假說**，並用**策略 DSL 積木**組出策略（訊號、濾網、部位、風控）。
- 確定性引擎把 DSL 編譯後執行，寫碼錯誤接近零，自由度也可以枚舉。
- 參數搜尋交給 Optuna，在 IS 內做 walk-forward。
- 是否通過由程式判定（DSR、PBO、White's Reality Check）。
- LLM 只寫診斷和下一個假說。只有 DSL 表達不了的新特徵才讓 LLM 寫自由程式碼，而且要人工審閱。

---

## 7. MVP 建議（1–2 週）

**要做**
- 一份附 hash 的資料快照（例如 BTC/ETH 1h，2019–2025），搭配固定的手續費、滑價與資金費率模型。
- 確定性回測器，由框架控制資料存取，並內建截斷不變性測試。
- 10–15 種 DSL 積木加 JSON schema；LLM 輸出 DSL，不輸出 Python。
- LangGraph 流程：Strategist → 驗證 → 回測 → 程式判定 → LLM 診斷 → 回 Strategist；`refine ≤ 3`，加上三種預算上限與 hash 重複偵測。
- 最終 holdout 只跑一次，用累計試驗次數計算 DSR。
- 用 Langfuse 或 LangSmith 追 trace，MLflow 記錄實驗，checkpointer 支援回放。
- **系統自身的 benchmark**：
  - 20 個已知策略（有效的，例如動量；無效的，例如隨機訊號）；
  - 10 個刻意植入前視偏差或漏算手續費的樣本，量測 Evaluator 的抓出率；
  - 基線為同預算下的「隨機 DSL + Optuna」。
- **停損標準**：如果 OOS 表現沒有顯著勝過隨機基線，就停止這個方向。
- 系統好壞看 OOS Sharpe 分布、IS→OOS 衰退比例、每產出一個 OOS 有效策略的成本；**IS 達標率不能當指標**。

**明確不做**：交易所 API／實盤／下單、自由 Python 程式碼生成、多幣種與多週期、鏈上與訂單簿資料、通知與報告美化、大批量排程。

---

## 8. 上線前閘門清單（未來階段）

1. 紅隊實測沙盒繞過手法，確認全部被擋，沙盒內沒有網路、看不到任何秘密。
2. 程式碼凍結：核准的是某個 commit hash，之後任何修改都要重新走完整流程。
3. 人工審查包完整（見 M10），審閱者與開發者不是同一人。
4. 統計閘門：OOS 與 lockbox 達標，DSR/PBO 達標，壓力情境測試通過。
5. Paper trading 至少 2–4 週，實際表現與回測的偏差在容忍範圍內。
6. 小資金灰度上線，資金上限分階段放大，每一階段都要人工核准。
7. 上線前實際觸發一次 kill switch、熔斷與倉位上限，確認都有效。
8. 確認 API Key 已設定禁止提領、IP 白名單、子帳號與額度上限。
9. 稽核紀錄完整，可以從頭重跑。
10. 監控告警、值班 runbook、回滾流程都已就位。
11. 依所在地法規確認登記或申報義務（例如台灣的 VASP 規範）。

---

## 9. 若只能改三件事

1. **回測引擎由人掌控**：LLM 只產生 signal 函式或 DSL；截斷不變性與 AST 前視檢查自動執行；成本模型寫實，包含資金費率、taker 手續費、滑價、K 線內先停損。
2. **三層資料紀律加全域試驗帳本**：迴圈只用 walk-forward；lockbox 對 LLM 完全隔離，只用一次，失敗不回迴圈。
3. **停止條件改成程式判定的統計閘門**：用 DSR/PBO、最低交易數、成本壓力、參數高原、行情狀態與跨資產穩健性，取代「Sharpe>1.5、MaxDD<15%」。
