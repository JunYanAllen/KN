# 共同需求（所有候選架構都必須滿足）

## 使用者需求
- 以 LangGraph 為主框架（StateGraph + TypedDict State + add_node + add_conditional_edges + START/END + compile/invoke）。
- 核心模式：「自我反思（Self-Reflection）+ 自我除錯（Self-Debugging）」工作流，用來迭代優化加密貨幣量化交易策略（例：BTC/USDT 永續合約，15m~1h K 線）。
- 目標：產出「樣本外仍有效」的策略，而不是 IS 指標漂亮的策略。

## 前一輪審查已確立的硬性約束（必須納入）
1. 回測引擎固定、確定性、由人審過；LLM 只產生 `generate_signals(df, params) -> pd.Series`（目標部位 [-1,1]）+ PARAM_SPACE 宣告（或 DSL）。引擎負責 shift 1 根、t+1 開盤成交、taker 手續費、滑價、資金費率、年化 365 天。
2. 前視偏差防護在框架層：AST 靜態掃描、schema 驗證、截斷不變性測試（df[:t] 與全量資料在 t 的訊號一致）。
3. 三層資料紀律：迴圈內只用 IS（walk-forward / purged CV）；每變體最多 1 次 OOS；lockbox 只在人工審閱前用 1 次，結果不回饋 LLM。
4. 全域試驗帳本：記錄所有績效變體（修語法不算試驗），用於 Deflated Sharpe Ratio / PBO 多重檢定修正。
5. 是否達標由程式（routing function）判定，不由 LLM 判定；LLM Evaluator 只輸出結構化 critique / 假說。
6. 三種迴圈分開計數：執行錯誤修復（不經 Evaluator）、參數/邏輯改良、策略重新發想。另有 token / USD / wall-clock / 回測次數預算上限；錯誤 hash 重複偵測。
7. State 只存可序列化的小資料；大型資料（K 線、equity、交易明細）存 artifact store，state 存 ID/路徑。attempts 用 reducer 累積，lessons 摘要化控制 token。
8. 沙盒：每次拋棄式容器（--network none、read-only、非 root、資源限制），不含任何秘密；import 白名單只是前置過濾。
9. HITL：interrupt() 放在獨立 human_review 節點（前面無副作用），需 checkpointer + thread_id，Command(resume=...) 恢復。
10. LLM 輸出一律結構化（JSON schema / tool use / with_structured_output）。
11. 可觀測、可重現：prompt 版本、模型 ID、code hash、資料快照 hash、引擎版本、指標、成本。

## LangGraph API 參考（前一輪直接讀 langgraph 1.2.12 wheel 原始碼確認）
- StateGraph(State), add_node(name, fn, retry_policy=..., cache_policy=..., ...), add_edge, add_conditional_edges(src, router, path_map)
- from langgraph.types import Send, Command, interrupt, RetryPolicy；Overwrite 可覆寫 reducer 欄位
- Send 可做 map-reduce 扇出；subgraph 可作為 node；checkpointer: InMemorySaver / SqliteSaver / PostgresSaver
- 預設 recursion_limit 10007（只當最後保險）
- 可於 PyPI 下載 wheel 讀原始碼查證（pip download langgraph --no-deps）；網頁文件可能被代理擋住。

## 評分標準（評審會用）
R1 樣本外有效性 / 過擬合控制（權重 25%）
R2 反思品質：反思是否真的帶來「有資訊量」的改進，而非事後編故事或原地打轉（15%）
R3 除錯效率與穩健性（10%）
R4 探索能力：能否跳出局部最佳（10%）
R5 成本與可控性（token/USD，預算）（10%）
R6 實作複雜度與可維護性（MVP 1–2 週可做出來？）（15%）
R7 可觀測性 / 可重現性 / HITL / 安全（15%）
