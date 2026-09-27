# LangGraph 自我反思＋自我除錯：交易策略優化的最佳架構

> 前一份文件是 [`quant-strategy-agent-evaluation.md`](quant-strategy-agent-evaluation.md)，它審查了原始提案。
> 本文用 agent team 找出一套最符合需求的架構，並附上可執行的參考實作 [`quant-agent/`](../quant-agent/)。

## 0. 結論先講

**採用「強化版 Reflexion 單線迴圈」當主幹，再從另外三種架構各取幾個經過驗證的元件。**
主幹是單一個 `StateGraph`，只用條件邊，沒有子圖、沒有 Supervisor、也沒有演化族群。

兩位評審各自獨立評分，都選了同一個主幹，建議嫁接的元件也高度重疊。原因可以歸納成一句話：

> 在 BTC 1h 這種 alpha 很稀薄的市場，瓶頸是**統計檢定力**，不是 LLM 的創意。
> 架構的第一目標應該是「每個想法都驗得乾淨，同時把有效試驗數壓低」，而不是一次試更多想法。

---

## 1. Agent team 的做法

```
            ┌─ 架構師 A：強化版 Reflexion 單線迴圈 ─┐
 共同需求 ──┼─ 架構師 B：階層式巢狀子圖（3 層）  ─┼─→ 評審 1：量化研究主管（統計/方法論） ─┐
 + 評分標準 ┼─ 架構師 C：族群演化（Send 扇出）    ─┤                                        ├─→ 整合 → 參考實作 → 對抗式 code review
            └─ 架構師 D：Supervisor + 多批判者    ─┘→ 評審 2：工程主管（實跑骨架/查證 API） ─┘
```

- **架構師（4 位）**：每位各自提出完整設計，內容包含架構圖、State schema、路由程式碼、反思與除錯機制、成本估算和自評。B、C、D 三位另外附上可執行骨架，已在 langgraph 1.2.12 上跑過。
- **評審（2 位）**：不看架構師的自評，自己重新打分。
  - 量化評審做了檢定力的數值計算。
  - 工程評審實際執行骨架，逐條檢查架構師聲稱的 LangGraph 行為是否屬實。
- **評分標準**：R1 樣本外有效性 25%｜R2 反思品質 15%｜R3 除錯 10%｜R4 探索 10%｜R5 成本 10%｜R6 實作複雜度 15%｜R7 可觀測／HITL／安全 15%。

### 評分結果（加權總分，滿分 10）

| 候選 | 架構師自評 | 量化評審 | 工程評審 | 評審平均 |
|---|---|---|---|---|
| **A 強化 Reflexion 單線** | 7.05 | **6.88** | **7.10** | **6.99** |
| B 階層巢狀子圖 | 7.25 | 6.60 | 6.35 | 6.48 |
| C 族群演化 | 7.05 | 6.48 | 6.45 | 6.47 |
| D Supervisor 多批判者 | 6.80 | 5.75 | 5.85 | 5.80 |

四份設計的 R1（過擬合控制）沒有一份拿到 7 分以上：每份至少都有一條 OOS 資訊洩漏路徑，或是試驗計數算錯。

---

## 2. 關鍵發現

### 2.1 統計現實（量化評審實算）

| 量 | 數值 | 對設計的意義 |
|---|---|---|
| 1 年 OOS 的年化 Sharpe 標準誤 | ≈ 1.0 | 以「OOS SR ≥ 0.5」為門檻時，純雜訊每看一次有 31% 會過；看 6 次有 89% |
| 3 個月 fold 的 SR 標準誤 | ≈ 2.0 | 逐 fold 設門檻，量到的基本上是雜訊 |
| 真實 SR=1.5 的策略通過 DSR 的機率 | N=10 時 49%、N=100 時 16%、N=320 時 8.5% | 族群演化需要找到真實 SR 高出約 0.4 的策略，才能打平 DSR 的懲罰 |

**因此：** 單線、低試驗數（N_eff ≤ 20）是最合理的配置。C 的族群演化在數學上等於「保證找不到」。
C 的架構師自己實測，也得到 `no_finalist_passed`。

### 2.2 LangGraph 行為查證（工程評審與架構師實測）

| 行為 | 結論 |
|---|---|
| 父子圖共用一個帶 reducer 的鍵 | **會重複累加**，例如 `['p0','p0','child']`。子圖需要設 `output_schema` |
| `error_handler` 回傳 dict | 圖會**默默停住**，必須回傳 `Command(goto=...)` |
| `add_node(destinations=...)` | 只影響畫圖，runtime 不檢查 |
| `RetryPolicy` 的預設 `retry_on` | 會連 `AssertionError`、預算例外、HTTP 400 都重試 |
| `interrupt()` 恢復 | 整個 node 會從頭重跑，所以 interrupt 前面不能有副作用 |
| `hash()` 當亂數種子 | 每個進程結果不同，無法重現 |
| 平行扇出時在 worker 內不檢查預算 | 實測超支 39% |

本實作的對策是**避開這些陷阱**：
- 不用子圖、不用 `Command` 路由、不用 `destinations` 當防護。
- 例外在 node 包裝層轉成 `halt_reason`，不依賴 `error_handler`。
- 種子一律用 sha256 算出。

---

## 3. 最終架構

```
START → init_run（公布門檻曲線）→ strategist ──→ coder → static_check → sandbox_validate → backtest_is → gate_is
                                     ▲              ▲        │  ▲            │                               │
                                     │              │        ▼  │            ▼                               │
                                     │              │     debugger ◄─────────┘  迴圈 A：除錯 ≤4 次；       │
                                     │              │        │                   同一錯誤 hash 出現 3 次即放棄│
                                     │              │        └→ abandon_variant                              │
                                     │              └─ reflect / control_change ◄────────────────────────────┤
                                     │                 迴圈 B：改良 ≤3 次；達標即停；連續 2 次沒進步即停       │
                                     └──── close_idea ◄──────────────────────────────────────────────────────┘
                                           迴圈 C：想法 ≤6 個
                                              │
                                              ▼
            finalize（純程式）：用最終 N 重新折算 DSR → shadow 段 → 事先登記順序的決賽者（≤2）
                               → 依序看 OOS（單尾檢定 SR>0，α 以 Bonferroni 分給各決賽者），第一位通過即停
                 ├─ 有冠軍 → lockbox_eval（每個快照 1 次）→ human_review [interrupt] → report → END
                 └─ 沒有冠軍 ──────────────────────────────────────────────────────→ report → END
```

### 3.1 從各候選嫁接的元件

| 元件 | 來源 | 實作位置 |
|---|---|---|
| 假說預登記（寫入帳本並算 hash）、lesson 以支持／反駁計分、每條 lesson 只能改一處 | A | `strategist`、`gate_is`、`close_idea` |
| 除錯升級階梯：便宜模型修補 → 強模型先解釋再修 → 重新生成 | B | `debugger` |
| 「進步」必須大於雜訊帶：配對 block bootstrap，Δ > max(0.10, 2·SE) | B（修正了評審指出的標準誤錯誤） | `gate_is`、`stats.paired_sharpe_delta` |
| 滿足即停；參數 ≤4 個、網格 ≤16 組 | B | routers、`ast_rules` |
| `interrupt` 回覆用 schema 驗證，不合格就重新詢問 | B | `human_review` |
| 呼叫前預算估算，已付費的 token 仍然記帳 | B | `llm.LLMClient.call` |
| 決賽者在看 OOS **之前**就寫入帳本，順序凍結；OOS 對 LLM 是 0 bit | C | `finalize`、`ledger.register_finalists` |
| 反思增益對照組：每 5 次改良中有 1 次是無引導改動 | C | `control_change`、report 的 `reflection_lift` |
| run 開始前公布門檻曲線 E[max SR \| N] | C | `init_run` |
| shadow 驗證段：IS 尾端，LLM 看不到 | C（量化評審建議提升進 MVP） | `data.Splits`、`finalize` |
| LLM 可見欄位白名單加上 canary 斷言 | D | `context.py` |
| 證據斷言 `(ref, op, value)` 由程式核對 | D | `nodes.validate_reflection` |
| 固定的確定性壓力檢查：fee×2、delay 1 根、參數高原、分段／分波動 | D | `backtest_is` |
| 除錯不得改動 PARAM_SPACE；和帳本內程式碼完全相同就不算新試驗 | D＋評審 | `static_check`、`sandbox_validate` |

### 3.2 明確不採用的元件

| 元件 | 不採用的理由 |
|---|---|
| C 的島嶼演化（crossover、遷移、20 代 × 16） | 試驗數暴增，DSR 門檻被墊高，淨效益為負 |
| D 的 LLM Supervisor | 本質上是 2–5 類的分類器，邊際價值沒有證實，又多一個不可重現的自由度 |
| D 的三位批判者、HMAC token、`assert` 型 guard | 意見相關性高、和閘門重複、在同一進程內沒有實際安全邊界、`-O` 下會失效 |
| B 的三層巢狀子圖 | 約 30% 是搬運樣板，還有 6 個 LangGraph 陷阱；用計數器就能達到同樣的隔離 |

---

## 4. 需求與實作元件對照

| 需求 | 實作方式 |
|---|---|
| 回測引擎固定，LLM 只寫訊號 | `engine.py`：訊號延後 1 根、次根開盤成交、taker 手續費加滑價、資金費率。LLM 只產生 `generate_signals(df, params)` 和 `PARAM_SPACE` |
| 前視防護放在框架層 | AST 靜態掃描（負 shift、`center=True`、`bfill`、I/O、模組逃逸、屬性指派／monkey-patch、日期字面值、dunder）＋**截斷不變性測試**：**每一組**參數都在兩個隨機切點和 n−1 上跑，最後才跑全量；比對在 **host 端**進行，而不是在不受信任的直譯器裡 |
| 三層資料紀律 | research（WF，唯一給 LLM 看的數字）→ shadow → OOS（依事先登記的順序，每人最多 1 次；每快照終身上限 3 次；以單尾 PSR 檢定，α 以 Bonferroni 分配）→ lockbox（每快照 1 次，只給人看；沙盒故障不會消耗額度） |
| 試驗帳本與多重檢定 | SQLite 帳本跨 run 共用；**變體**才算試驗，語法修復和重複程式碼不算；DSR 用同一組試驗的 N 和 V[SR]；finalize 時用最終 N 重新折算，並在報告揭露「實際需要的 IS Sharpe」和試驗離散度 |
| 是否達標由程式判定 | `passes_is_gate()` 和所有 router 都是純程式；門檻從不進 prompt |
| 三種迴圈分開計數，並有預算上限 | 除錯、改良、想法各自有上限；另有 USD、active time（不含人工等待）、每快照試驗數上限；超過就 `halt_reason` → finalize → report，不會崩潰；報告寫明 `stop_reason` |
| State 只放小資料 | 程式碼、訊號網格、PnL 存在內容定址的 artifact store，state 只存 ref；可以用 SqliteSaver 跨進程恢復（有測試） |
| 沙盒 | `SubprocessSandbox`（開發用：空環境變數、rlimit、逾時強殺；**不隔離網路**；接真實 LLM 時 `make_deps` 會拒絕使用，除非明確允許）；`DockerSandbox`（`--network none`、唯讀、非 root、cap-drop、gVisor；docker 本身故障歸類為 E_INFRA，不會丟給 LLM 除錯） |
| HITL | `human_review` 前面沒有副作用；審閱包包含程式碼、參數、IS/shadow/OOS/lockbox、試驗數、門檻曲線、成本假設 |
| 結構化輸出 | Pydantic schema；schema 失敗會帶錯誤重試 1 次；refusal 和連線錯誤（transient）分開處理 |
| 可觀測、可重現 | 事件 JSONL（每次 LLM 呼叫、每個節點、每次路由與原因）、設定指紋、prompt 版本、資料快照 hash |

---

## 5. 參考實作的驗證結果

離線執行（stub LLM ＋ 合成資料），langgraph 1.2.12，**38 個測試全部通過**：

- **除錯迴圈**：stub 刻意犯三種錯（NameError、`shift(-1)`、全樣本 z-score），全部被抓到並修好；其中全樣本 z-score 只有截斷不變性測試抓得到，AST 掃描看不出來。
- **誠實的結果**：在預設的稀薄 alpha 資料上，i1.v1 在第 2 次試驗時通過 IS 閘門（DSR 0.99）。等到 finalize 用最終 N=11 重新折算，它就被淘汰，最後沒有任何決賽者。報告揭露了原因：試驗間離散度是虛無假設下的 5.97 倍，實際需要 IS Sharpe 3.86 才能通過。這正是「早期通過其實是被灌水」的效應。
- **完整 HITL 路徑**：在帶有真實趨勢的合成資料上，會依序走完 IS → 重新折算 → shadow → OOS → lockbox → interrupt。回覆不合法時會重新詢問；也測過從 SQLite checkpoint 在新的 app 實例中恢復，且沒有重跑研究。
- **資訊隔離**：確認所有送進 LLM 的 payload 都不含 oos/lockbox/shadow/dsr 等鍵，也不含 OOS 的數值。
- **故障路徑**：預算耗盡時會正常收尾，花費不超過上限；沙盒基礎設施故障時會 halt，不會寫出錯誤的 lesson。
- **反思增益**：報告會輸出 reflect 組和 control 組的改善率。stub 的反思沒有贏過對照組，**這個數字正是設計要量的東西**。

### 對抗式 code review（第二輪 agent）與修正

另一個審查 agent 用腳本實際重現了 10 個問題，全部已修正並補上回歸測試：

| 嚴重度 | 問題 | 修正 |
|---|---|---|
| 高 | OOS 判準的標準誤單位錯誤，實際上退化成「SR>0 就通過」，純雜訊下每位決賽者有 50% 假陽性 | 改成單尾 PSR 檢定 SR>0，α=0.10 以 Bonferroni 分配；OOS 段拉長到 24% |
| 高 | 截斷不變性只抽查 3 組參數，`lag=-1` 可以混進去；比對在不受信任的直譯器內執行，可被 `np.isclose = …` 繞過 | 每一組參數都檢查；比對移到 host 端；AST 禁止屬性指派 |
| 高 | AST 黑名單有漏洞：`pd.read_table`、`tofile`、`pd.io.common.os.system` 等 | 只允許 `import numpy/pandas/math`；封鎖子模組鏈、I/O、`query`/`eval`/`random` |
| 高 | DockerSandbox 的暫存目錄權限是 0700，非 root 容器讀不到；docker 故障被當成程式錯誤丟給 LLM | 調整權限（755/644/777）；docker 的 125–127 退出碼歸類為 E_INFRA |
| 中 | 公布的門檻曲線只用虛無變異數，嚴重低估實際門檻 | 改用帳本中的實際 V[SR]，並在 finalize 揭露最終門檻 |
| 中 | 同一錯誤 2 次就放棄，升級階梯從未走到強模型 | 改成 3 次（便宜修補 → 強模型修補 → 放棄） |
| 中 | LLM 連線失敗、反思第 2 次呼叫中止時，已付的費用漏記 | 例外帶著累計花費一起拋出 |
| 中 | 預算或試驗上限觸發的停止，報告寫成 `halt_reason=None` | 新增 `stop_reason` |
| 低 | 參數值型別混雜會讓 run 崩潰；WF 在 fold 邊界換參數時沒算換倉成本；lockbox 遇到沙盒錯誤會燒掉額度 | 參數值型別驗證；補計邊界換倉成本；沙盒錯誤不寫入 lockbox |
| 待確認 | 資金費率應扣在哪個持倉 | 改為結算時點持有的倉位（進入 bar t 前的倉位） |

### 已知限制

- stub 不是真的研究員，換上真實 LLM 後的效果**尚未驗證**。`LangChainBackend` 已經寫好，但沒有接真實模型測試。
- `SubprocessSandbox` 不隔離網路和檔案系統；`DockerSandbox` 的指令已寫好，但這個環境沒有 Docker，所以沒測過。
- 合成資料只用來驗證流程。真實研究要換成交易所的 OHLCV 和歷史資金費率，並處理缺漏 K 線與時區對齊。
- 還沒做：N_eff 聚類（目前用 raw N，比較保守）、PBO/CSCV、Stat Auditor LLM 批判者、`Send` 廣度批次。這些是 v2 項目，見下一節。

---

## 6. 建議的試驗預算與路線圖

**預算配置（量化評審建議）：**
- 每個資料快照 8–10 個想法，每個想法最多改良 3 次，每個快照原始變體數 ≤40，N_eff ≤20。
- OOS 每個快照終身最多 3 次，lockbox 1 次。
- 單次 run 的預算 ≤ $12。

**路線圖：**
1. **MVP（本實作）**：單線迴圈加上本文列出的所有閘門。
2. **接真實資料與模型**：換成真實 OHLCV 和資金費率；用小預算（約 $3）試跑真實 LLM；觀察 `reflection_lift`。
3. **v2（feature flag）**：
   - Stat Auditor 批判者，只讀程式碼，需先用 A/B 證明有增益才開；
   - strategist 用 `Send` 平行跑 2–3 個假說，扇出前先預扣預算，只縮短等待時間、不增加選擇壓力；
   - PBO/CSCV、N_eff 報告、PostgresSaver。
4. **上線之前**：通過 lockbox，再做至少 3 個月 paper trading，並走完前一份文件的上線閘門清單。
