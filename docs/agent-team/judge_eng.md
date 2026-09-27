# 工程評審報告：LangGraph 自我反思 + 自我除錯交易策略優化架構（A / B / C / D）

> 評審角色：負責上線的工程主管。架構師自評分數一律不採用。
> 驗證環境：`scratchpad/venv`（langgraph 1.2.12，Python 3.11）
> 驗證腳本（全部在 `scratchpad/judge_tests/`，沒有修改 `/home/user/KN` 或候選檔案）：
> - `t_langgraph.py`：LangGraph 行為查證（B 的 F3/F4/F5/F7/F9/F10/F12、C 的 Send + Overwrite、D 的 destinations / reducer / Overwrite / RetryPolicy）
> - `t_candidates.py`：對 B/C/D 骨架做 monkeypatch 探測（預算、反思驗證、RESURRECT、aggregate 健壯性、id 冪等）
> - `t_c_budget.py`：C 的預算超支實測
> - `t_b_infra.py`：B 在基礎設施故障時的行為
> - `t_A_topology.py`：用 stub 節點依 A 文件 §9 重建圖，確認能編譯，並測預算例外路徑

---

## 1. 評分表

權重：R1 25%、R2 15%、R3 10%、R4 10%、R5 10%、R6 15%、R7 15%。分數 1–10。

| 候選 | R1 樣本外/過擬合 | R2 反思品質 | R3 除錯 | R4 探索 | R5 成本/可控 | R6 複雜度 | R7 觀測/重現/HITL/安全 | **加權總分** |
|---|---|---|---|---|---|---|---|---|
| **A** 強化 Reflexion 單線 | 7 | 7 | 8 | 4 | 7 | **8** | **8** | **7.10** |
| **B** 三層巢狀子圖 | 7 | 6 | 7 | 5 | 7 | 5 | 7 | **6.35** |
| **C** 族群演化 + Send | 7 | **8** | 7 | **9** | 4 | 4 | 6 | **6.45** |
| **D** Supervisor + 多批判者 | 6 | 6 | 7 | 5 | 6 | 5 | 6 | **5.85** |

計算方式：
- A = 1.75 + 1.05 + 0.80 + 0.40 + 0.70 + 1.20 + 1.20 = 7.10
- B = 1.75 + 0.90 + 0.70 + 0.50 + 0.70 + 0.75 + 1.05 = 6.35
- C = 1.75 + 1.20 + 0.70 + 0.90 + 0.40 + 0.60 + 0.90 = 6.45
- D = 1.50 + 0.90 + 0.70 + 0.50 + 0.60 + 0.75 + 0.90 = 5.85

### 各項打分理由

**R1 樣本外有效性 / 過擬合控制**
- A（7）：DSR 放在 IS 閘門，所以 OOS 之前就先做多重檢定；OOS 只給 LLM 1 bit；帳本有持久化的 `oos_used` / `lockbox_used` 欄位（可跨 run）；lockbox 有 canary 掃描。扣分：OOS 每 run 最多用 6 次，而且「第一個過 OOS 就停」，OOS 實質上變成篩子。
- B（7）：滿足即停、OOS t 門檻隨使用次數提高、OOS 失敗只回傳類別。扣分：OOS 與 lockbox 的一次性許可放在記憶體內的 `Deps`（`oos_tokens: set`、`lockbox_used: bool`），進程重啟或換 thread 就會重置；OOS 失敗的類別（`decay,t_stat` 等）會回流給 ideate，洩漏量比 A 多。
- C（7）：OOS 紀律最乾淨：迴圈內完全不用 OOS，決賽者依 IS DSR 預先登記順序，各看 1 次。扣分：20 代選擇壓力都作用在同一組 IS folds 上；DSR 以 N_eff 為主，而聚類門檻本身就是一個可以調的自由度。
- D（6）：閘門規則本身完整。扣分：OOS / lockbox 次數只在 thread 內計數；**實測 RESURRECT 可以撿回 OOS 已失敗的假說**（違反它自己的規則）；Supervisor 看得到 OOS pass/fail 與粗分類。

**R2 反思品質**
- A（7）：預登記 + fold-bootstrap CI 由程式判定 verdict + lesson 支持/反駁計分與淘汰 + observed/conjecture 分離，整套最完整、也最容易實作。
- B（6）：設計上有數字對帳，但**實測：驗證不通過的反思仍然會被送去 `prepare_debug`，裡面的 change 照樣被實作**，`verify_reflection` 也沒有像文件說的那樣累加 `no_progress`。「驗證」實際上沒有擋住任何東西。
- C（8）：唯一做了「反思增益 A/B 消融」的設計（hypothesis_test 子代 vs. 無引導 mutate 子代的勝率），可以直接回答「反思有沒有用」。族群對照組讓證據的資訊量最高。扣分：骨架的 `validate_reflection` 只檢查引用數 ≥ 2，沒有核對 pair 是否存在，也沒有核對方向。
- D（6）：證據斷言 `(ref, op, value)` 的格式設計得好。扣分：預測計分用的是 `hypothesis.prediction`，不是文件說的「被採用 directive 的 Prediction」，所以批判者的兌現率歸因是錯的；**實測 `aggregate` 遇到 `proposal=None`（schema 允許）會 TypeError 崩潰**；同一代模型的多個批判者，錯誤之間高度相關。

**R3 除錯效率與穩健性**（本次重點）
- A（8）：9 類錯誤各有專屬上下文與上限；hash 正規化；行為指紋去重；E_INFRA 不給 LLM；debugger 不看績效；禁止改 PARAM_SPACE；有 run 層級上限。缺點：沒有模型升級階梯。
- B（7）：Haiku patch → Sonnet 先解釋再補丁 → 依 spec 重寫，這條升級階梯是四份裡最好的。但**實測：沙盒持續 E_INFRA 時，6 個 idea 全部燒光，還寫了 6 條「could not be implemented」lesson**。文件說「L3 視為 crashed 並停止 run」，這不屬實，而且會造成 lesson 中毒。此外骨架的 `RetryPolicy` 都沒有設 `retry_on`，用的是預設值（見 §3 的共通陷阱）。
- C（7）：錯誤在 worker 內局部處理；同一 hash 出現第 2 次就放棄；前視錯誤只給一次修正機會；錯誤直方圖會回饋成 pitfalls。缺點：worker 沒有 try/except，**實測任一 worker 例外都會讓整個 superstep 失敗、`invoke` 直接拋出**（resume 後只重跑失敗的那個，這點屬實）；斷路器沒實作；錯誤分類較粗。
- D（7）：同一 hash 重複時升級到 Opus；有全域上限 30；AST 語意 diff 會把「改了邏輯」的除錯計為一次試驗，這點設計很好。缺點：`code_hash` 混入 `attempts`，程式碼去重形同失效；`variant_id` 用 uuid4，節點重試後 id 會變。

**R4 探索**：C 的島嶼、crossover、fresh 子代與新穎性機制明顯最強（9）；B/D 靠換題與候選（5）；A 是序列爬山（4）。

**R5 成本可控**（本次重點：預算檢查擋不擋得住、平行扇出會不會超支）
- A（7）：每個 router 的第一行都檢查預算，加上 per-call wrapper，每次 run 約 $8。**但實測：wrapper 丟出的 `BudgetExceeded` 會讓整個 run 崩潰**，因為 `strategist→coder` 是無條件邊，沒有 router 攔截，也沒有 error_handler，結果不會走到 finalize，也不會產報告。另外 `wall_start_ts` 用 `time.time()`，人工審閱的等待時間也會被算進 wall-clock 預算。
- B（7）：預算由上而下切 allowance、由下而上回收。實測超支 1 次呼叫（0.13 / 0.12），屬實。**wall-clock 上限沒有作用**：全程沒有任何節點累加 `wall_s`，實測把預算設成 `wall_s=1` 仍然跑到 candidate。
- C（4）：**實測超支 39%**（預算 $2.5，實際花 $3.47）。原因：(1) `select_elites` 寫入 `gen_stats.usd=None`，所以 `_est_gen_cost` 永遠用預設值 0.08，文件說「用實際單價預估」不屬實；(2) 預算只在世代邊界檢查，N 個平行 worker 內部（codegen + 最多 3 次 debug）完全不檢查。平行扇出時，超支上限就是一整代。
- D（6）：預算只在 `allowed_actions`（也就是 Supervisor）檢查。researcher → coder → debug×3 → 3 位批判者這一整圈都沒有檢查，實測 0.32 / 0.30。`wall_s` 是寫死的常數（sandbox 20、backtest 180），不是實測值，LLM 延遲沒算進去。

**R6 實作複雜度**（本次重點：1–2 週 MVP、節點數、樣板量、陷阱數）

| | 節點數 | 圖數 | 使用的 LangGraph 機制 | 已知陷阱數（本次查證） | 1–2 週 MVP |
|---|---|---|---|---|---|
| A | 15（+7 router） | 1 | 條件邊、interrupt、reducer | 2（interrupt 重跑、reducer） | **可以**（前提是引擎與沙盒已存在） |
| B | 22 | 3 | 子圖當 node、input/output_schema、error_handler、Runtime context | 6（F3 重複累加、F4/F5 checkpointer 語意、F7、F9 handler 必須回傳 Command、F10、NotRequired KeyError） | 偏緊；約 30% 程式碼是搬運用的樣板 |
| C | 15 + 4（worker） | 2 | Send、子圖、Overwrite、2 個 interrupt、max_concurrency | 5（Send payload、Overwrite 與 append 同步驟會吃掉資料、兄弟任務 pending writes、hash seed、平行預算） | 只能做出縮減版 |
| D | 20 | 1 | Command goto、Send、Overwrite、destinations、HMAC | 5（destinations 不強制、assert_topology 只驗宣告、assert 被 retry、Overwrite 單寫、Command 路由無法靜態檢查） | 需要先砍掉 LLM Supervisor |

**R7 可觀測 / 可重現 / HITL / 安全**
- A（8）：每節點寫 JSONL 事件；帳本冪等鍵 = variant_id / prereg_hash；帳本裡的 OOS / lockbox 鎖是持久的；canary；沙盒旗標寫得很明確；human_review 前無副作用。缺點：沒有中途 HITL。
- B（7）：namespace 分層稽核；**`interrupt(response_schema=...)` 實測會驗證 resume 值，屬實**；有 data_manifest。缺點：OOS / lockbox 一次性許可在記憶體；HITL 只有 approve / reject。
- C（6）：**`random.Random(hash((run_id, gen)))` 因為 PYTHONHASHSEED 每個進程不同，實測 3 個進程得到 3 個不同亂數**，「以 (run_id, gen) 為種子、可無 LLM 重播」不屬實。每 5 代的期中 interrupt 會讓「跑一晚」的批次作業卡住等人。`oos_evaluate` 在程式裡並不冪等。
- D（6）：`LLM_VISIBLE` 投影白名單與 supervisor_log 很好。缺點：`variant_id` / `round_id` / `report_ref` 用 uuid4，不可重現；guardrail 靠 `assert`；HMAC 在單一進程內基本上是安全劇場。

---

## 2. 骨架實跑結果與行為查證

### 2.1 骨架實跑

| 骨架 | 結果 | 關鍵輸出 |
|---|---|---|
| B `candidate_B_skeleton.py` | ✅ exit 0 | 停在 `strategy_review` interrupt → resume 成功 → report；主 state 不含私有鍵（assert 通過）；namespace 為 `refine:<id>\|debug:<id>`；`spent.wall_s = 0` |
| C `candidate_C_skeleton.py` | ✅ exit 0 | 第 5 代 midrun interrupt → resume → 第 8 代停滯停止；118 trials、$6.048、oos_peeks = 3、inbox 最後為 0；成本逐項對得上（128 × 0.03 + 52 × 0.004 + 8 × 0.25），沒有重複累加 |
| D `candidate_D_skeleton.py` | ✅ exit 0 | 正常路徑：researcher → coder → debug 1 次 → backtest → 3 位批判者並行 → aggregate → gate → OOS → lockbox → interrupt → finalize；`supervisor_log` 為空（正常路徑根本沒呼叫 Supervisor） |
| A（無骨架） | 依 §9 重建拓撲後可以編譯並跑到 interrupt | 預算例外路徑會崩潰（見下表） |

### 2.2 架構師聲稱的行為 vs. 實測

| # | 聲稱 | 結論 | 證據 |
|---|---|---|---|
| B-F3 | 父子圖共用 reducer 鍵會重複累加 | ✅ 屬實 | `['p0','p0','child']` |
| B-F3 fix | 子圖設 `output_schema` 只輸出結果鍵就能避免 | ✅ 屬實 | `{'log':['p0'],'result':'r'}` |
| B-F4 | 子圖 `checkpointer=None` 時每次呼叫都是全新 state | ✅ 屬實 | 連續 3 次呼叫得到 `[1,1,1]` |
| B-F5 | `checkpointer=True` 會跨呼叫累積 | ✅ 屬實 | `[1,2,3]` |
| B-F9 | error_handler 必須回傳 `Command(goto)`，回傳 dict 會讓圖默默停止 | ✅ 屬實 | 回傳 dict 時後續節點 `after` 沒有執行，也不報錯 |
| B-F12 | `interrupt(response_schema=TypedDict)` 會驗證 resume 值 | ✅ 屬實（補充） | 值不合法時，**resume 呼叫本身拋出 `ValidationError`**，不會自動重新 interrupt，呼叫端必須自己 catch |
| B-F7 | 子圖內 interrupt，resume 時從子圖 checkpoint 接續 | ✅ 屬實 | 子圖當 node 時 `pre=1, hr=2, parent_before=1` |
| B-F10 | recursion_limit 會繼承到子圖 | ✅ 屬實 | 限制 20、子圖要跑 50 步時觸發 `GraphRecursionError` |
| B | 預算最多超支一次呼叫 | ✅ 屬實 | 0.13 / 0.12 |
| B | run wall-clock 上限 2h | ❌ 不屬實 | 沒有節點累加 `wall_s`；`wall_s=1` 的預算仍跑到 candidate |
| B | 驗證不通過的反思不採用、計一次 no_progress | ❌ 不屬實 | `route_after_verify` → `prepare_debug`，`debug_request.change` 就是無效反思裡的 change |
| B | E_INFRA 時 L3 視為 crashed 並停止 run | ❌ 不屬實 | 6 個 idea 全燒光、寫了 6 條 infeasible lesson（`t_b_infra.py`） |
| C | Send 扇出到已編譯子圖，reducer 扇入 | ✅ 屬實 | 每代 inbox = 4/4/4、成本 24 = 預期值；payload 帶不帶 `inbox:[]` 結果相同 |
| C | `Overwrite([])` 清空扇入緩衝 | ✅ 屬實 | 最後 inbox 為 `[]`。**隱藏陷阱**：同一 superstep 內若有 Overwrite 又有 append，append 會被**默默丟掉**（`{'inbox': []}`） |
| C | 已完成的兄弟任務以 pending writes 保存，恢復時只重跑失敗者 | ✅ 屬實（補充） | 但第一次 `invoke` 會直接拋出 KeyError，run 中止，需要外部 resume |
| C | RNG 以 (run_id, gen) 為種子、可重播 | ❌ 不屬實 | `hash()` 受 PYTHONHASHSEED 影響，3 個進程得到 3 個不同亂數 |
| C | route_budget 用最近幾代實際單價 ×1.3 預估 | ❌ 不屬實 | `gen_stats.usd` 永遠是 None，估算恆為 0.08；實測超支 +39% |
| C | 子圖步數不計入父圖 | ⚠️ 部分屬實 | 父子各自計步，但子圖每次呼叫同樣受 recursion_limit 限制 |
| D | `destinations` 只用於畫圖、runtime 不強制 | ✅ 屬實（而且更糟） | 宣告 `('coder',)` 卻 goto 到 `secret` 照樣執行；`get_graph()` 畫出不存在的 `sup→coder`，實際走的 `sup→secret` 卻沒畫出來 → **D 的 `assert_topology()` 只驗證宣告，無法證明 runtime 行為** |
| D | reducer 必須剛好是 (a, b) | ✅ 屬實 | `ValueError: Invalid reducer signature` |
| D | Overwrite 同一 superstep 只能寫一次 | ✅ 屬實 | `InvalidUpdateError` |
| D | `guarded()` 限制 Supervisor | ⚠️ 部分屬實 | 正常情況下 Supervisor 經靜態 `ACTION_TO_NODE` 映射，guard 根本不會觸發。一旦觸發：`AssertionError` 會被預設 `RetryPolicy` **重試 3 次**（`default_retry_on(AssertionError)=True`，等於 3 次 LLM 呼叫）後讓 run 崩潰，不會退回 STOP；`python -O` 下 assert 被移除，guard 完全失效 |
| D | OOS 失敗的變體不能 RESURRECT | ❌ 不屬實 | `allowed_actions` 有檢查，但 `supervisor()` 實際用 `next(a for a in archive if hyp_id != cur)` 挑，沒有排除 `oos_failed` → 實測挑中 H1（已 OOS 失敗） |
| D | aggregate 確定性且健壯 | ❌ 不屬實 | `proposal=None`（schema 允許）時 `TypeError` 崩潰 |
| D | 預算在 guardrail 第一行，超支就只剩 STOP | ✅ 屬實（有限度） | 0.32 / 0.30；在 Supervisor 之間的一整圈都不檢查 |
| A | 每個 router 第一行檢查預算 + per-call 硬上限 | ⚠️ 部分屬實 | wrapper 丟出 `BudgetExceeded` 後 run 崩潰，停在 `next=('coder',)`，不會走到 finalize |
| 全體 | 模型 ID 與價格（Opus 5 $5/$25、Sonnet 5 $2/$10、Haiku 4.5 $1/$5）；Opus 5 不能調 temperature | ✅ 屬實 | 依 claude-api skill（快取 2026-06-24）。另外 `claude-opus-5-5`（$4/$20）即將上線，屬於未來的降價選項 |

---

## 3. 各候選的工程錯誤與陷阱清單

### 共通（四份都有）
1. **`RetryPolicy` 的預設 `retry_on`**：預設會重試任何不在排除清單裡的例外，包括 `AssertionError`、自訂的 `BudgetExceeded`、`anthropic.BadRequestError`（400）。B / C / D 都沒指定 `retry_on`，結果是非暫時性錯誤被重試 3 次、LLM 被重複呼叫。A 有指定，是唯一正確的一份。
2. **Opus 5 的 `stop_reason: "refusal"`**：沒有任何一份處理。結構化輸出會拿到 None，validator 就會崩潰。應該在 LLM wrapper 裡檢查 stop_reason，並開啟 server-side fallbacks。
3. **wall-clock 預算**：A / C 用 `time.time() - start`，人工等待時間也被算進去；B 完全不累加；D 用常數。正確做法是由每個節點外殼累加實際執行耗時（active time），HITL 等待不算。
4. **節點冪等 id**：D 用 uuid4，C 在 worker 裡用了 hash seed，都會讓重試或重播時帳本對不上。id 必須是內容 hash 或 `(run_id, 計數器)` 這類確定性值。

### A
- `BudgetExceeded` 從 `llm_structured` 丟出，run 直接崩潰（已實測）。`strategist→coder` 是無條件邊，沒有 router 攔截。修法：LLM 節點加 `error_handler` → `Command(goto="finalize")`（採用 B 驗證過的模式），或在 `strategist→coder` 之間加一個 budget router。
- 人工選 `continue` 之後回到 strategist，下一次 finalize 的 lambda router 沒有檢查 lockbox 是否已用過；再次進入 `lockbox_eval` 時會碰到帳本鎖，拋例外崩潰。router 應該先讀 `lockbox_used`。
- `merge_lessons` 依 `support - refute` 排序後截到 12 條：一旦 12 條舊 lesson 都有正分，新 lesson（0 分）會被立刻擠掉，永遠進不了記憶。需要保留「新 lesson 的試用席位」（例如 2 條）。
- DSR 的 N 取原始組態數（每 run 約 1900），加上 `MIN_DSR = 0.95`，幾乎不可能過閘門。這在統計上是對的，但產品上的結果是「永遠 0 個候選」，需要在 run 開始前預告門檻曲線（C 的做法）。
- `route_after_is_gate` 裡的 `budget_exceeded` 沒有檢查 `oos_runs`（由 LIMITS 另外擋，可以接受）。
- 沒有可執行骨架。文件裡的程式碼已經接近可跑，但仍需第 1 天補 stub e2e。

### B
- 反思驗證沒有生效（已實測）：`valid=False` 仍走 `prepare_debug`；`verify_reflection` 不累加 `no_progress`。
- E_INFRA 被當成 `gave_up` → `infeasible`：基礎設施故障時燒光所有 idea，並寫出錯誤 lesson（已實測），等於 lesson 中毒。
- `wall_s` 上限沒有作用（已實測）。
- OOS / lockbox 的一次性許可放在 `Deps`（Runtime context，不在 checkpoint 裡）：換進程或換 thread 就重置。應該改放在帳本 DB。
- `m_oos_eval` 在許可已用過時 raise `RuntimeError`，但 `oos_eval` 沒有 error_handler，run 會直接崩潰。
- `sandbox_run` 回傳 dict 代表失敗，而不是 raise，所以掛在它上面的 `RetryPolicy(max_attempts=2)` 永遠不會觸發。文件說「RetryPolicy 處理 InfraError」和骨架不一致。
- 骨架的 `r_score_prediction` 讀 `p["metric"]`，文件的 schema 是 `metric_path`，兩者不一致。
- 樣板量：3 組 In/Out/State schema，加上 4 個 prepare/absorb 轉接節點。

### C
- 預算估算器失效，加上平行扇出時 worker 內部沒有預算檢查，實測超支 +39%。平行扇出的正確做法：扇出前**預扣**「N × per-worker 上限」；每個 worker 帶自己的 allowance；worker 的 LLM wrapper 做呼叫前估算。
- `hash()` 做種子不可重現。改成 `int(hashlib.sha256(f"{run_id}|{gen}".encode()).hexdigest()[:16], 16)`。
- worker 沒有 try/except：任何例外都會讓整代中止（文件聲稱「worker 永遠不 raise」，程式沒有保證）。應該在 `w_emit` 前加包裝，或掛 error_handler 並回傳 `status="dead"`。
- Overwrite 與 append 在同一 superstep 會吃掉資料：目前只有 `select_elites` 寫 inbox，所以還沒出事；但只要將來有人讓 `population_reflect` 與 `select_elites` 並行，或在同一步寫 inbox，就會默默丟失結果。建議加 CI 測試。
- `oos_evaluate` 在迴圈裡直接呼叫 OOS，沒有先查帳本的 oos_peek；節點重試會重看 OOS。
- 每 5 代的 `checkpoint_review` interrupt 會讓「跑一晚」的批次卡住。應該改成非阻塞通知 + 可選的停止旗標。
- `validate_reflection` 只檢查引用數，沒有核對證據存在性與方向（文件說有）。
- 元件數最多（島嶼、niching、遷移、N_eff 聚類、CSCV、假說帳本、2 個 HITL），1–2 週只能做縮減版。

### D
- `destinations` 加上 `assert_topology()` 只驗證宣告，不驗證行為（已實測）。Command 路由無法靜態驗證，這是 Command-heavy 設計的根本弱點。
- `guarded()` 用 `assert`：在 `-O` 下失效；觸發時被 RetryPolicy 重試 3 次（3 次 LLM 費用）後崩潰（已實測）。應改用明確的 `if ...: return Command(goto=default)` 並記錄 violation。
- RESURRECT 會撿回 OOS 失敗的假說（已實測）。`allowed_actions` 與 `supervisor()` 的挑選邏輯不一致。
- `aggregate` 在 `proposal=None` 時崩潰（已實測）。
- 預測歸因錯誤：用 `hypothesis.prediction`，不是被採用 directive 的 prediction，所以 `critic.hit` 統計不可信，而「依兌現率自動停用批判者」正是建立在這個統計上。
- `coder_debug` 的 `code_hash` 混入 `attempts`，去重失效；`variant_id` 用 uuid4，不冪等。
- `oos_eval` 的「每變體 1 次」只比對 `state.oos` 最後一筆，而且 OOS 次數只在 thread 內計數。
- HMAC token：密鑰和 gate 在同一個進程，LLM 本來就寫不到 state。增加了複雜度，卻沒有提供實際的安全邊界。
- 預算檢查只在 Supervisor，一整圈（最多 coder + 3 次 debug + 3 位批判者）不受控。

---

## 4. 推薦方案

### 主幹：**A（強化版 Reflexion 單線迴圈）**

理由：R6 / R7 最好，而 R1 / R2 / R3 與最佳者相差不到 1 分。A 是唯一不用 Send、子圖或 Command 的設計，LangGraph 陷阱最少，checkpoint 歷史就是研究日誌。它的主要工程錯誤（預算例外崩潰、lockbox 重燒、lesson 擠壓）都是局部可修的，不需要動拓撲。A 的 R4 弱點可以用「受控扇出」在 v2 補上，而不需要搬出整套演化框架。

### 要嫁接的元件

| # | 元件 | 來源 | 放到 A 的哪裡 | 理由 |
|---|---|---|---|---|
| 1 | **LLM 節點的 `error_handler` → `Command(goto="finalize")`**（BudgetExceeded / 結構化輸出失敗 / refusal） | B（F9 已驗證） | strategist / coder / debugger / reflect | 修掉 A「預算例外讓 run 崩潰」的問題；handler 必須回傳 Command，這點已實測 |
| 2 | **呼叫前預算估算**：輸入 token 數 + `max_tokens` × 輸出單價 | B §10 | `llm/client.py` | 把超支從「一整次呼叫」壓到「估算誤差」 |
| 3 | **`interrupt(..., response_schema=HumanDecision)`** 並 catch `ValidationError` 後重新詢問 | B（F12 已驗證） | human_review | 需求 10：人工回覆也要結構化驗證 |
| 4 | **除錯升級階梯**：Haiku patch → Sonnet explain_then_patch → regenerate | B §7.2 | debugger（依同一 hash 出現次數選擇） | A 唯一的 R3 缺口 |
| 5 | **「進步」必須超過雜訊帶**：`Δ > max(0.10, 2 × bootstrap SE)` | B `r_gate` | `_stalled()` / 改良判定 | 避免用雜訊驅動 refine |
| 6 | **Runtime context（`context_schema=Deps`）** 放帳本、store、沙盒 client | B | `graph.py` | 不可序列化的依賴不進 state |
| 7 | **OOS 只給預先登記順序的決賽者（≤3）各 1 次**，取代 A 的「每變體 OOS、第一個通過就停」 | C §5.3 | finalize 前新增 `select_finalists → oos_evaluate → pick_champion` | 迴圈內不再使用 OOS，R1 提升最多，實作成本低 |
| 8 | **反思增益消融**：固定比例（例如 20%）的 refine 改用程式模板產生的無引導改動，程式比較兩者的勝率 | C §6.3-5 | route_after_reflect / coder | 讓「反思有沒有用」變成可量測的數字，這是 R2 的核心問題 |
| 9 | **門檻曲線預告**（E[max SR \| N]） | C §8.2 | init_run 報告 | 讓人事先知道 DSR 會多嚴，緩解「永遠 0 候選」的體驗問題 |
| 10 | **`LLM_VISIBLE` 投影白名單 + `project(role, state)`** | D §5.2 | `context.py`（取代分散的 build_*_context） | 資訊隔離集中在一處，可以用單元測試窮舉 |
| 11 | **證據斷言格式 `(ref, op, value)`** 由程式驗證 | D §3.2 | Reflection schema（升級 A 的 `evidence_keys`） | 比「key 存在」更嚴格 |
| 12 | **確定性檢查登錄表**（`fee_x2, delay_1bar, drop_top5_trades, subperiod_split, param_jitter, shuffle_returns_null`），LLM 只能「加」檢查 | D §6.3 | IS gate | Stat Auditor 最有價值的部分，而且不需要多一個 LLM |
| 13 | **除錯後的 AST 語意 diff → `logic_changed` 計為一次試驗** | D §7 | debugger 後置檢查 | 堵住「拿除錯當改良」的後門（A 目前只拒絕，D 的做法同時把它記帳） |
| 14 | （v2，feature flag）**Stat Auditor 單一 LLM 批判者**，吃 code 全文 + 帳本統計 | D §6.5 | reflect 之前並行一個節點 | 語意型過擬合（magic number、日期條件）是單一反思者常漏掉的；先上確定性檢查，A/B 證明有增益後再開 |
| 15 | （v2）**strategist 用 `Send` 扇出 K = 2–3 個假說**，每個帶獨立 allowance、扇出前預扣預算、worker 內 try/except | C 的機制，搭配 §3 C 的修正 | strategist → coder 之間 | 補 R4，同時避免 C 的平行超支問題 |

### 不要採用的元件

| 元件 | 來源 | 理由 |
|---|---|---|
| 三層巢狀子圖 + 信箱 schema + prepare/absorb 轉接節點 | B | 樣板約 30%，陷阱 6 個，換到的「分層稽核」用單圖 + 結構化事件日誌就能做到 |
| 用 `checkpoint_ns` 前綴做資料分區（DataGateway） | B | 靠 namespace 字串做存取控制很脆弱；應該用不同的容器掛載 / volume（A 的做法） |
| OOS / lockbox 許可放在記憶體 `Deps` | B | 必須放進持久化帳本 |
| 島嶼模型、crossover、niching、遷移、N_eff 聚類（作為主要 DSR） | C | MVP 範圍外；N_eff 門檻是額外的自由度；主要判定保持用 raw N，N_eff 只放報告 |
| 每 N 代阻塞式期中 interrupt | C | 批次作業會卡住；改成非阻塞通知 |
| `hash()` 做 RNG 種子 | C | 不可重現 |
| LLM Supervisor | D | 本質是 2–5 類分類器；A 的確定性 router 已經在做同樣的事，而且正常路徑下它根本沒被呼叫 |
| HMAC gate token | D | 同進程內沒有實際的安全邊界，只增加複雜度 |
| 用 `destinations` + `assert_topology()` 當 guardrail | D | 已實測只驗證宣告；請用條件邊（可靜態檢查）取代 Command 路由 |
| `assert` 型 guard | D | `-O` 下失效、會被 RetryPolicy 重試 |
| 三位批判者並行（尤其 Risk Critic） | D | 同一代模型錯誤相關；Risk 與 gate 重複；佔成本 36% |

---

## 5. MVP 實作順序與模組切分

> 前提：固定、已由人審過的回測引擎與拋棄式沙盒已經存在（四份候選都有這個依賴）。如果沒有，整體需要加 1 週。

### 模組 / 檔案清單

```
quantloop/
  config.py            # LIMITS / IS_GATE / OOS_GATE / CLASS_CAP / 價格表；整份 hash 進 engine_version
  state.py             # TypedDict + reducers（add_budget, merge_lessons 含試用席位）；只放 ID / 小數字
  graph.py             # build_graph()：單一 StateGraph、條件邊、error_handler、SqliteSaver/PostgresSaver
  routers.py           # 全部純函式：route_after_static/validate/is_gate/reflect/abandon/human；budget_exceeded
  context.py           # LLM_VISIBLE 白名單 + project(role, state)；lockbox 沒有任何讀取路徑
  deps.py              # Runtime context dataclass：ledger、artifact store、sandbox client、llm client
  nodes/
    init_run.py        # 資料快照 hash、切分、門檻曲線預告
    strategist.py      # T1；預登記 + prereg_hash；novelty（Jaccard）
    coder.py           # T2；確定性 variant_id = f"{idea_id}.{refine_iter}"
    checks.py          # static_check（AST 規則）、sandbox_validate（截斷不變性、行為指紋）
    debugger.py        # 升級階梯 + AST 語意 diff（logic_changed）
    backtest.py        # backtest_is（WF + purge/embargo）+ 確定性檢查登錄表
    record_trial.py    # 帳本冪等寫入、DSR、預測 bootstrap 判定、lesson 計分
    reflect.py         # T1；Reflection schema（證據斷言）；validate_reflection
    memory.py          # compact_memory、IdeaCard
    finalize.py        # select_finalists（預登記順序）→ oos_evaluate（每人 1 次）→ pick_champion；PBO
    lockbox.py         # 帳本鎖 + canary
    human_review.py    # interrupt(response_schema=HumanDecision)，前面無副作用
    error_handlers.py  # BudgetExceeded / Refusal / SchemaFail → Command(goto="finalize"|"abandon_variant")
  llm/
    client.py          # structured output、stop_reason/refusal 處理、呼叫前預算估算、usage→cost、RetryPolicy retry_on 白名單
    schemas.py         # Pydantic：HypothesisOut、CoderOut、DebugOut、Reflection、HumanDecision
    prompts/*.md       # 版本化 prompt（版本號進事件日誌）
  safety/
    ast_rules.py       # 前視 / import / IO 規則
    sandbox.py         # docker run --rm --network none --read-only --user 65534 ... ；E_INFRA 分類
    invariance.py      # 截斷不變性
  ledger/
    db.py              # SQLite：trials、oos_used、lockbox_used、prereg（冪等鍵）
    dsr.py  pbo.py
  artifacts/store.py   # 內容定址 sha256 → 路徑
  obs/events.py        # JSONL：run_id, node, variant_id, prompt_ver, model_id, code_hash, data_hash, cost, active_ms, route_reason
tests/
  test_routers.py      # property test（例如 debug_iter≥4 ⇒ abandon；預算超標 ⇒ finalize）
  test_resume.py       # interrupt → resume；節點重試不重複寫帳本；lockbox 只燒一次
  test_budget.py       # 預算例外 → finalize（不崩潰）；wall-clock 不含人工等待
  test_isolation.py    # canary：lockbox/OOS 數字不出現在任何 prompt
  test_langgraph_traps.py  # Overwrite+append 同步驟、RetryPolicy retry_on、reducer 簽名
```

### 第 1 週：跑通「產生 → 檢查 → 除錯 → IS 回測 → 帳本」

| 天 | 內容 | 完成標準 |
|---|---|---|
| D1 | `config.py`、`state.py`、`ledger/db.py`（冪等鍵）、`artifacts/store.py`、`obs/events.py` | 帳本 upsert 重跑不增列 |
| D2 | `safety/ast_rules.py`、`sandbox.py`、`invariance.py` + 錯誤分類與 hash 正規化 | 100 個已知壞程式碼的分類回歸測試 |
| D3 | `llm/client.py`（結構化輸出、refusal、呼叫前估算、retry_on 白名單）、schemas、prompts v1 | stub 與真實模型都能回傳 schema；超預算時丟 `BudgetExceeded` |
| D4 | `graph.py` 主線：init_run → strategist → coder → static/sandbox ↔ debugger → abandon；`routers.py`；`error_handlers.py` | stub LLM e2e；預算耗盡時走到 finalize，不崩潰 |
| D5 | `backtest.py` 接引擎（WF + purge/embargo）、`record_trial.py`（DSR、raw N）、IS gate；SqliteSaver + resume 測試 | 一個真實想法跑完 IS；節點重試不重複寫帳本 |

### 第 2 週：反思、過擬合收尾、HITL

| 天 | 內容 | 完成標準 |
|---|---|---|
| D6 | `reflect.py` + `validate_reflection`（證據斷言、單一改動、改動指紋）；預測 bootstrap 判定 | 捏造的證據被拒絕；verdict 由程式決定 |
| D7 | `memory.py`（lessons 試用席位、IdeaCard）、停滯 / 預測連錯路由、除錯升級階梯 | 迴圈能換想法；lesson 數維持 ≤ 12 |
| D8 | `finalize.py`（預登記決賽者 → OOS 每人 1 次 → pick_champion、PBO）、`lockbox.py`、`human_review.py`（response_schema） | OOS / lockbox 鎖在帳本；resume 值不合法時會重新詢問 |
| D9 | 確定性檢查登錄表（fee_x2、delay_1bar…）、反思增益消融（20% 無引導對照）、canary 測試、報告 | 報告裡有反思命中率與 lift |
| D10 | 真實 LLM 小預算 dry run（例如 $3）+ 混沌測試：沙盒持續 E_INFRA、預算耗盡、interrupt 後隔天 resume、重試冪等 | 所有路徑都有報告產出，沒有未處理例外 |

v2（第 3 週以後，放在 feature flag 後面）：Stat Auditor LLM 批判者（A/B）、strategist 扇出 K = 2–3 個假說（每個有獨立 allowance、扇出前預扣、worker 內 try/except）、PostgresSaver、有效 N（ONC）只放報告。
