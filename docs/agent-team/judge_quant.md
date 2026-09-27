# 評審報告：量化研究主管視角（judge_quant）

> 評審對象：candidate_A（強化版 Reflexion 單線）、candidate_B（階層巢狀子圖）、candidate_C（族群演化 + Send）、candidate_D（Supervisor + 多批判者）
> 評分依據：requirements.md R1–R7 權重。架構師自評一律不採用。
> 本報告的統計數字都是用下列假設實算的（python，常態近似）：年化 Sharpe 的標準誤 SE ≈ 1/√T年；跨試驗 SR 標準差 σ_SR ≈ 0.47（4.5 年 IS 在虛無假設下的 SE）；DSR 門檻 SR₀ = σ_SR·[(1−γ)Φ⁻¹(1−1/N)+γΦ⁻¹(1−1/(Ne))]，通過 DSR≥0.95 所需的觀測 SR ≈ SR₀ + 1.645/√T。

---

## 0. 先講清楚的統計現實（所有評分的共同基準）

| 量 | 數值 | 對設計的含義 |
|---|---|---|
| 1 年 OOS 的年化 SR 標準誤 | ≈ 1.0 | OOS 門檻「SR ≥ 0.5」在虛無假設下每次有 **31%** 會通過；看 3 次，至少一個假策略通過的機率 67%；看 6 次是 **89%** |
| 3 個月 fold 的 SR 標準誤 | ≈ 2.0 | 「pos_fold ≥ 4/6」：真 SR=1.0 只有 73% 通過，純雜訊也有 34% 通過。逐 fold 門檻基本上是在量雜訊 |
| 4.5 年 IS 的 SR 標準誤 | ≈ 0.47 | 反思提出的「Δ Sharpe ≥ 0.15」如果不做配對檢定，完全落在雜訊內 |
| DSR 門檻（T=4.5y） | N=10：需 SR≈1.52；N=20：1.67；N=50：1.85；N=100：1.96；N=320：2.15；N=1900：2.39 | 見 §5。真 SR=1.5 的策略通過 DSR 的機率：N=10 是 49%，N=20 是 36%，N=100 是 16%，N=320 是 8.5%，N=1900 是 3% |
| B 的 OOS t≥1.96（1 年） | 等於要求 OOS SR ≳ 2.0 | 真 SR=1.0 時檢定力只有 17%；到第 6 次 OOS 時 t_req=3.71，真 SR=1.5 的檢定力只剩約 1% |

結論：在 BTC 15m–1h 這種 alpha 稀薄的環境，**統計檢定力（power）才是瓶頸，不是 LLM 的創意**。任何會把有效試驗數推到上百的設計，在數學上就等於「保證找不到」。反過來，只要 OOS 門檻寬鬆又可以看很多次，OOS 幾乎擋不住假陽性。四份設計都沒有做檢定力分析，這是共同的缺陷。

---

## 1. 評分表

| 項目（權重） | A 強化 Reflexion | B 巢狀子圖 | C 族群演化 | D Supervisor 多批判 |
|---|---|---|---|---|
| R1 樣本外有效性／過擬合控制（25%） | 6 | 6 | 6.5 | 5 |
| R2 反思品質（15%） | 7 | 6 | 7.5 | 5.5 |
| R3 除錯效率與穩健性（10%） | 8 | 8 | 7 | 7 |
| R4 探索能力（10%） | 4 | 5 | 8 | 5 |
| R5 成本與可控性（10%） | 8 | 8 | 5 | 6 |
| R6 實作複雜度／可維護性（15%） | 8 | 6 | 4 | 5 |
| R7 可觀測／重現／HITL／安全（15%） | 7.5 | 8 | 7.5 | 7.5 |
| **加權總分** | **6.88** | **6.60** | **6.48** | **5.75** |

計算式：A = 1.50+1.05+0.80+0.40+0.80+1.20+1.125；B = 1.50+0.90+0.80+0.50+0.80+0.90+1.20；C = 1.625+1.125+0.70+0.80+0.50+0.60+1.125；D = 1.25+0.825+0.70+0.50+0.60+0.75+1.125。

評分說明（摘要）：
- **R1**：四份都沒拿到 7 分以上。每份都有至少一條 OOS 資訊洩漏路徑或試驗計數錯誤（見 §3）。C 分數最高，因為它對 LLM 做到 OOS 0 bit、決賽名單在看 OOS 前就預先登記好，還有跨 run 的 OOS 次數上限。但 C 的試驗量和 20 代的選擇壓力抵銷掉大部分優勢。D 最低：沒有定義資料切分，lockbox 只限「每 thread 1 次」，LLM 還看得到 OOS 分類。
- **R2**：C 有兩個別人沒有的關鍵機制：跨 ≥3 個親代、≥2 個島的**重複驗證**，以及 reflection lift 消融（對照無引導突變）。A 的預登記最完整（prereg_hash、效果量下限、observed/conjecture 分離、lesson 計分）。B 的證據對帳最嚴，但它的「進步」判定規則有錯。D 的證據斷言可以被空洞的不等式通過，預測計分也忽略了預測的指標。
- **R4**：C 的探索能力確實最強，所以給 8 分。但「探索能力」不等於「淨效益」，淨效益的分析見 §5。

---

## 2. 各候選的最大優點與最致命問題

### A：強化版 Reflexion 單線
**最大優點**
1. 預登記做得最完整：`prereg_hash` 在回測前寫入帳本；預測由程式判定並附 CI；validator 強制最小效果量；lesson 有 support/refute 計分與淘汰；另有全域「反思校準分數」。反思品質是可以量測的。
2. 圖最簡單（15 個節點、無 Send、無 subgraph），預算檢查集中在每個 router 的第一行，USD 8/run（上限 15），1–2 週真的做得出來。
3. 流程順序正確：DSR 放在 IS 閘門裡，**先過 DSR 才能消耗 OOS**；PBO 沒過就不跑 lockbox；lockbox 以資料快照為單位只用 1 次，並有 canary 掃描。

**最致命問題**
1. **試驗單位混淆**：每個 PARAM_SPACE 組態（≤48）都算 1 次試驗，但被 deflate 的是「巢狀 WF 選參後」的 OOF Sharpe。OOF 已經內含選參的懲罰，等於重複扣分。另外，用高度相關組態算出的 V[SR] 去搭配原始 N（約 1900），兩者在公式假設上互相矛盾。結果是門檻可能高到永遠過不了，也可能在 σ 很小時反而偏鬆，方向無法預測。
2. **OOS 是篩子，也是洩漏源**：OOS 每次看都用寬鬆門檻（SR≥0.5，α≈0.31），每 run 可以看 6 次，而且「第一個通過就停」。另外 `IdeaCard.closed_reason="oos_fail"` 與 route 結果會進 strategist 的 context，所以洩漏的不只是 1 bit，而是可以累積、會引導搜尋方向的訊號。
3. **IS 設計的統計力太弱**：6 個 fold × 3 個月只產生 18 個月的 OOF，54 個月的 IS 用不到一半。逐 fold 門檻（worst fold ≥ −0.5、4/6 為正）都在量雜訊。fold-bootstrap 只有 6 個重抽單位，CI 會被低估。

### B：階層式巢狀子圖
**最大優點**
1. 「滿足即停」（IS 閘門一過就停止精修），加上參數 ≤4、網格 ≤16，這是四份設計中唯一在結構上主動壓低試驗數的做法。
2. 基礎設施層的資料紀律最強：DataGateway 依 checkpoint namespace 強制只能讀 IS；OOS 與 lockbox 用一次性 handle；ledger 以 trial_id 冪等寫入。
3. 證據對帳最嚴格：`observed` 必須等於診斷包裡的數值（|Δ|≤1e-6），而且比較方向必須成立。Self-Debug 的升級階梯（Haiku patch → Sonnet 先解釋再補丁 → 依 spec 重寫）也有文獻依據。語意退化不進除錯迴圈，這條邊界劃得正確。

**最致命問題**
1. **DSR 和 PBO 在 OOS 已經用掉之後才算**（`stat_gate` 在 `oos_eval` 之後）。DSR 和 PBO 都是只用 IS 就能算的量，應該拿來決定能不能看 OOS。現在的順序等於把 OOS 次數浪費在本來就會被 DSR 否決的候選上，每浪費一次，多重比較的負擔也跟著增加。
2. **「進步」規則用錯標準誤**：`improved` 要求 Δ > max(0.10, 2×SE)，這裡的 SE 是「水準值」的 SE（約 0.5），所以要進步 +1.0 Sharpe 才算數。實務上第一版之後幾乎不會再有「進步」，`no_progress` 很快累積到 2，精修迴圈形同虛設。正確做法是用兩個變體之間的「配對差異」SE（兩個變體高度相關，差異的 SE 小很多）。
3. OOS 失敗時，會把「哪幾項檢查沒過」（decay、t_stat…）寫成 lesson 回灌給 ideate，這是多個 bit 的 OOS 洩漏。t 門檻 `1.96+0.35(k−1)` 的遞增幅度是隨意訂的，不是 Holm 或 Bonferroni。lockbox 和 OOS 的上限都以 run 為單位，沒有跨 run 的上限。三層子圖對 MVP 也是過度工程（見 §4）。

### C：族群演化（Send 扇出）
**最大優點**
1. 試驗計數在內部是一致的：每個變體只用預設參數，鄰域只拿來算 plateau，不從中挑最佳，所以「1 變體 = 1 試驗」的定義正確。DSR 同時用 N_eff（相關性聚類）和原始 N 做敏感度檢查。
2. OOS 紀律是四份中最好的：決賽者（≤3 個、彼此 ρ<0.7）在看 OOS **之前**就依 IS DSR 排好順序並登記；OOS 結果完全不回饋 LLM（0 bit）；同一個資料快照的 OOS 累計最多 9 次，lockbox 終身只用 1 次。
3. 反思有對照組：hypothesis_test 把同一個改動套到 ≥3 個親代、跨 ≥2 個島，是真正的重複驗證。reflection lift 用「無引導的 mutate_logic」當對照組做持續消融；lessons 由程式渲染，並附上 n 與中位 Δ。

**最致命問題**
1. **試驗數與選擇壓力**：320 次原始試驗，歷經 20 代，選擇壓力都作用在同一段 IS 上。更嚴重的是，C 的 walk-forward 在每個 fold 裡**完全不做擬合**（參數固定），所以它只是把時間切片分開評估，對演化選擇沒有任何保護作用。整段 IS 實際上就是演化的訓練集。
2. **反覆檢查就停（optional stopping）**：每一代都檢查名人堂第一名的 DSR，一旦 ≥0.95 就停。DSR 並不是為了反覆檢查設計的；邊看邊停會墊高假陽性率。N 雖然會隨代數增加，但這只是部分補償。另外，`sr_var=0.25` 寫死、N_eff 取 `max(…, √n)` 當下限、聚類門檻本身也是自由度，這三個都會直接改變門檻。
3. 假說判定有偏：Δ 是相對於「精英親代」算的，而精英是被挑出來的，本身就帶著正向運氣（winner's curse）。子代會系統性地回歸平均，假說因此偏向被判為 refuted。判定時也用 fitness 的 Δ，而不是假說預測的那個指標（例如 worst_sr）。contrast_pairs 只挑「Δ 最大的正負各半」給 LLM 看，等於專門把雜訊的極端值拿給它編故事。成本是單線設計的 3–5 倍，1–2 週做不完。

### D：Supervisor + 多批判者
**最大優點**
1. Stat Auditor 讀完整程式碼，專門抓語意型過擬合（magic number、日期條件、條件層層疊加），這是唯一和診斷包「正交」的資訊來源。
2. 除錯的後門被堵住：AST 語意 diff 如果發現邏輯改變，就標 `logic_changed` 並**改記一次試驗**。embargo 以最長持有期 ×2 來設，這才是正確的依據（其他三份都用 lookback 來 purge）。
3. `assert_topology()` 在 CI 裡檢查「OOS 的唯一入邊是 gate」；每個角色都有 `LLM_VISIBLE` 白名單投影；已經觀察到 `destinations` 在 runtime 不會被檢查這個 API 事實。

**最致命問題**
1. **R1 基本面不完整**：沒有定義 IS/OOS/lockbox 的期間；lockbox 只限「每 thread 1 次」（多開幾個 thread 就能重複使用）；Supervisor 看得到 `oos` 欄位（pass/fail 加 `severe_decay` 之類的分類），researcher 看得到含 OOS 失敗標記的 archive，所以 OOS 資訊會流回 LLM。每個參數組合都用原始 N 計入（40 變體 × 64 組 ≈ 2560），DSR≥0.95 在數學上幾乎不可能通過。
2. **反思計分有 bug**：`aggregate` 的 outcome 一律用 `cv.sharpe_median` 的 Δ 對照 `min_delta` 判定，完全忽略預測的 metric 與 direction；用的也是假說層級的預測，不是被採用 directive 自己的 Proposal.prediction。證據斷言只檢查 `op(actual, value)`，所以「high_vol.sharpe < 100」這種空洞的斷言也能通過。批判者自動停用所依據的「兌現率」建立在單點雜訊比較和極小樣本上，停用誰差不多是隨機的。
3. **過度工程**：LLM Supervisor 本質上是一個 2–5 類的分類器，架構師自己也預期它和規則版的一致率會超過 85%。HMAC token 防的是不存在的威脅：LLM 不能在 orchestrator 行程裡寫 state，拓撲測試就已經足夠。Risk Critic 和 gate 幾乎完全重疊。LLM 可以自選 `requested_checks`，使得每個變體面對的閘門不一樣。`shuffle_returns_null` 被拿來當前視偵測，其實是方法論錯誤（見 §3）。

---

## 3. 方法論錯誤清單

### 共同錯誤（多份都有）
| # | 錯誤 | 出現在 | 說明與修正 |
|---|---|---|---|
| X1 | purge = 最大 lookback | A、B、C | purge/embargo 是為了處理**標籤或持倉期跨越邊界**，應依最長持有期設定。指標的 lookback 向訓練段取暖機資料不算洩漏，應該允許（D 的做法正確）。 |
| X2 | walk-forward 只保護網格內的選參 | 全部 | LLM 看完整段 OOF 的切片診斷後再改結構，OOF 對「研究者」來說已經是樣本內。必要的修正：切出一段 **LLM 完全看不到的 shadow 驗證段**（C 列在 v2），只給 router 用，而且在看 OOS 之前使用。 |
| X3 | 預測判定沒有用配對檢定 | A（6-fold bootstrap）、B（單點比較＋水準 SE）、C（相對精英親代）、D（單點比較） | 兩個變體跑在同一段資料上，而且高度相關，應該對**日報酬差異序列**做 stationary block bootstrap，求 ΔSR 的 CI，再判定 confirmed / refuted / inconclusive。這樣「Δ≥0.15」才有機會被測出來。 |
| X4 | OOS 門檻沒有檢定力分析 | 全部 | 1 年 OOS 的 SE≈1。A 和 C 的 SR≥0.5 形同虛設（α≈0.31）；B 的 t≥1.96 則幾乎不可能通過（真 SR 1.0 的檢定力約 17%）。修正：OOS 改成「和經過 haircut 的 IS 預期是否一致」的一致性檢定，加上 SR>0；並把 k 次查看的 α 做 Bonferroni/Holm 分配。能延長 OOS 到 18 個月更好。 |
| X5 | DSR 的 N 與 V[SR] 配對不一致 | A、D（原始 N ＋ 相關試驗的變異數）、C（N_eff ＋ 全試驗變異數） | Bailey–López de Prado 公式假設獨立試驗。正確配對是：以 ONC 聚類得到 N_eff，並用**各群代表報酬的 SR 變異數**，不是全部試驗的變異數。 |
| X6 | DSR 在迴圈中隨時間計算 | A（IS 閘門）、C（每代檢查） | 早期變體面對的 N 比較小，有先到先得的優勢；邊看邊停又墊高假陽性率。修正：finalize 時用**最終 N** 重新 deflate 所有候選；中途的 DSR 只當成「可否申請 OOS」的條件，不當成停止條件。 |
| X7 | 切片診斷沒有多重比較意識 | A、B、D（波動三分位 × 趨勢 × 多空 ≈ 12 格以上） | LLM 會挑最差的那一格加 regime filter，這是分岔路徑花園（garden of forking paths）。修正：切片維度預先宣告、每格附上 CI 與交易數；由切片引出的 filter 在複雜度上要額外扣分。另外，regime 標籤本身必須是因果的（只用過去資料），否則診斷就已經帶有前視。 |
| X8 | lockbox/OOS 上限以 run 或 thread 為單位 | B、D（A 的 OOS 也是） | 必須以**資料快照**為單位跨 run 累計（C 的做法正確）。 |

### A 專屬
- A1：每個組態都計入 N，又 deflate OOF（重複扣分，見 §2）。
- A2：`E_DEGENERATE`（交易 < 5、曝險 < 1%）送進 debugger，等於允許在不計試驗的情況下調門檻或單位。應該像 B 一樣移出除錯迴圈，或改計一次試驗。
- A3：PBO 的門檻設 0.5 太鬆（0.5 等於擲硬幣）。CSCV 的輸入是「每個變體已經挑好的最佳組態」，而且來自序列式的適應性搜尋，違反 CSCV「試驗集合事先固定」的前提。
- A4：6 × 3 個月的 fold 只用到 IS 的 18/54 個月。
- A5：`IdeaCard.closed_reason="oos_fail"` 和 OOS 失敗後強制換想法的路由，都會把 OOS 資訊帶給 strategist。

### B 專屬
- B1：DSR 和 PBO 在 OOS 之後才算（順序錯誤）。
- B2：`improved` 用水準值的 SE，而不是差異的 SE。
- B3：骨架中 tabu 的 fingerprint 附加了 `-{iter}`，所以永遠不會命中，tabu 失效（骨架 bug）。
- B4：`distill_failure` 把失敗的檢查名稱回灌給 ideate，造成多 bit 的 OOS 洩漏。
- B5：t 門檻遞增幅度隨意；`pbo` 在骨架中寫死為 0.2；PBO 只在單一 idea 內計算，看不到跨 idea 的選擇。
- B6：只有當變體「進步」時才會更新 champion，所以一個通過閘門但沒有進步的變體會被丟掉。這條規則是用 IS 的最大值在爬山。

### C 專屬
- C1：walk-forward 裡不做擬合，整段 IS 都暴露在演化選擇之下。
- C2：每代檢查 DSR、達標就停，屬於 optional stopping。
- C3：`_judge_hypothesis` 用 fitness 的 Δ，而不是 `predicted.metric`；Δ 相對於精英親代，會有回歸平均的偏差。
- C4：contrast_pairs 只挑極端 Δ 給 LLM 看；tag attribution 的 bootstrap 把彼此相依的親子配對當成獨立樣本。
- C5：`sr_var=0.25` 寫死、N_eff 的 `√n` 下限是隨意的，而且會直接影響門檻。
- C6：期中 HITL 每 5 代介入一次，這是人為的自由度（架構師有寫進帳本，但沒有計入 N）。

### D 專屬
- D1：沒有定義資料切分期間。
- D2：lockbox 以 thread 為單位；Supervisor 看得到 OOS 的 bucket；archive 會帶 OOS 失敗標記給 researcher。
- D3：`shuffle_returns_null` 是顯著性檢定，不是前視偵測。「延遲 1 根 K 線後 Sharpe 崩掉 70%」在引擎已經做 shift-1 加下一根開盤成交的前提下，代表的是延遲敏感度或 alpha 衰減，不是前視。在 15m 上這條規則會誤殺真正的短週期 edge。
- D4：預測計分忽略 metric 與 direction；證據斷言可以是空洞的不等式。
- D5：LLM 可以自選 `requested_checks`，使得不同變體面對不同的閘門，結果無法互相比較。
- D6：批判者的停用規則沒有對照基線（例如「任何改動的平均兌現率」），樣本也太小。

---

## 4. 推薦方案

### 主幹：**A（強化版 Reflexion 單線）**
理由：
1. 在 alpha 稀薄、檢定力不足的環境，真正的瓶頸是「每個想法都要驗得乾淨，同時把有效試驗數壓低」，不是廣度。單線結構本身就讓 N_eff 保持小。
2. A 的圖最簡單、最便宜、最容易稽核。它的缺陷都在「統計細節」（試驗單位、fold 設計、OOS 回饋），這些都可以局部修正，不必改拓撲。
3. B 的層級化帶來的是工程治理，不是統計增益；C 和 D 的主要機制（GA、LLM Supervisor）在這個環境下的淨效益是負的。

### 嫁接元件
| 元件 | 來源 | 放在 A 的哪裡 | 理由 |
|---|---|---|---|
| 變體級試驗單位 + N_eff（ONC 聚類）+ 原始 N 敏感度 | C §8.2 | `record_trial` / `finalize` | 修正 A1 與 X5。組態只拿來算 plateau/PBO，不計入 N；deflate 時用群代表的 SR 變異數 |
| 預先登記的決賽名單與順序，再看 OOS | C `select_finalists` / `pick_champion` | 取代 A 的「第一個過 OOS 就停」 | IS 階段只累積「OOS 申請資格」；finalize 時用最終 N 重新 deflate，挑出 ≤2 個低相關的決賽者並登記順序，才一次看 OOS。消除邊看邊篩的偏誤 |
| OOS 對 LLM 0 bit；依資料快照跨 run 累計 OOS 次數上限 | C §8.1 | `oos_check`、IdeaCard | 刪掉 `oos_fail` 這個 closed_reason；OOS 不再有「失敗後換想法」的路由。OOS 只在決賽階段出現，自然不會回饋 |
| reflection lift 消融（對照組） | C §6.3-5 | `route_after_reflect` 加對照臂 | 每個想法固定抽 1 個 refine 名額，改用「無引導的隨機單一改動」（從 ChangeSpec 列舉中抽）。反思命中率必須贏過這個對照組才有意義；對照臂同樣計入試驗 |
| lesson 升級到 global 前需跨想法重複驗證，並附 n 與 Δ | C §6.3-3/4 | `compact_memory` | lesson 至少在 2 個不同想法上 confirmed 才能變成 global scope；渲染格式固定為「條件 → Δ 中位數、CI、n」 |
| shadow 驗證段 | C §8.4 v2（提升到 MVP） | `split_spec` | IS 最後 9–12 個月不給 LLM 看任何數字，只讓 router 在決賽篩選時用一次。這樣在消耗 OOS 之前就多一道真正的樣本外關卡 |
| 滿足即停 + 參數 ≤4、網格 ≤16 | B §5 | `route_after_is_gate` / `static_check` | 壓低試驗數與選擇壓力 |
| 數值等值的證據對帳 | B `verify_reflection` | `validate_reflection` | 比 A 只檢查 key 是否存在更嚴格；比 D 的不等式斷言更難鑽漏洞 |
| Self-Debug 升級階梯 + 語意退化移出除錯 | B §7 | `debugger` / `_debug_or_abandon` | 修正 A2；regenerate 可以避免在壞程式碼上反覆修補 |
| DataGateway 分區強制 + `interrupt(response_schema=…)` + 呼叫前成本估算 | B §8、F12、§10 | 沙盒掛載層、`human_review`、`llm_structured` | 在基礎設施層保證 IS/OOS/lockbox 的隔離；resume 值做 schema 驗證；超支幅度壓到只有一次估算誤差 |
| ideate 一次產生 3 個候選 + 確定性 novelty 挑選 | B `ideate` | `strategist` | 產生候選不算試驗（沒有評估績效），是很便宜的廣度 |
| Stat Auditor（單一、只讀程式碼） | D §6.5 | `reflect` 之前，並行 1 個 Send | 唯一和診斷包正交的資訊來源（magic number、日期條件）。它的意見要轉成確定性 lint 或固定檢查 |
| `logic_changed` 改記一次試驗 | D §7 | `debugger` 的後置檢查 | 堵住「拿除錯當改良」的後門 |
| embargo = 最長持有期 ×2 | D §8 | `backtest_is` | 修正 X1 |
| 固定的確定性檢查登錄表（fee_x2、delay_1bar、drop_top5、subperiod、param_jitter） | D §3.2 | `backtest_is` **一律全部執行** | 保留檢查本身，但拿掉 LLM 自選，讓所有變體面對同一組閘門 |
| `assert_topology()` CI 測試 + 每角色 `LLM_VISIBLE` 投影 | D §5、§9.3 | CI、context builder | 成本很低，能防止工程師無意間開出繞過 gate 的邊 |

### 不要採用
| 元件 | 來源 | 理由 |
|---|---|---|
| 完整 GA（島嶼、crossover、遷移、20 代 × 16） | C | 試驗數與選擇壓力的代價大於收益（§5）。crossover 的產物很難歸因，也讓反思無從下手 |
| 期中 HITL 每 5 代介入一次 | C | 人為自由度，而且沒有計入 N |
| 只挑極端 contrast pairs 給 LLM | C | 專門餵雜訊的極端值 |
| LLM Supervisor | D | 本質上是 2–5 類的分類器，邊際價值未經證實，又多一個無法重現的自由度。用 A 的確定性 router 就夠了 |
| HMAC gate token | D | 防的是不存在的威脅，拓撲測試已經足夠 |
| 三位批判者（尤其 Risk Critic）與依兌現率自動停用 | D | Risk Critic 和 gate 重疊；兌現率建立在雜訊上 |
| LLM 自選 `requested_checks` | D | 閘門會因變體而異 |
| `shuffle_returns_null` 或「delay_1bar 崩 70%」當前視證據 | D | 方法論錯誤（D3）。前視偵測交給截斷不變性測試 |
| 三層巢狀子圖 + 信箱 I/O | B | F3/F4/F7/F9 的坑多，約 30% 是搬運程式碼。A 用計數器就能做到同樣的隔離。如果需要，只把 debug 迴圈包成一個子圖 |
| OOS 失敗類別回灌成 lesson；遞增幅度隨意的 t 門檻 | B | 洩漏；α 分配沒有依據 |
| 「進步 > 2×水準 SE」規則 | B | 錯誤的標準誤 |
| A 自己要拿掉的：每組態計入 N、OOS 1 bit 與 `oos_fail` 回饋、第一個過 OOS 就停、3 個月 fold 門檻、6-fold bootstrap、E_DEGENERATE 進 debugger、PBO≤0.5 | A | 見 §3 |

---

## 5. 推薦的試驗預算配置

### 5.1 R4 的核心問題：族群探索划得來嗎？
- C 預設配置有 320 次原始試驗。演化子代高度相關，N_eff 估計約 60–120。依 §0 的表，要通過 DSR 需要 IS SR ≈ 1.9–2.0（4.5 年）；真 SR=1.5 的策略通過率約 16–20%。
- 單線配置（下面的建議）的 N_eff 約 10–15，需要 SR ≈ 1.5–1.6，真 SR=1.5 的策略通過率約 40–49%。
- 所以族群搜尋必須找到**真實 SR 高出約 0.4 以上**的策略，才能打平。在 BTC 1h 淨成本後，真實 SR 的分布集中在 0 附近，右尾很薄（可以實現的 edge 大約在 1–1.5）。擴大搜尋主要是在挖 IS 雜訊的右尾，換不到更高的真實 SR。**結論：族群式探索的收益不足以抵銷 DSR 懲罰。**
- 但 C 有兩個機制值得留下：「多家族並行起步」（廣度）和「重複驗證」。這兩者都不需要演化的選擇壓力。

### 5.2 建議配置（單一資料快照、BTCUSDT 永續、1h 為主；15m 只在成本模型通過 2× 壓力測試後才開放）

| 參數 | 建議值 | 理由 |
|---|---|---|
| 資料切分 | IS 2020-01 ～ 2023-12（其中 2023 年當 shadow 驗證段，LLM 看不到）；OOS 2024-01 ～ 2025-06（18 個月）；lockbox 2025-07 ～ 2026-06；之後的資料做 paper trading | OOS 的 SE 從 1.0 降到約 0.82；shadow 段在消耗 OOS 前多一道關卡 |
| IS 內 WF | expanding，初始訓練 12 個月，測試 fold 6 個月 × 5（2021–2023H1 的研究段），OOF 全部串接 | fold 的 SE 從 2.0 降到約 1.4；閘門以 OOF SR 和它的 CI 為主，逐 fold 只做粗檢（≥60% 為正） |
| 每個快照的 ideas 總數 | **8–10**（每 run 最多 8，每個快照最多 2 個 run 且共用帳本） | 家族輪替 ≥4 個家族，每個家族 ≤2 個 idea |
| 每個 idea 的 refine 次數 | **≤3**（最多 4 個變體），滿足即停，停滯或連錯 2 次就提前結束 | 配對檢定下，第 2–3 次改動之後邊際收益很小；lesson 的樣本也不會因此變多 |
| 參數網格 | ≤3–4 個參數、≤16 組；只用於 WF 內層選參與 plateau/PBO，**不計入 N** | 試驗單位 = 變體（它的 OOF 已經內含選參） |
| 每個快照的試驗上限 | 原始變體 ≤40；**N_eff ≤ 20**（寫死，超過就停止並要求換資料快照） | 依 §0，N_eff=20 時需要 SR≈1.67，是這個環境還能接受的上限 |
| 對照臂 | 每個 idea 抽 1 個 refine 名額做無引導的隨機改動（計入試驗） | 量測 reflection lift；lift ≤ 0 就降級成規則式精修 |
| 族群 / Send | **不用 GA**。只用 Send 做「廣度批次」：每批 N=3–4 個 idea 平行跑首版加精修，G=2–3 批（總共 8–10 個 idea）。批與批之間只傳遞 lessons（附 n、Δ、CI），不做親代選擇、不做 crossover | 平行只是為了縮短 wall-clock，不製造選擇壓力，N_eff 和單線一樣 |
| OOS | 每個快照終身最多 3 次；每個 run 只在 finalize 時對 ≤2 個預先登記順序的決賽者各看 1 次；α 用 Holm 分配；判準改為 SR>0 且和 haircut 後的 IS 預期一致（例如 OOS SR ≥ 0.5×IS_deflated − 1×SE） | 在 18 個月的 OOS 上，這是檢定力和假陽性之間比較可行的平衡 |
| lockbox | 每個快照 1 次，只給冠軍，結果只寫給人看 | 同 A/C |
| 預算 | USD ≤ 12/run；wall-clock ≤ 3h（有平行）；T1 只用在 strategist 與 reflect | 約 32 個變體 × 約 $0.2 加上 ideate，約 $7–9 |

### 5.3 預期結果與溝通
在這個配置下，真 SR=1.5 的策略通過 DSR 的機率大約四到五成，真 SR=1.0 的策略則幾乎不會通過（約 10–14%）。報告必須在 run 開始前預告「門檻曲線」（C 的做法），並把「找不到策略」列為正常的預期結果。真正的上線判斷交給 lockbox 加上 ≥3 個月的 paper trading，不是任何單一的回測數字。
