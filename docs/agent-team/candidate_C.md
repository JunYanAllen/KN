# 候選架構 C：族群演化式搜尋（Population-based / Evolutionary with Reflection）

> 完整可執行骨架（外部服務以 stub 代替，已在 `langgraph==1.2.12` 實跑通過：Send 扇出到子圖、reducer 扇入、`Overwrite` 清空緩衝、兩種 `interrupt()` + `Command(resume=...)` 恢復）：
> `docs/agent-team/candidate_C_skeleton.py`
> 本文件中的程式碼都是從該檔摘錄。

---

## 1. 一句話定位 + 適用情境

**定位**：把「單一策略自我反思／自我除錯」擴成「一個族群」。每一代用 LangGraph `Send` 平行扇出 N 個策略變體，每個變體在自己的子圖內跑短的自我除錯迴圈。扇入後由**確定性程式**依 IS walk-forward 穩健分數、複雜度懲罰與多樣性挑出各島精英。LLM 在**族群層級**比較好壞變體，提出**事先登記、可被否證**的假說，下一代用 A/B 名額去驗證。整個過程由全域試驗帳本 + 有效試驗數 DSR + CSCV-PBO 做多重檢定折扣，最後只讓 ≤3 個預先排序的決賽者各看一次 OOS。

**適用情境**
- 策略空間寬、單一路線容易卡在局部最佳（例如只在 EMA 交叉上微調）；想同時探索趨勢、均值回歸、資金費率 carry、波動突破等不同家族。
- 有平行沙盒算力（≥8 個拋棄式容器同時跑），每次 run 的 LLM 預算在 USD 20–60 之間。
- 研究型批次作業（跑一晚、隔天人工審閱），不是互動式的逐步調整。
- **不適用**：資料很短（< 2 年 1h K 線）、只想微調一個已知策略、或預算 < USD 5。這些情況試驗數一多，DSR 門檻就墊到無法達標，應改用架構 A/B 這種單線、試驗數少的設計。

---

## 2. ASCII 架構圖

```
                                   ┌──────────────────────────────── 父圖 EvoState（checkpointer: PostgresSaver, thread_id=run_id）
START
  │
  ▼
[init_run] 確定性：凍結資料快照 hash、切 IS/OOS/lockbox、開 ledger run、記錄 prompt/模型/引擎版本
  │         （seed：第 1 代所有 slot 都是 operator="fresh"，每島指定一個策略家族）
  ▼
┌─►[budget_gate]──(route_budget: USD/token/wall/backtest/試驗預算 ≥ 下一代估計成本×1.3 ?)──否──┐
│      │是                                                                                   │
│      ▼                                                                                     │
│  [plan_generation] 確定性：各島挑親代、依 OPERATOR_MIX 分配 operator、排入假說 A/B 名額        │
│      │                                                                                     │
│      │ fan_out() ⇒ [Send("variant_worker", payload_i) for i in 1..N]      ◄── 扇出（世代內平行） │
│      ▼                                                                                     │
│  ╔═══════════════ variant_worker 子圖（N 個平行實例，各自獨立） ═══════════════╗              │
│  ║ [w_codegen] LLM-中 ─► [w_check_and_run] 確定性：AST/schema/白名單 →        ║              │
│  ║                        沙盒：截斷不變性 → IS walk-forward → 參數鄰域        ║              │
│  ║                              │                                           ║              │
│  ║              route_after_run ├─ 錯誤 & 次數<3 & hash 未重複 ─► [w_debug]   ║              │
│  ║                              │         LLM-小（看不到績效）  │            ║              │
│  ║                              │◄──────────── 自我除錯短迴圈 ──┘            ║              │
│  ║                              └─ 成功 / 放棄 ─► [w_emit] 冪等寫 ledger       ║              │
│  ║                                               回傳 {"inbox":[result]}     ║              │
│  ╚═══════════════════════════════════════════════════════════════════════╝              │
│      │ reducer: inbox = operator.add      ◄── 扇入（全部 Send 任務完成後才進下一節點）        │
│      ▼                                                                                     │
│  [select_elites] 確定性：去重(signal hash / ρ≥0.95)、資格門檻、fitness、新穎性、島內 niching、  │
│      │           名人堂、假說判定、N / N_eff 更新、inbox=Overwrite([])                        │
│      ▼                                                                                     │
│  [population_reflect] LLM-強：讀 PopulationDigest（對照組 Δ、tag 歸因表、假說結果）             │
│      ▼                                                                                     │
│  [validate_reflection] 確定性：核對引用證據、限制數量/去重 → 假說帳本；lessons 由程式渲染          │
│      ▼                                                                                     │
│  [island_maintenance] 確定性：環狀遷移（需低相關）、停滯島重置                                  │
│      │                                                                                     │
│      ├─ route_after_generation ─► 每 5 代 ─► [checkpoint_review] ◄ HITL#1 interrupt()       │
│      │                                          │（只看 IS 儀表板；continue / stop / 調整）   │
│      │                                          ├─ continue ────────────────────────────┐   │
└──────┴─ 其他情況 ◄────────────────────────────────────────────────────────────────────────┘   │
         │ IS 達標(DSR_Neff≥0.95) / 停滯 / 達最大代數 / human stop                                 │
         ▼                                                                          ◄────────┘
     [select_finalists] 確定性：名人堂取 ≤3 個彼此 ρ<0.7，依 IS DSR「預先登記」順序；算 PBO(CSCV)
         ▼
     [oos_evaluate] 每個決賽者恰好 1 次 OOS（ledger 記 oos_peek；結果不回饋任何 LLM）
         ▼
     [pick_champion] 確定性：依預先登記的順序，取第一個通過全部門檻者（OOS 只做 pass/fail 過濾）
         ├─ 有冠軍 ─► [lockbox_eval] 每個資料快照僅 1 次；結果只進報告
         ▼                 │
     [prepare_review] ◄────┘  寫報告 artifact（副作用都放在 interrupt 節點之前）
         ▼
     [human_review]  ◄ HITL#2 interrupt()：approve_paper_trading / reject / archive
         ▼
        END
```

三種迴圈分開計數（`counters` reducer）：`debug_iters`（worker 內修錯，不經 Evaluator、不算試驗）、`improve`（mutate_logic / crossover / hypothesis_test 子代）、`reideate`（fresh 子代 + 島重置）。

---

## 3. State schema

### 3.1 reducers 與父圖 State

```python
# ───────────────────────────── reducers ─────────────────────────────
def sum_costs(a: dict, b: dict) -> dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + v
    return out


def merge_by_id(a: dict, b: dict) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


def cap_list(limit: int):
    def _r(a: list, b: list) -> list:
        return ((a or []) + (b or []))[-limit:]
    return _r


# ───────────────────────────── State ─────────────────────────────
class Budget(TypedDict):
    usd: float
    tokens: int
    wall_s: float
    backtests: int


class VariantResult(TypedDict):
    variant_id: str
    generation: int
    island_id: int
    operator: str
    parent_ids: list[str]
    hypothesis_id: Optional[str]
    status: Literal["ok", "dead", "duplicate", "degenerate"]
    code_ref: Optional[str]          # artifact store key (程式碼)
    returns_ref: Optional[str]       # artifact store key (IS 日報酬序列)
    signal_hash: Optional[str]
    metrics: dict                    # fold 級小指標：median_sr, worst_sr, pos_fold_ratio, n_trades, max_dd, complexity, n_params, plateau_ratio, tags
    debug_attempts: int
    error_hashes: list[str]
    counted_as_trial: bool


class EliteRef(TypedDict):
    variant_id: str
    island_id: int
    fitness: float
    code_ref: str
    tags: list[str]


class Hypothesis(TypedDict):
    hyp_id: str
    statement: str
    mechanism: str
    evidence_pair_ids: list[str]
    predicted: dict                  # {"edit": tag, "metric": "median_sr", "direction": "+", "min_effect": 0.05}
    status: Literal["proposed", "testing", "confirmed", "refuted", "rejected"]
    born_gen: int
    tests: list[dict]                # [{variant_id, parent_id, delta}]


class EvoState(TypedDict):
    run_id: str
    data_snapshot_id: str
    engine_version: str
    prompt_versions: dict
    generation: int
    phase: Literal["evolve", "finalize", "done"]
    # 當代扇入緩衝：reducer 累加；select_elites 用 Overwrite([]) 清空
    inbox: Annotated[list[VariantResult], operator.add]
    plan: list[dict]                               # 當代 VariantTask（無 reducer，覆寫）
    islands: dict                                  # {island_id: [EliteRef]}（只放小資料）
    hall_of_fame: list[EliteRef]
    hypotheses: Annotated[dict, merge_by_id]       # {hyp_id: Hypothesis}
    lessons_digest: str                            # ≤ 800 tokens，反思節點覆寫
    reflection_raw: Optional[dict]                 # LLM 反思原始輸出（驗證後清空；全文另存 artifact）
    pitfalls: list[str]                            # 常見錯誤（除錯統計彙整）
    gen_stats: Annotated[list[dict], cap_list(50)] # 每代摘要（觀測用）
    best_fitness_hist: list[float]
    ledger_id: str                                 # 全域試驗帳本 (DB) 句柄
    n_trials: int                                  # 本 run 計數（權威值在帳本）
    n_eff: float
    counters: Annotated[dict, sum_costs]           # debug_iters / improve / reideate / oos_peeks / lockbox
    costs: Annotated[dict, sum_costs]              # usd / tokens / backtests
    wall_start: float
    budget: Budget
    finalists: list[dict]
    champion: Optional[dict]
    review_ref: Optional[str]
    human_decision: Optional[dict]
    stop_reason: Optional[str]
```

### 3.2 Send 扇出時每個 worker 拿到的 payload 與子圖 State

```python
class VariantTask(TypedDict):
    """Send 扇出時每個 worker 拿到的 payload（只含小資料與 artifact key）。"""
    run_id: str
    variant_id: str
    generation: int
    slot: int
    island_id: int
    operator: Literal["mutate_logic", "crossover", "hypothesis_test", "fresh"]
    parent_ids: list[str]
    parent_code_refs: list[str]
    hypothesis: Optional[dict]        # hypothesis_test 時帶 {hyp_id, predicted, test_plan}
    instruction: str                  # 由程式模板填寫的突變指令
    lessons_digest: str
    pitfalls: list[str]
    data_snapshot_id: str
    engine_version: str
    models: dict                      # {"codegen": "...", "debug": "..."}
    limits: dict                      # {max_debug, max_tokens, timeout_s}


class WorkerState(TypedDict):
    task: VariantTask
    code_ref: Optional[str]
    last_error: Optional[dict]        # {kind, msg, trace_ref, hash}
    error_hashes: list[str]
    error_kinds: list[str]
    debug_attempts: int
    eval_out: Optional[dict]
    # ↓ 與父圖同名的鍵 → 子圖結束時寫回父圖（經父圖 reducer 累加）
    inbox: Annotated[list[VariantResult], operator.add]
    costs: Annotated[dict, sum_costs]
    counters: Annotated[dict, sum_costs]
```

**設計重點**
- `inbox` 是**每一代**的扇入緩衝：worker 子圖回傳 `{"inbox": [result]}`，父圖用 `operator.add` 累加；`select_elites` 消化完後回傳 `Overwrite([])` 清空，避免 state 無限變大（已實測：最後 `inbox` 長度是 0）。
- 子圖 `WorkerState` 和父圖只共用 `inbox / costs / counters` 三個鍵。子圖結束時，這三個鍵經父圖 reducer 合併；`task / code_ref / eval_out` 等子圖私有鍵不會污染父圖。
- 大型資料（程式碼、IS 日報酬序列、equity、交易明細、完整 traceback）都存在 artifact store，state 只放 key（`code_ref / returns_ref / trace_ref`）。
- 試驗帳本（ledger）是外部 DB（Postgres），權威計數在帳本；state 的 `n_trials` 只是快取。worker 以 `variant_id = sha1(run_id|gen|slot)` 做**冪等 upsert**，節點重試或從 checkpoint 恢復時不會重複計數。

---

## 4. 節點清單

| 節點 | 類型 / 模型等級 | 輸入（讀 state） | 輸出（寫 state） |
|---|---|---|---|
| `init_run` | 確定性 | run_id, data_snapshot_id, budget | islands 空殼、wall_start、版本資訊；ledger 開 run 記錄 |
| `budget_gate` + `route_budget` | 確定性 | costs, budget, n_trials, gen_stats | 路由：plan_generation / select_finalists |
| `plan_generation` | 確定性（RNG 以 run_id+gen 為種子） | islands, hypotheses, lessons_digest, pitfalls | `plan: list[VariantTask]`, generation+1 |
| `fan_out`（conditional edge） | 確定性 | plan | `list[Send("variant_worker", …)]` |
| `variant_worker.w_codegen` | **LLM-中**（`claude-sonnet-5`，effort medium，structured output：`{code, param_space, tags_declared, rationale}`） | task（親代程式碼 artifact、operator 模板、lessons、pitfalls、假說） | code_ref, costs |
| `variant_worker.w_check_and_run` | 確定性 + 沙盒 | code_ref | eval_out 或 last_error（含 error hash） |
| `variant_worker.w_debug` | **LLM-小**（`claude-haiku-4-5`，structured output：`{patched_code, change_summary}`） | code、traceback 摘要、違反的規則（**不含任何績效**） | 新 code_ref, debug_attempts+1, error_hashes |
| `variant_worker.w_emit` | 確定性 | 全部子圖 state | `inbox:[VariantResult]`, counters；ledger upsert |
| `select_elites` | 確定性 | inbox, islands, hall_of_fame, hypotheses | islands, hall_of_fame, hypotheses 判定, n_trials, n_eff, gen_stats, `inbox=Overwrite([])` |
| `population_reflect` | **LLM-強**（`claude-opus-5`，adaptive thinking，effort high，structured output `ReflectionReport`） | 由程式建好的 `PopulationDigest`（見 §6） | reflection_raw, costs |
| `validate_reflection` | 確定性 | reflection_raw, hypotheses, 帳本 | hypotheses（proposed）、lessons_digest（程式渲染）、清空 reflection_raw |
| `island_maintenance` | 確定性 | islands, generation, best_fitness_hist | islands（遷移/重置） |
| `checkpoint_review` | HITL（`interrupt`，前面無副作用） | IS 儀表板 artifact | human_decision |
| `select_finalists` | 確定性 | hall_of_fame, n_eff, n_trials | finalists（含 dsr, dsr_raw, pbo, 預先登記順序） |
| `oos_evaluate` | 確定性 + 沙盒 | finalists | finalists[].oos, counters.oos_peeks |
| `pick_champion` | 確定性 | finalists | champion 或 stop_reason |
| `lockbox_eval` | 確定性 + 沙盒 | champion | champion.lockbox, counters.lockbox |
| `prepare_review` | 確定性（寫 HTML/JSON 報告 artifact） | 全部 | review_ref |
| `human_review` | HITL（`interrupt`） | review_ref | human_decision, phase="done" |

選擇模型等級的理由：codegen 呼叫次數最多（N×G），用中階模型；除錯只是語法或 API 層面的修補，用小模型就夠；反思每代只呼叫 1 次，而且是全系統資訊密度最高的推理，用強模型。系統共用 prompt 放最前面並加 `cache_control`，各呼叫才能共享快取。

---

## 5. routing functions 與 Send 扇出函式

### 5.1 預算閘門與世代規劃

```python
def budget_gate(s: EvoState) -> dict:
    return {}   # 純路由用節點（決策在 route_budget）


def _remaining(s: EvoState) -> dict:
    c, b = s.get("costs", {}), s["budget"]
    return {"usd": b["usd"] - c.get("usd", 0), "tokens": b["tokens"] - c.get("tokens", 0),
            "backtests": b["backtests"] - c.get("backtests", 0),
            "wall_s": b["wall_s"] - (now() - s["wall_start"]),
            "trials": MAX_TRIALS_PER_RUN - s["n_trials"]}


def _est_gen_cost(s: EvoState) -> dict:
    hist = [g for g in s.get("gen_stats", []) if g.get("usd")]
    per_var = (sum(g["usd"] for g in hist) / max(1, sum(g["n"] for g in hist))) if hist else 0.08
    return {"usd": 1.3 * (per_var * POP_PER_GEN + 0.35), "trials": POP_PER_GEN, "backtests": 16 * POP_PER_GEN}


def route_budget(s: EvoState) -> Literal["plan_generation", "select_finalists"]:
    r, e = _remaining(s), _est_gen_cost(s)
    if s["generation"] >= MAX_GENERATIONS:
        return "select_finalists"
    if r["usd"] < e["usd"] or r["trials"] < e["trials"] or r["backtests"] < e["backtests"] \
            or r["wall_s"] < 600 or r["tokens"] < 400_000:
        return "select_finalists"
    return "plan_generation"


def plan_generation(s: EvoState) -> dict:
    """確定性：挑親代、分配 operator、把待驗證假說排進 A/B 測試名額。"""
    gen = s["generation"] + 1
    rng = random.Random(hash((s["run_id"], gen)))
    testing = [h for h in s.get("hypotheses", {}).values() if h["status"] in ("proposed", "testing")]
    tasks: list[dict] = []
    ops = list(OPERATOR_MIX)
    weights = [OPERATOR_MIX[o] for o in ops]
    for slot in range(POP_PER_GEN):
        isl = slot % N_ISLANDS
        elites = s["islands"].get(isl, [])
        op = "fresh" if not elites else rng.choices(ops, weights)[0]
        if op == "hypothesis_test" and not testing:
            op = "mutate_logic"
        if op == "crossover" and len(elites) < 2:
            op = "mutate_logic"
        parents = [] if op == "fresh" else (
            rng.sample(elites, 2) if op == "crossover" else [max(elites, key=lambda e: e["fitness"] + rng.random() * 0.2)])
        hyp = None
        if op == "hypothesis_test":
            h = testing[slot % len(testing)]
            hyp = {"hyp_id": h["hyp_id"], "predicted": h["predicted"]}
        vid = hashlib.sha1(f"{s['run_id']}|{gen}|{slot}".encode()).hexdigest()[:12]
        tasks.append(VariantTask(
            run_id=s["run_id"], variant_id=vid, generation=gen, slot=slot, island_id=isl, operator=op,
            parent_ids=[p["variant_id"] for p in parents], parent_code_refs=[p["code_ref"] for p in parents],
            hypothesis=hyp, instruction=f"{op} template v3", lessons_digest=s["lessons_digest"],
            pitfalls=s["pitfalls"][-8:], data_snapshot_id=s["data_snapshot_id"],
            engine_version=s["engine_version"],
            models={"codegen": "claude-sonnet-5", "debug": "claude-haiku-4-5"},
            limits={"max_debug": MAX_DEBUG_ATTEMPTS, "max_tokens": 12000, "timeout_s": 300}))
    return {"plan": tasks, "generation": gen}


def fan_out(s: EvoState) -> list[Send]:
    """每個 VariantTask 變成一個 Send → 平行執行 variant_worker 子圖。"""
    return [Send("variant_worker", {"task": t, "inbox": [], "costs": {}, "counters": {},
                                    "error_hashes": [], "error_kinds": [], "debug_attempts": 0,
                                    "code_ref": None, "last_error": None, "eval_out": None})
            for t in s["plan"]]
```

### 5.2 精英選擇（達標與選擇規則寫死）

```python
def _fitness(m: dict) -> float:
    """寫死的適應度：穩健統計 + 複雜度懲罰 + 參數高原懲罰。"""
    return (m["median_sr"] + 0.5 * min(m["worst_sr"], 0.0)
            - LAMBDA_COMPLEXITY * (m["complexity"] / 100.0)
            - LAMBDA_PLATEAU * (1.0 - m["plateau_ratio"]))


def _eligible(m: dict) -> bool:
    return (m["n_trades"] >= GATE["min_trades"] and m["max_dd"] <= GATE["max_dd"]
            and m["pos_fold_ratio"] >= GATE["pos_fold_ratio"] and m["n_params"] <= GATE["max_params"]
            and m["complexity"] <= GATE["max_complexity"])


def select_elites(s: EvoState) -> dict:
    """確定性扇入：去重、計分、島內 niching 精英、名人堂、假說判定、帳本統計。"""
    results = s["inbox"]
    seen_hash = {e.get("signal_hash") for isl in s["islands"].values() for e in isl}
    islands = {k: list(v) for k, v in s["islands"].items()}
    scored = []
    for r in results:
        if r["status"] != "ok" or r["signal_hash"] in seen_hash or not _eligible(r["metrics"]):
            continue
        seen_hash.add(r["signal_hash"])
        f = _fitness(r["metrics"])
        corr = Stub.max_corr_to(r["returns_ref"], [e.get("returns_ref") for e in islands[r["island_id"]]])
        f_rank = f - LAMBDA_NOVELTY * max(0.0, corr - 0.6)       # 新穎性懲罰只用於排序
        scored.append((f_rank, f, corr, r))
    for isl in islands:
        pool = [(e["fitness"], e) for e in islands[isl]]
        pool += [(fr, {"variant_id": r["variant_id"], "island_id": isl, "fitness": f, "code_ref": r["code_ref"],
                       "tags": r["metrics"]["tags"], "returns_ref": r["returns_ref"],
                       "signal_hash": r["signal_hash"]})
                 for fr, f, c, r in scored if r["island_id"] == isl and c < CORR_DUP]
        pool.sort(key=lambda x: -x[0])
        chosen: list = []
        for _, e in pool:
            if len(chosen) >= ELITES_PER_ISLAND:
                break
            if all(Stub.max_corr_to(e.get("returns_ref"), [c.get("returns_ref")]) < CORR_NICHE for c in chosen):
                chosen.append(e)
        islands[isl] = chosen
    # 假說判定（預先登記的預測 vs. 實測 Δ）
    hyp_updates = {}
    parents_fit = {e["variant_id"]: e["fitness"] for isl in s["islands"].values() for e in isl}
    for r in results:
        hid = r["hypothesis_id"]
        if not hid or r["status"] != "ok":
            continue
        h = dict(s["hypotheses"][hid])
        base = parents_fit.get(r["parent_ids"][0]) if r["parent_ids"] else None
        if base is None:
            continue
        h["tests"] = h["tests"] + [{"variant_id": r["variant_id"], "delta": _fitness(r["metrics"]) - base}]
        h["status"] = _judge_hypothesis(h)
        hyp_updates[hid] = h
    stats = Stub.ledger_stats(s["ledger_id"])
    n_trials = s["n_trials"] + sum(r["counted_as_trial"] for r in results)
    all_elites = [e for v in islands.values() for e in v]
    hof = sorted({e["variant_id"]: e for e in s["hall_of_fame"] + all_elites}.values(),
                 key=lambda e: -e["fitness"])[:10]
    best = hof[0]["fitness"] if hof else float("-inf")
    return {"inbox": Overwrite([]),                         # 清空扇入緩衝
            "islands": islands, "hall_of_fame": hof, "hypotheses": hyp_updates,
            "n_trials": n_trials, "n_eff": max(stats["n_eff"], math.sqrt(n_trials)),
            "best_fitness_hist": s["best_fitness_hist"] + [best],
            "gen_stats": [{"gen": s["generation"], "n": len(results), "ok": len(scored), "best": best,
                           "dead": sum(r["status"] == "dead" for r in results), "usd": None}]}


def _judge_hypothesis(h: dict) -> str:
    """程式判定：≥3 次測試且 ≥2/3 方向一致且中位 Δ ≥ min_effect → confirmed。"""
    ds = sorted(t["delta"] for t in h["tests"])
    if len(ds) < 3:
        return "testing"
    sign_ok = sum(d > 0 for d in ds) / len(ds) if h["predicted"]["direction"] == "+" else sum(d < 0 for d in ds) / len(ds)
    med = ds[len(ds) // 2]
    if sign_ok >= 2 / 3 and abs(med) >= h["predicted"]["min_effect"]:
        return "confirmed"
    return "refuted" if len(ds) >= 4 else "testing"
```

### 5.3 世代後路由、決賽與冠軍規則

```python
def route_after_generation(s: EvoState) -> Literal["budget_gate", "checkpoint_review", "select_finalists"]:
    """寫死的停止規則：IS 達標提早停（再搜只會墊高 DSR 門檻）、停滯、定期 HITL。"""
    g, hist = s["generation"], s["best_fitness_hist"]
    if g >= MIN_GENERATIONS and s["hall_of_fame"]:
        top = s["hall_of_fame"][0]
        if Stub.dsr(top.get("returns_ref"), s["n_eff"], 0.25) >= GATE["dsr_min"] and g >= MIN_GENERATIONS + 2:
            return "select_finalists"
    if len(hist) > STAGNATION_GENS and max(hist[-STAGNATION_GENS:]) - max(hist[:-STAGNATION_GENS]) < STAGNATION_EPS \
            and g >= MIN_GENERATIONS:
        return "select_finalists"
    if g % HITL_EVERY == 0:
        return "checkpoint_review"
    return "budget_gate"


def checkpoint_review(s: EvoState) -> dict:
    """期中 HITL：只呈現 IS 儀表板（報告已由上游寫入 artifact）；本節點無副作用。"""
    decision = interrupt({"type": "midrun", "generation": s["generation"], "n_trials": s["n_trials"],
                          "n_eff": s["n_eff"], "hof": [e["variant_id"] for e in s["hall_of_fame"][:5]]})
    return {"human_decision": decision}


def route_after_checkpoint(s: EvoState) -> Literal["budget_gate", "select_finalists"]:
    d = s.get("human_decision") or {}
    return "select_finalists" if d.get("action") == "stop" else "budget_gate"


def select_finalists(s: EvoState) -> dict:
    """確定性：從名人堂挑 ≤3 個彼此低相關者，預先登記順序（依 IS DSR），之後才看 OOS。"""
    cands = []
    for e in s["hall_of_fame"]:
        d = Stub.dsr(e.get("returns_ref"), s["n_eff"], 0.25)          # 主：有效試驗數
        d_raw = Stub.dsr(e.get("returns_ref"), s["n_trials"], 0.25)   # 敏感度：原始試驗數
        cands.append({**e, "dsr": d, "dsr_raw": d_raw})
    cands.sort(key=lambda e: -e["dsr"])
    fin = []
    for e in cands:
        if len(fin) >= N_FINALISTS:
            break
        if all(Stub.max_corr_to(e.get("returns_ref"), [f.get("returns_ref")]) < 0.7 for f in fin):
            fin.append(e)
    pbo = Stub.pbo_cscv([e.get("returns_ref") for e in s["hall_of_fame"]])
    for f in fin:
        f["pbo"] = pbo
    return {"finalists": fin, "phase": "finalize"}


def oos_evaluate(s: EvoState) -> dict:
    """每個 finalist 恰好 1 次 OOS（帳本記錄 oos_peek，重跑冪等）。結果不回饋 LLM。"""
    out = []
    for f in s["finalists"]:
        o = Stub.oos_eval(f["code_ref"])
        out.append({**f, "oos": o})
    return {"finalists": out, "counters": {"oos_peeks": len(out)}}


def pick_champion(s: EvoState) -> dict:
    """依『預先登記順序』取第一個通過全部門檻者；OOS 只當 pass/fail 過濾器。"""
    for f in s["finalists"]:
        o = f["oos"]
        passed = (f["dsr"] >= GATE["dsr_min"] and f["dsr_raw"] >= GATE["dsr_raw_min"]
                  and f["pbo"] <= GATE["pbo_max"]
                  and o["sr"] >= GATE["oos_sr_min"] and o["net_ret"] > 0
                  and o["max_dd"] <= GATE["oos_dd_mult_max"] * GATE["max_dd"])
        if passed:
            return {"champion": f}
    return {"champion": None, "stop_reason": "no_finalist_passed"}


def route_champion(s: EvoState) -> Literal["lockbox_eval", "prepare_review"]:
    return "lockbox_eval" if s["champion"] else "prepare_review"


def lockbox_eval(s: EvoState) -> dict:
    lb = Stub.lockbox_eval(s["champion"]["code_ref"])      # 每資料快照僅一次；結果只進報告
    return {"champion": {**s["champion"], "lockbox": lb}, "counters": {"lockbox": 1}}


def prepare_review(s: EvoState) -> dict:
    """有副作用（寫報告 artifact）的步驟放在 interrupt 節點之前。"""
    return {"review_ref": f"reports/{s['run_id']}.html"}


def human_review(s: EvoState) -> dict:
    decision = interrupt({"type": "final", "review_ref": s["review_ref"],
                          "champion": (s["champion"] or {}).get("variant_id"),
                          "options": ["approve_paper_trading", "reject", "archive"]})
    return {"human_decision": decision, "phase": "done"}
```

**寫死的規則摘要**
- **資格**：總交易數 ≥ 100、IS max DD ≤ 35%、正報酬 fold 比例 ≥ 50%、參數數 ≤ 6、AST 複雜度 ≤ 400。
- **fitness** = `median_fold_SR + 0.5·min(worst_fold_SR,0) − 0.15·complexity/100 − 0.5·(1 − plateau_ratio)`。
- **排序分數** = fitness − 1.0·max(0, 與同島精英的最大報酬相關 − 0.6)。新穎性懲罰只影響排序，不改寫 fitness 本身。
- **島內 niching**：精英彼此 ρ < 0.85；ρ ≥ 0.95 或 signal hash 相同視為重複。
- **提早停止**：第 7 代以後，只要名人堂第一名的 DSR（用 N_eff 計）≥ 0.95 就停。理由是**再搜下去只會墊高自己的門檻**。另外，連續 4 代最佳 fitness 進步 < 0.05 也停。
- **冠軍**：必須同時通過 DSR_Neff ≥ 0.95、DSR_rawN ≥ 0.80、PBO ≤ 0.30、OOS SR ≥ 0.5、OOS 淨報酬 > 0、OOS DD ≤ 1.5×35%，並依**預先登記順序**取第一個通過者，不挑 OOS 最好的。

---

## 6. 反思機制細節（族群層級）

### 6.1 輸入：`PopulationDigest`（全部由程式產生，LLM 不能自己查資料）

```python
class ContrastPair(TypedDict):
    pair_id: str                 # "g7:p3"
    island_id: int
    operator: str                # mutate_logic / crossover / hypothesis_test
    parent_id: str
    child_id: str
    code_diff: str               # unified diff，截 ≤ 60 行
    tags_added: list[str]        # AST 標籤器產生：atr_stop / funding / vol_filter / session / regime_ma ...
    tags_removed: list[str]
    delta: dict                  # {"median_sr": +0.12, "worst_sr": +0.30, "n_trades": -40, "complexity": +35}
    delta_per_fold: list[float]  # 每個 fold 的 ΔSR（看是否只靠某個 regime）

class TagAttribution(TypedDict):  # 跨所有親→子配對的確定性統計
    tag: str
    n_added: int
    median_delta_fitness: float
    sign_consistency: float      # Δ>0 的比例
    boot_ci90: tuple[float, float]
    islands_seen: list[int]

class PopulationDigest(TypedDict):
    generation: int
    island_summaries: list[dict] # 每島：best/median fitness（以排名與 Δ 呈現）、多樣性、停滯代數
    contrast_pairs: list[ContrastPair]      # 本代 ≤ 24 組：Δ 最大的正/負各半 + 所有 hypothesis_test
    top_vs_bottom: list[dict]               # 每島 top-3 vs bottom-3 的 tag 與描述子差異
    tag_attribution: list[TagAttribution]   # 累積到本代
    hypothesis_outcomes: list[dict]         # 上一輪假說的「預測 vs 實測」與程式判定
    debug_histogram: dict                   # 錯誤種類統計（只給反思看，讓它知道哪些寫法常壞）
    # 刻意不提供：任何 OOS / lockbox 數字、絕對 Sharpe 排行榜、上一輪反思的自由文字
```

### 6.2 輸出：`ReflectionReport`（structured output）

```python
class HypothesisOut(BaseModel):
    hyp_id: str
    statement: str                       # "在 1h 趨勢策略加 ATR(14)×k 移動停損，可改善最差 fold"
    mechanism: str                       # 市場機制解釋（≤ 60 字），例："截斷高波動期的尾部虧損"
    evidence_pair_ids: list[str]         # ≥ 2 個，必須存在於 digest
    evidence_tags: list[str]
    predicted: dict                      # {"edit": "atr_stop", "metric": "worst_sr", "direction": "+", "min_effect": 0.10}
    test_plan: str                       # 給 codegen 的具體改法（套到哪類親代）
    confidence: Literal["low", "med", "high"]

class ReflectionReport(BaseModel):
    hypotheses: list[HypothesisOut]      # ≤ 3 個新假說
    anti_patterns: list[dict]            # {"pattern": "...", "evidence_pair_ids": [...]} 供 fresh 子代避開
    retire: list[str]                    # 建議退休的假說 id（程式仍依數據決定）
    island_notes: dict[int, str]         # 只供觀測，不影響路由
```

### 6.3 如何避免「事後編故事」

1. **證據先由程式算好，LLM 只做詮釋**。tag 歸因表與對照組 Δ 都是確定性統計，LLM 看不到原始 K 線，不能「自己發現規律」。
2. **引用驗證**（`validate_reflection`）：每個假說至少引用 2 個存在的 pair_id；被引用配對的 Δ 方向必須和 `predicted.direction` 一致，否則直接拒絕並標 `status="rejected"`。如果該 tag 的 bootstrap CI 跨 0，只能標成 `low` 信心，而且只拿探索名額。
3. **事先登記 + 程式判定**：每個假說都帶可否證的預測（改什麼、影響哪個指標、方向、最小效果量）。`plan_generation` 把 25% 名額排成 `hypothesis_test`，把同一個改動套到 ≥3 個親代、跨 ≥2 島。`_judge_hypothesis` 判定規則：≥2/3 方向一致且中位 Δ ≥ min_effect 才算 confirmed；測到 4 次仍不達標就 refuted。
4. **lessons 由程式渲染**：`render_lessons()` 只把 confirmed / refuted 的假說轉成一行帶 n 與中位 Δ 的條目放進 codegen prompt。LLM 的自由敘事不會進下一代，因此不會出現「自我強化的故事」。
5. **反思增益（reflection lift）監控**：程式每 3 代比較「hypothesis_test 子代勝過親代的比率」和「無引導 mutate_logic 子代勝過親代的比率」。如果連續兩個窗口 lift ≤ 0，程式自動把 hypothesis_test 名額從 25% 降到 10%，並在報告標示「反思無效」。這等於對反思本身做持續的 A/B 消融，直接回應 R2。
6. **遮蔽**：反思看不到 OOS 和 lockbox；以排名和 Δ 取代絕對 Sharpe，降低錨定效應。上一輪反思的原文也不回灌，只看假說判定結果。
7. **數量上限**：每代最多 3 個新假說，同時活躍 ≤ 6 個，並以 `(edit, metric)` 去重，避免假說彩券式地大量投放。

> 註：假說驗證本身也有多重比較（6 個假說裡可能有 1 個靠運氣被判 confirmed）。但假說只影響**搜尋方向**，不影響最終門檻；最終仍由 DSR / PBO / OOS 把關，而且 hypothesis_test 子代同樣記入試驗帳本。

---

## 7. 除錯機制細節

```python
# ───────────────────────────── worker 子圖（除錯短迴圈） ─────────────────────────────
def w_codegen(s: WorkerState) -> dict:
    out = Stub.llm_codegen(s["task"])
    code_ref = f"code/{s['task']['variant_id']}/v0"
    return {"code_ref": code_ref, "debug_attempts": 0, "error_hashes": [], "error_kinds": [],
            "costs": {"usd": out["usd"], "tokens": out["tokens"]}}


def _err_hash(err: dict) -> str:
    import re
    norm = re.sub(r"\d+", "N", f"{err['kind']}:{err['msg']}")
    return hashlib.sha1(norm.encode()).hexdigest()[:10]


def w_check_and_run(s: WorkerState) -> dict:
    """確定性：AST/schema/白名單 → 沙盒（截斷不變性 + IS walk-forward + 參數鄰域）。"""
    err = Stub.static_checks(s["code_ref"])
    if err is None:
        r = Stub.sandbox_eval(s["code_ref"])
        if "error" in r:
            err = r["error"]
        else:
            return {"last_error": None, "eval_out": r, "costs": {"backtests": r["backtests"]}}
    err["hash"] = _err_hash(err)
    return {"last_error": err, "eval_out": None}


def w_debug(s: WorkerState) -> dict:
    """小模型；只看程式碼 + traceback + 違反的規則，看不到任何績效。"""
    out = Stub.llm_debug(s["code_ref"], s["last_error"])
    n = s["debug_attempts"] + 1
    return {"code_ref": f"code/{s['task']['variant_id']}/v{n}", "debug_attempts": n,
            "error_hashes": s["error_hashes"] + [s["last_error"]["hash"]],
            "error_kinds": s["error_kinds"] + [s["last_error"]["kind"]],
            "costs": {"usd": out["usd"], "tokens": out["tokens"]}, "counters": {"debug_iters": 1}}


def route_after_run(s: WorkerState) -> Literal["w_debug", "w_emit"]:
    err = s["last_error"]
    if err is None:
        return "w_emit"
    if s["debug_attempts"] >= s["task"]["limits"]["max_debug"]:
        return "w_emit"                              # 放棄 → dead
    if err["hash"] in s["error_hashes"]:
        return "w_emit"                              # 同一錯誤重複 → 不再燒錢
    if err["kind"] == "LookaheadDetected" and "LookaheadDetected" in s["error_kinds"]:
        return "w_emit"                              # 前視偏差只給一次修正機會
    return "w_debug"


def w_emit(s: WorkerState) -> dict:
    t = s["task"]
    ok = s["eval_out"] is not None
    m = s["eval_out"]["metrics"] if ok else {}
    status = "ok" if ok else "dead"
    if ok and m.get("n_trades", 0) < 30:
        status = "degenerate"
    res: VariantResult = {
        "variant_id": t["variant_id"], "generation": t["generation"], "island_id": t["island_id"],
        "operator": t["operator"], "parent_ids": t["parent_ids"],
        "hypothesis_id": (t["hypothesis"] or {}).get("hyp_id"), "status": status,
        "code_ref": s["code_ref"], "returns_ref": s["eval_out"]["returns_ref"] if ok else None,
        "signal_hash": s["eval_out"]["signal_hash"] if ok else None, "metrics": m,
        "debug_attempts": s["debug_attempts"], "error_hashes": s["error_hashes"],
        "counted_as_trial": ok,                     # 有績效就是試驗（含 degenerate）；dead 不算
    }
    Stub.ledger_upsert(res)                         # 冪等 upsert：節點重跑不會重複計數
    kind = "reideate" if t["operator"] == "fresh" else "improve"
    return {"inbox": [res], "counters": {kind: 1}}


def build_worker():
    g = StateGraph(WorkerState)
    g.add_node("w_codegen", w_codegen, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("w_check_and_run", w_check_and_run,
               retry_policy=RetryPolicy(max_attempts=2, retry_on=(TimeoutError, ConnectionError)))
    g.add_node("w_debug", w_debug, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("w_emit", w_emit)
    g.add_edge(START, "w_codegen")
    g.add_edge("w_codegen", "w_check_and_run")
    g.add_conditional_edges("w_check_and_run", route_after_run, ["w_debug", "w_emit"])
    g.add_edge("w_debug", "w_check_and_run")
    g.add_edge("w_emit", END)
    return g.compile()
```

- **錯誤分類**：`SyntaxError / ImportBlocked / SchemaError(PARAM_SPACE) / LookaheadDetected(截斷不變性失敗) / RuntimeError / OutputInvalid(NaN、超出[-1,1]、長度不符) / Timeout / InfraError`。
- **error hash** = `sha1(kind + 去掉數字的訊息)`。同一個 worker 內重複出現就立即放棄（`route_after_run`），不再燒 token。
- **修復前先跑確定性自動修補**（MVP 可以先不做）：`clip(-1,1)`、`reindex(df.index)`、dtype 轉換。能自動修的就不叫 LLM。
- **除錯 LLM 看不到績效**：輸入只有程式碼、traceback 摘要（≤ 40 行）和違反的規則說明，因此它無法藉修 bug 之名偷偷做績效優化。
- **防止「修 bug 變改策略」**：patch 後用 AST diff 檢查，改動行數超過 30%，或新增了原本沒有的指標 tag，就視為重寫，直接判 dead。
- **前視偏差只給一次修正機會**：第二次 `LookaheadDetected` 直接判 dead，避免 LLM 反覆學習如何繞過檢測器（截斷不變性是行為層面的測試，本來就難以繞過）。
- **不算試驗**：debug 過程中沒有產生績效，不記入試驗；修好後的那一次績效評估，才是該變體唯一的 1 次試驗。
- **故障隔離**：worker 永遠不 raise，失敗就回傳 `status="dead"`。沙盒啟動失敗之類的基礎設施錯誤交給 `RetryPolicy(retry_on=(TimeoutError, ConnectionError))`，不送進 LLM 除錯。LangGraph 在同一 superstep 中，已完成的兄弟任務寫入會以 pending writes 保存在 checkpoint，恢復時只重跑失敗的那個。
- **族群層級除錯回饋**：`select_elites` 統計本代 error hash 直方圖，前 8 名轉成 `pitfalls` 注入下一代 codegen prompt。這是 debug 資訊往上游流動的唯一管道。
- **斷路器**：某一代 dead 比率 > 50%（例如引擎升版造成 API 不相容）時，`route_after_generation` 應直接導向 `checkpoint_review` 讓人處理（骨架未實作，MVP 必做）。

---

## 8. 過擬合控制如何嵌入（重點：大量試驗下的多重檢定）

平行演化的原罪是**試驗數爆炸**：N=16 × G=20 = 320 次試驗。本架構的立場是：**不假裝沒有試驗，而是把全部試驗計入，並讓門檻隨之上升；同時用演算法設計減少「有效」試驗數與選擇壓力。**

### 8.1 資料切分（BTC/USDT 永續 1h 範例）
- IS：2020-01 ～ 2024-06，expanding walk-forward 6 folds，purge = 最大 lookback 根數，embargo = 24 根。
- OOS：2024-07 ～ 2025-06，只給 ≤3 個決賽者各 1 次。
- lockbox：2025-07 ～ 2026-06，只給冠軍 1 次，而且每個資料快照終身只用 1 次。
- ledger 以 `data_snapshot_id` 為範圍累計 `oos_peeks`；同一快照累計超過 9 次（約 3 個 run）就拒絕再開新 run，必須換新資料切分或等新資料累積。

### 8.2 試驗帳本 → DSR / PBO
| 機制 | 做法 |
|---|---|
| 全域帳本 | 每個有績效的變體（含 degenerate、hypothesis_test、crossover）記 1 筆：variant_id、parent、island、operator、prompt/模型/code/data/engine hash、IS 日報酬 artifact、SR、偏態、峰態、T。修語法不記。範圍跨 run（同快照、同標的、同頻率） |
| 有效試驗數 N_eff | 對所有試驗的 IS 日報酬做相關矩陣，階層式聚類（距離 √(½(1−ρ))，ONC 或固定門檻 ρ=0.7），N_eff = 群數。演化產生的子代高度相關，N_eff 通常遠小於 raw N |
| DSR | 用 Bailey & López de Prado 公式，以 N_eff 與跨試驗 SR 變異數計算 SR₀；門檻 DSR_Neff ≥ 0.95，另以 raw N 做敏感度檢查（DSR_rawN ≥ 0.80） |
| PBO | 對名人堂 + 各島精英（約 20–40 條）的報酬矩陣做 CSCV（S=16 區塊），PBO ≤ 0.30 |
| 門檻曲線預告 | run 開始前報告 E[max SR | null] ≈ σ_SR·[(1−γ)Φ⁻¹(1−1/N)+γΦ⁻¹(1−1/(Ne))]。N=20 → 1.90σ、N=100 → 2.53σ、N=300 → 2.90σ。讓人清楚知道「多搜 10 代要付出多少門檻」 |
| 試驗預算 | `MAX_TRIALS_PER_RUN=400` 寫死；budget_gate 同時檢查剩餘試驗數 |
| 提早停止 | IS 一達標就停（§5.3），不「加碼搜更好的」 |

### 8.3 降低選擇壓力與過擬合的演算法設計
- **fitness 用穩健統計**：以 fold 中位數 SR 加最差 fold 懲罰，不用平均或總報酬。只靠單一 regime 賺錢的策略分數會很低。
- **參數高原（plateau）**：每個變體在宣告的預設參數跑一次，再對每個參數 ±1 步跑鄰域評估（≤ 12 次引擎呼叫）。鄰域**只拿來算 plateau_ratio 當懲罰，不從鄰域挑最佳參數**，所以不增加選擇自由度。
- **防 plateau 作弊**：擾動後部位序列完全不變的參數判為「死參數」，計入複雜度，而且不參與 plateau 計算，避免 LLM 塞入無作用參數把 plateau 灌到 1。
- **複雜度懲罰與硬上限**：參數 ≤ 6、AST 節點 ≤ 400。演化會自然出現 bloat（複雜度爬升），懲罰項可以壓住它。
- **去重不計新試驗**：signal hash（部位序列四捨五入後的 hash）與既有試驗相同，就沿用舊結果、不算新試驗，同時也省成本。
- **OOS 只當過濾器**：決賽者在看 OOS **之前**就依 IS DSR 排好序，冠軍是「第一個通過者」，不是「OOS 最好者」。OOS 結果不回饋任何 LLM、不進帳本的選擇統計。
- **lockbox**：只給冠軍看 1 次，結果只進人工報告，不改變冠軍。

### 8.4 避免族群收斂到同一種過擬合策略
- **島嶼模型（4 島）**，每島用不同的種子家族：趨勢 / 均值回歸 / 資金費率與基差 / 波動突破。遷移每 3 代一次、環狀進行，且移民必須和目標島精英 ρ < 0.7。
- **新穎性懲罰**（與同島精英相關 > 0.6 開始扣分）加上 **niching**（精英彼此 ρ < 0.85）。
- **停滯島重置**：島內 4 代無進步，就以 `fresh` operator 重建。prompt 帶入已 refuted 假說與 anti_patterns 當負面約束。
- **收斂監控**：精英平均兩兩相關 > 0.7，或所有島的前 3 名 tag 集合 Jaccard > 0.6，就強制重置最差島並記錄在 gen_stats。
- **v2 選項**：MAP-Elites 行為格（交易頻率 × 持倉風格）；影子 fold（LLM 與反思完全看不到的 IS 區段，只用於決賽者篩選）；各島用不同 fold 權重的「目標函數 bagging」。

---

## 9. 建圖程式碼骨架（真實 LangGraph API）

```python
# ───────────────────────────── 建圖 ─────────────────────────────
def build_graph(checkpointer=None):
    g = StateGraph(EvoState)
    g.add_node("init_run", init_run)
    g.add_node("budget_gate", budget_gate)
    g.add_node("plan_generation", plan_generation)
    g.add_node("variant_worker", build_worker())            # 子圖作為節點，被 Send 扇出
    g.add_node("select_elites", select_elites)
    g.add_node("population_reflect", population_reflect, retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("validate_reflection", validate_reflection)
    g.add_node("island_maintenance", island_maintenance)
    g.add_node("checkpoint_review", checkpoint_review)
    g.add_node("select_finalists", select_finalists)
    g.add_node("oos_evaluate", oos_evaluate)
    g.add_node("pick_champion", pick_champion)
    g.add_node("lockbox_eval", lockbox_eval)
    g.add_node("prepare_review", prepare_review)
    g.add_node("human_review", human_review)

    g.add_edge(START, "init_run")
    g.add_edge("init_run", "budget_gate")
    g.add_conditional_edges("budget_gate", route_budget, ["plan_generation", "select_finalists"])
    g.add_conditional_edges("plan_generation", fan_out, ["variant_worker"])
    g.add_edge("variant_worker", "select_elites")            # 所有 Send 任務完成後才執行一次
    g.add_edge("select_elites", "population_reflect")
    g.add_edge("population_reflect", "validate_reflection")
    g.add_edge("validate_reflection", "island_maintenance")
    g.add_conditional_edges("island_maintenance", route_after_generation,
                            ["budget_gate", "checkpoint_review", "select_finalists"])
    g.add_conditional_edges("checkpoint_review", route_after_checkpoint, ["budget_gate", "select_finalists"])
    g.add_edge("select_finalists", "oos_evaluate")
    g.add_edge("oos_evaluate", "pick_champion")
    g.add_conditional_edges("pick_champion", route_champion, ["lockbox_eval", "prepare_review"])
    g.add_edge("lockbox_eval", "prepare_review")
    g.add_edge("prepare_review", "human_review")
    g.add_edge("human_review", END)
    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "run-001"}, "recursion_limit": 300}
    init = {"run_id": "run-001", "data_snapshot_id": "btcusdt_1h_sha256:ab12", "engine_version": "bt-1.4.0",
            "prompt_versions": {"codegen": "v3", "reflect": "v2", "debug": "v1"}, "ledger_id": "ledger-main",
            "budget": BUDGET, "hypotheses": {}, "inbox": [], "costs": {}, "counters": {}, "gen_stats": [],
            "plan": []}
    res = app.invoke(init, cfg)
    while "__interrupt__" in res:
        payload = res["__interrupt__"][0].value
        print("INTERRUPT:", payload["type"], {k: payload[k] for k in payload if k != "type"})
        resume = {"action": "continue"} if payload["type"] == "midrun" else {"action": "reject", "note": "demo"}
        res = app.invoke(Command(resume=resume), cfg)
    print("gen", res["generation"], "trials", res["n_trials"], "counters", res["counters"],
          "costs", {k: round(v, 3) for k, v in res["costs"].items()}, "champion",
          (res["champion"] or {}).get("variant_id"), "stop", res["stop_reason"], "inbox", len(res["inbox"]))
```

**已查證的 API 行為**（讀 `langgraph-1.2.12` wheel 原始碼並實跑骨架）
- `Send(node, arg)` 來自 `langgraph.types`，由 `add_conditional_edges` 的 path 函式回傳 `list[Send]`，可指向**已編譯的子圖節點**，`arg` 就是子圖的輸入 state。
- 子圖輸出中與父圖同名的鍵（`inbox / costs / counters`）會經父圖 reducer 合併，私有鍵被忽略。
- `add_edge("variant_worker", "select_elites")`：同一 superstep 的全部 Send 任務完成後，`select_elites` 只執行一次，這就是扇入。
- `Overwrite([])` 能繞過 `operator.add` reducer 清空 channel。同一 superstep 只能有一個 Overwrite，這裡只有 `select_elites` 會寫。
- `interrupt(value)` 需要 checkpointer + thread_id；resume 時整個節點會從頭重跑，所以 `checkpoint_review` / `human_review` 內部沒有副作用，寫報告放在前一個節點 `prepare_review`。
- 1.2.12 的 `interrupt()` 支援 `response_schema=` 參數，可以把人工回覆也做 schema 驗證（符合約束 10）。
- 需要設定 `recursion_limit`。每代 7 個父圖 superstep（子圖內部步數不計入父圖），20 代約 140 步；已實測 20 代在 `recursion_limit=300` 內跑完。

**執行範例**
```python
app = build_graph(PostgresSaver.from_conn_string(DB_URL))   # MVP 可用 SqliteSaver / InMemorySaver
cfg = {"configurable": {"thread_id": run_id}, "recursion_limit": 300}
res = app.invoke(initial_state, cfg)
while "__interrupt__" in res:
    payload = res["__interrupt__"][0].value
    res = app.invoke(Command(resume=ask_human(payload)), cfg)
```
平行度控制：`app.invoke(..., {"max_concurrency": 8, ...})` 限制同時執行的任務數，沙盒池再做一層信號量限制。

---

## 10. 成本估算

單價取自目前的 Claude API 牌價（每百萬 token 輸入/輸出）：Opus 5 $5/$25、Sonnet 5 $2/$10、Haiku 4.5 $1/$5。

| 項目 | 次數 | 每次 tokens（入/出） | 每次 USD |
|---|---|---|---|
| codegen（Sonnet 5） | N×G | 8k / 2k（系統 prompt 3k 可快取） | ≈ $0.036 |
| debug（Haiku 4.5） | 平均 0.5 次/變體 | 4k / 1.2k | ≈ $0.010 |
| 族群反思（Opus 5, effort high） | G | 30k / 4k（含思考） | ≈ $0.25 |
| 初始種子發想 | 1 | 併入第 1 代的 fresh codegen | — |
| 回測引擎 | 每變體 ≈ 14 次（截斷測試 + WF + 鄰域） | — | 算力成本另計 |

| 設定 | 試驗數（raw） | LLM 成本（含 1.3× 重試/結構化修正） | tokens | 引擎回測數 | wall-clock（8 並行沙盒） |
|---|---|---|---|---|---|
| **MVP：N=8, G=10** | 80 | ≈ $5 | ≈ 1.2M | ≈ 1.1k | ≈ 30 分鐘 |
| **預設：N=16, G=20** | 320 | ≈ $25 | ≈ 4.5M | ≈ 4.5k | ≈ 80 分鐘 |
| 大型：N=32, G=30 | 960 | ≈ $70 | ≈ 13M | ≈ 13k | ≈ 3.5 小時 |

- 預設組合每代約 16×$0.041 + $0.25 ≈ $0.9，其中反思占約 27%，codegen 約 65%。
- 預算上限寫死為 `BUDGET = {usd: 60, tokens: 6M, wall_s: 4h, backtests: 6000}`，加上 `MAX_TRIALS_PER_RUN=400`。`route_budget` 用最近幾代的實際單價 ×1.3 預估下一代成本，不夠就提早進入決賽。
- **最大的成本不是 USD，而是試驗數**。大型設定的 raw N ≈ 960，SR₀ 約 3.1σ，大多數策略都過不了，所以預設組合是刻意選的甜蜜點。
- 省錢手段：共用 system prompt 做 prompt caching；signal hash 去重；確定性自動修補；錯誤 hash 早停；非即時需求時反思可改用 Batch API（半價，但會拉長 wall-clock）。

---

## 11. 優點、弱點、失敗模式

**優點**
- 探索能力最強：多島、多家族、crossover、fresh 重發想，能跳出單線迭代的局部最佳。
- 反思有對照組：族群層級的比較（親→子 Δ、tag 歸因、跨島重複驗證）比單一策略的自我檢討資訊量高得多；而且反思可被否證、成效被量化（reflection lift）。
- 平行帶來 wall-clock 效率：N 個變體同時跑，除錯迴圈局部化，一個變體壞掉不影響其他變體。
- 多重檢定內建在流程裡：帳本、N_eff、DSR、PBO、預先登記順序、提早停止，都寫在路由程式中。
- 可重現：RNG 以 `(run_id, gen)` 為種子，LLM 輸出以 prompt hash 快取成 artifact，可以「無 LLM 重播」整個演化過程。

**弱點**
- 試驗數天生大，DSR 門檻高。最可能的結局是「誠實地找不到」，對使用者體驗是挑戰。
- 元件多：島嶼、niching、假說帳本、N_eff 聚類、CSCV、子圖、兩個 HITL。工程量明顯高於單線架構。
- USD 與算力成本約為單線架構的 5–15 倍。
- 選擇壓力持續作用在同一組 IS folds 上，歷經 20 代之後，這些 folds 事實上已經被「訓練」過。PBO 與 OOS 能偵測，但無法預防。
- N_eff 的估計方法（聚類門檻）本身就是一個自由度，可能被調整到讓結果「好看」。必須固定寫死並同時報告 raw N。

**失敗模式與對策**
| 失敗模式 | 徵兆 | 對策 |
|---|---|---|
| DSR 門檻過高、無人通過 | `stop_reason=no_finalist_passed` | run 前預告門檻曲線；縮小 N×G；強化種子先驗；接受「沒有」是正確答案 |
| 族群塌縮成同一種策略（如趨勢 + ATR 停損） | 精英平均相關 > 0.7、tag Jaccard 高 | niching、新穎性懲罰、強制重置最差島 |
| 適應度作弊（死參數、極低頻交易壓低 fold 變異） | plateau ≈ 1、n_trades 貼近門檻 | 死參數偵測、最小交易數、交易頻率納入描述子 |
| 反思淪為假說彩券 | reflection lift ≤ 0 | 程式自動降低 hypothesis_test 名額；報告標示 |
| IS folds 被演化「訓練」 | IS 分數持續升但 PBO 升高 | 世代上限、提早停止、PBO 門檻；v2 加影子 fold |
| 除錯風暴（引擎升版等） | 單代 dead > 50% | 斷路器 → checkpoint_review |
| 重跑造成帳本重複計數 | 同 variant_id 多筆 | 冪等 upsert（主鍵 variant_id） |
| 單一 regime 依賴（2020–21 大多頭） | 某 fold ΔSR 主導 | 最差 fold 懲罰、delta_per_fold 呈現給反思 |
| 人工期中介入成為另一個自由度 | — | 人工動作寫進帳本（type=intervention），計入報告 |

---

## 12. 依評分標準 R1–R7 自評

| 標準 | 權重 | 分數 | 理由（誠實版） |
|---|---|---|---|
| R1 樣本外有效性 / 過擬合控制 | 25% | **7** | 多重檢定機制完整：跨 run 帳本、N_eff + raw N 雙重 DSR、CSCV-PBO、OOS 只做 pass/fail 且預先登記順序、lockbox 1 次、提早停止、plateau 與複雜度懲罰。但演化本質是大量試驗加上持續選擇壓力，IS folds 會被「訓練」；N_eff 估計有不確定性。機制能讓結果誠實，卻不保證找得到樣本外有效的策略。 |
| R2 反思品質 | 15% | **7** | 族群層級對照、程式算好的證據、引用核對、事先登記的預測、程式判定、lessons 由程式渲染、reflection lift 消融監控，大幅壓低編故事的空間。扣分：Δ 雜訊大，3–4 次測試的判定仍可能誤判；LLM 產生假說的品質本身無法保證。 |
| R3 除錯效率與穩健性 | 10% | **8** | 每個變體在子圖內局部除錯、小模型、看不到績效、error hash 早停、AST diff 防改策略、故障隔離、pending writes 恢復、族群層級 pitfalls 回饋。扣分：自動修補與斷路器在骨架中尚未實作。 |
| R4 探索能力 | 10% | **9** | 這是本架構的核心強項：多島、多家族、crossover、fresh、遷移、停滯重置、新穎性。 |
| R5 成本與可控性 | 10% | **6** | 預算閘門依實際單價預估、多種硬上限、三類迴圈計數、去重與快取都齊全。但絕對成本是單線架構的 5–15 倍，算力與 wall-clock 也較高；試驗數同時是統計成本。 |
| R6 實作複雜度與可維護性 | 15% | **5** | 元件數多，1–2 週只能做出縮減版 MVP：2 島、無遷移、固定 operator 比例、簡化 N_eff 聚類、單一 HITL。完整版需 3–4 週。優點是骨架已驗證能跑，而且節點大多是純函式、容易單元測試。 |
| R7 可觀測 / 可重現 / HITL / 安全 | 15% | **8** | 帳本完整記錄血緣與所有 hash；RNG 有種子；LLM 輸出快取可重播；checkpointer 可恢復；兩個無副作用的 interrupt 節點；每變體一個拋棄式沙盒。扣分：平行任務使 trace 較難閱讀，需要血緣樹視覺化工具。 |

**加權總分 ≈ 0.25×7 + 0.15×7 + 0.10×8 + 0.10×9 + 0.10×6 + 0.15×5 + 0.15×8 = 7.05 / 10**

**建議的 MVP 切法（2 週）**：第 1 週做帳本 + worker 子圖（codegen/檢查/沙盒/除錯）+ `select_elites`（2 島、固定規則）+ 預算閘門；第 2 週做 `population_reflect` + 假說帳本 + DSR（N_eff 用固定門檻聚類）+ PBO + 決賽 / OOS / lockbox + `human_review`。遷移、MAP-Elites、影子 fold、自適應 operator 比例、期中 HITL 放 v2。
