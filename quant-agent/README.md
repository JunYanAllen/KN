# quantloop — LangGraph 自我反思 + 自我除錯的策略研究迴圈（參考實作）

架構說明與設計取捨請看 [`../docs/quant-strategy-architecture.md`](../docs/quant-strategy-architecture.md)。

```bash
cd quant-agent
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                                   # 38 個測試
.venv/bin/python -m quantloop.run --run-dir runs/demo --approve # 離線示範（stub LLM + 合成資料）
```

`--approve` 只在示範時自動回覆人工審閱。省略它的話，會在終端機裡互動詢問。

## 模組

| 檔案 | 角色 | 是否由 LLM 撰寫 |
|---|---|---|
| `config.py` | 所有上限與閘門門檻（永遠不進 prompt） | — |
| `data.py` | 資料快照、hash、research / shadow / oos / lockbox 切分；合成資料 | — |
| `engine.py` | 固定回測引擎（次根開盤成交、手續費、滑價、資金費率）、walk-forward、參數高原 | 否 |
| `stats.py` | PSR / DSR / 門檻曲線、配對 block bootstrap、確定性種子 | 否 |
| `safety/ast_rules.py` | 靜態前置過濾（import、I/O、前視寫法、日期字面值、介面） | 否 |
| `safety/sandbox.py` | 子進程 / Docker 沙盒、截斷不變性測試、輸出 schema 驗證 | 否 |
| `ledger.py` | SQLite 試驗帳本、預登記、決賽者凍結、OOS / lockbox 配額 | 否 |
| `llm.py` | Pydantic schemas、呼叫前預算估算、refusal / schema / transient 處理、LangChain 轉接器 | 否 |
| `context.py` | 各角色的 LLM 可見欄位白名單（唯一的資訊閘門） | 否 |
| `nodes.py` / `graph.py` | LangGraph 節點與純函式 routers | 否 |
| `stub.py` | 離線 stub 後端（會故意犯三種錯，用來驗證除錯迴圈） | — |
| 策略模組（執行期產生） | `PARAM_SPACE` + `generate_signals(df, params)` | **是（唯一）** |

## 接真實模型

```python
from langchain_anthropic import ChatAnthropic   # 或其他任何 LangChain chat model
from quantloop.config import Config
from quantloop.llm import LLMClient, LangChainBackend
from quantloop.prompts import VERSION
from quantloop.run import make_deps, checkpointer_for
from quantloop.graph import build_graph
from quantloop.safety.sandbox import DockerSandbox

cfg = Config()   # 先把 cfg.pricing 換成實際單價
backend = LangChainBackend({"strong": ChatAnthropic(model="..."), "cheap": ChatAnthropic(model="...")})
deps = make_deps("runs/real", cfg, snapshot=my_snapshot,   # data.make_snapshot(真實 OHLCV + funding)
                 llm=LLMClient(backend, cfg.pricing.table, VERSION),
                 sandbox=DockerSandbox(image="your-registry/backtest-sandbox@sha256:..."))
app = build_graph(deps, checkpointer_for(deps.run_dir))
```

上線研究前還要做三件事：
- 用 `DockerSandbox`（映像檔只含 python、numpy、pandas，並以 digest 鎖定版本）。傳入真實 LLM 卻沒指定沙盒時，`make_deps` 會直接拒絕，因為子進程版本不隔離網路。
- 資料用真實的交易所 OHLCV 和歷史資金費率，並且處理缺漏的 K 線、統一使用 UTC 的 open time。
- 設定每筆交易的實際成本。

## 產出

每次 run 會在 `runs/<dir>/` 底下產生以下檔案：

| 檔案 | 內容 |
|---|---|
| `events.jsonl` | 每次 LLM 呼叫、每個節點、每次路由及其原因 |
| `ledger.sqlite` | 跨 run 的試驗與配額紀錄 |
| `checkpoints.sqlite` | LangGraph checkpoint，可以從中斷處恢復 |
| `artifacts/` | 程式碼、訊號、PnL，以內容定址方式儲存 |
| `report_<run_id>.json` | 本次 run 的報告，包含 `reflection_lift`：反思組相對於無引導對照組的改善率差 |
