# Agent team 原始產出

這個資料夾收錄 [`../quant-strategy-architecture.md`](../quant-strategy-architecture.md) 所依據的 agent team 原始檔案，未經修改，只把檔案路徑換成 repo 內的相對位置。

| 檔案 | 作者角色 | 內容 |
|---|---|---|
| `requirements.md` | 協調者 | 共同需求、硬性約束、評分標準 R1–R7 |
| `candidate_A.md` | 架構師 A | 強化版 Reflexion 單線迴圈（最後選為主幹） |
| `candidate_B.md` + `candidate_B_skeleton.py` | 架構師 B | 階層式巢狀子圖 |
| `candidate_C.md` + `candidate_C_skeleton.py` | 架構師 C | 族群演化搜尋（Send 扇出） |
| `candidate_D.md` + `candidate_D_skeleton.py` | 架構師 D | Supervisor 多代理 + 多批判者 |
| `judge_quant.md` | 評審 1：量化研究主管 | 檢定力分析、方法論錯誤、試驗預算建議 |
| `judge_eng.md` | 評審 2：工程主管 | 骨架實跑、LangGraph 行為查證、MVP 模組切分 |

骨架程式是設計驗證用的 stub，不是正式實作。正式實作在 [`../../quant-agent/`](../../quant-agent/)。
