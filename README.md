# jevtree

语言: 中文 | [English](README_EN.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tests-pytest-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

**一份数据 + 一句自然语言需求 → 可审计的决策树 / DecisionSOP。**

```text
data (CSV / JSON / 文本 / 目录)  +  goal (自然语言)
                ↓  LLM normalize
          FeatureTable
                ↓  IG grow
     tree.json · sop.json · story.md
                ↓  optional trace
           可审计问答路径
```

```mermaid
flowchart LR
  A["data + NL goal"] --> B[LLM → FeatureTable]
  B --> C[IG grow]
  C --> D[Tree / SOP / story]
  D --> E[run + trace]
```

---

## 五分钟上手（已验证 smoke）

`jevtree decide` 需要 LLM：从 [`.env.example`](.env.example) 复制 `.env` 并填写 `JEVTREE_LLM_API_KEY`（切勿提交 `.env`）。演示 CSV 已提交在仓库内。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # 填 JEVTREE_LLM_API_KEY

# 推荐：一键 smoke（非 0 退出即失败）
./scripts/smoke_decide.sh

# 或手动跑同一条主路径（loan CSV）
jevtree decide \
  --data examples/data/bring_your_csv/loan_approve.csv \
  --goal "按是否批准贷款做决策树" \
  --out results/smoke_loan
```

`scripts/smoke_decide.sh` 会写入 `results/smoke_loan/`（已 gitignore），并可选对 `examples/artifacts/bring_your_csv/sample_obs.json` 跑一次 `jevtree run --trace`。

对已有 SOP 跑一条观察：

```bash
jevtree run \
  --sop results/smoke_loan/sop.json \
  --input examples/artifacts/bring_your_csv/sample_obs.json \
  --trace
```

别名同样可用：`jevtree run-job` / `jevtree from-data`。

产物目录（`--out` 或默认 `results/decide_<timestamp>/`）包含：

| 文件 | 内容 |
|------|------|
| `feature_table.json` / `.csv` | LLM 归一化后的特征表 |
| `tree.json` | IG 决策树 |
| `sop.json` | 可运行 DecisionSOP |
| `story.md` | 中英双语叙事 |
| `traces.json` | 可选示例 trace（`--trace-examples N`） |

更多演示数据（邮件 / 短答 / 杂乱文本）见 [`examples/data/`](examples/data/) 与 [`examples/data/SOURCES.md`](examples/data/SOURCES.md)。

## CLI

```bash
jevtree decide --help     # 唯一公开产品入口：--data + --goal
jevtree run --sop … --trace
jevtree validate
jevtree eval-afa --config configs/…
jevtree version
```

研究向 hard-budget 曲线：

```bash
jevtree eval-afa --config configs/eval_miniboone_hard.yaml
```

---

## 环境

需要 **Python ≥ 3.11** 与 `.env` 中的 `JEVTREE_LLM_*`（兼容 OpenAI 风格 base URL / model / key；亦接受遗留 `META_JEV_LLM_*`）。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# 可选: pip install -r requirements.lock
cp .env.example .env   # 填 JEVTREE_LLM_API_KEY — 切勿提交密钥
```

更多脚本说明见 [`examples/README.md`](examples/README.md)。
