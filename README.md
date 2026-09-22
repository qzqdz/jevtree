# Meta-Jev

**Grow IG-driven decision trees / Jev SOPs and evaluate them under AFABench hard-budget protocols.**

> Research / AFABench iteration skeleton（非最终产品目录）。阶段性设计文档不进本仓库。

## 项目文档（飞书 Wiki）

统一文档挂在知识库节点下：

- **Wiki 节点**：[Meta-Jev](https://hcnzpmamw7fl.feishu.cn/wiki/PeOfwAaTOik9cDkCOlWcschanEf)
- **文档索引**：[项目文档索引](https://hcnzpmamw7fl.feishu.cn/wiki/EmQ2wJRoqiQIcRk7nZsc5Eb7ntf)
- 架构约定：https://hcnzpmamw7fl.feishu.cn/wiki/LyY7wfbYBiXgB8k3yoxcZuunnb5
- 开发路线图：https://hcnzpmamw7fl.feishu.cn/wiki/N1jFwDo6OiQxFqkT4jZctYdin6b

## Official commands

```bash
pip install -e .
# or: PYTHONPATH=src python3 -m meta_jev --help
meta-jev eval-afa --config configs/eval_miniboone_hard.yaml
```

`eval-afa` is the **only** path to Acc/F1-vs-budget. Examples must not claim leaderboard numbers.

```bash
meta-jev grow --help
meta-jev validate --help
meta-jev run --help
meta-jev version
```


## Environment (minimal)

Python ≥3.11. Recreate the eval runtime with pinned deps:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.lock   # or: uv sync  (if uv.lock present)
```

`numpy` is required for `logistic_impute` / discriminative policies. Docs stay on Feishu; this note is only for reproducible eval envs.

## Demos (not scoring)

- `examples/entropy_split_demo.py` — Play-Tennis IG / gain-ratio sanity
- `examples/ticket_routing.py` — secondary SOP construction example
