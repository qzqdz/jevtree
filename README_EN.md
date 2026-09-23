# jevtree

Language: [中文](README.md) | English

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tests-pytest-green.svg)](#)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey.svg)](./pyproject.toml)

**One data path + one natural-language goal → an auditable decision tree / DecisionSOP.**

```text
data (CSV / JSON / text / directory)  +  goal (natural language)
                ↓  LLM normalize
          FeatureTable
                ↓  IG grow
     tree.json · sop.json · story.md
                ↓  optional trace
           auditable Q&A path
```

```mermaid
flowchart LR
  A["data + NL goal"] --> B[LLM → FeatureTable]
  B --> C[IG grow]
  C --> D[Tree / SOP / story]
  D --> E[run + trace]
```

---

## Five-minute quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# optional: pip install -r requirements.lock
cp .env.example .env   # set JEVTREE_LLM_API_KEY (never commit .env)

# Single product entry: data + natural-language goal
jevtree decide \
  --data examples/data/bring_your_csv/loan_approve.csv \
  --goal "Grow a loan-approval decision tree" \
  --out results/loan_decide

jevtree decide \
  --data examples/data/batch_texts/support_emails.csv \
  --goal "Classify emails into spam/ham or support categories" \
  --out results/email_decide

jevtree decide \
  --data examples/data/homework_scoring/short_answers.csv \
  --goal "Score short answers 1-5 and grow an auditable grading SOP" \
  --out results/hw_decide \
  --trace-examples 2

jevtree decide \
  --data examples/data/messy_notes_sample.txt \
  --goal "Route to the right team" \
  --out results/messy_decide
```

Aliases also work: `jevtree run-job` / `jevtree from-data`.

The output directory (`--out`, or default `results/decide_<timestamp>/`) contains:

| File | Contents |
|------|----------|
| `feature_table.json` / `.csv` | LLM-normalized feature table |
| `tree.json` | IG decision tree |
| `sop.json` | Runnable DecisionSOP |
| `story.md` | Bilingual narrative |
| `traces.json` | Optional example traces (`--trace-examples N`) |

Run one observation against an existing SOP:

```bash
printf '{"income_band":"high","credit_history":"good","employment_years":"5plus","debt_ratio":"low","collateral":"yes"}' \
  > /tmp/obs.json
jevtree run --sop results/loan_decide/sop.json --input /tmp/obs.json --trace
```

Data sources: [`examples/data/SOURCES.md`](examples/data/SOURCES.md).

---

## Project docs (Feishu Wiki)

- **Wiki**: [jevtree](https://hcnzpmamw7fl.feishu.cn/wiki/PeOfwAaTOik9cDkCOlWcschanEf)
- **Index**: [项目文档索引](https://hcnzpmamw7fl.feishu.cn/wiki/EmQ2wJRoqiQIcRk7nZsc5Eb7ntf)

---

## CLI

```bash
jevtree decide --help     # sole public product entry: --data + --goal
jevtree run --sop … --trace
jevtree validate
jevtree eval-afa --config configs/…
jevtree version
```

Research hard-budget curves:

```bash
jevtree eval-afa --config configs/eval_miniboone_hard.yaml
```

---

## Environment

Requires **Python ≥ 3.11** and `JEVTREE_LLM_*` in `.env` (mimo-compatible base URL / model / key).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.lock
cp .env.example .env   # set JEVTREE_LLM_API_KEY — never commit secrets
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md). More scripts: [`examples/README_EN.md`](examples/README_EN.md).
