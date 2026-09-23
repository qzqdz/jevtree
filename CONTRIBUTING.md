# Contributing to Meta-Jev

Thanks for helping. Keep the product spine clear: **data + NL goal → FeatureTable → IG tree/SOP → auditable run**.

Public CLI surface: **`decide`** (aliases `run-job` / `from-data`), **`validate`**, **`run`**, **`eval-afa`**, **`version`**.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.lock
```

Never commit `.env`, API keys, or anything under `llm_model/` / `ref/`.

## Checks before a PR

```bash
.venv/bin/pytest -q
# optional offline decide demo:
.venv/bin/python examples/decide_demo.py
```

## Scope notes

- Vertical stories (loan, support, homework, tickets) are **sample data for `decide`**, not separate products or CLI doors.
- Do **not** quote Acc@budget / leaderboard numbers outside `meta-jev eval-afa`.
- Prefer tiny, license-safe demo data under `examples/data/` and document sources in `examples/data/SOURCES.md`.
