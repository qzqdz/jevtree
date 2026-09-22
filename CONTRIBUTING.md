# Contributing to Meta-Jev

Thanks for helping. Keep the product spine clear: **data/materials + goal → FeatureTable → IG tree/SOP → auditable run**.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.lock
# optional: pytest already available via your env / pip install pytest
```

Never commit `.env`, API keys, or anything under `llm_model/` / `ref/`.

## Checks before a PR

```bash
.venv/bin/pytest -q
# optional demos (offline):
.venv/bin/python examples/bring_your_csv.py
.venv/bin/python examples/batch_texts_to_tree.py
```

## Scope notes

- Vertical stories (loan, support, homework, tickets) are **examples/presets**, not separate products.
- Do **not** quote Acc@budget / leaderboard numbers outside `meta-jev eval-afa`.
- Prefer tiny, license-safe demo data under `examples/data/` and document sources in `examples/data/SOURCES.md`.
