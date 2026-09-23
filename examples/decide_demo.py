#!/usr/bin/env python3
"""Single Meta-Jev product demo: data + NL goal → FeatureTable → tree/SOP.

Uses a mocked LLM so it runs offline without META_JEV_LLM_API_KEY.
For a real call, use the CLI instead:

  meta-jev decide --data examples/data/bring_your_csv/loan_approve.csv \\
    --goal "按是否批准贷款做决策树" --out results/loan_decide

Run from repo root:

  PYTHONPATH=src python3 examples/decide_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from meta_jev.data.decide_pipeline import run_decide
from meta_jev.runtime.engine import RuntimeEngine

MOCK_PAYLOAD = {
    "feature_keys": ["income_band", "credit_history", "debt_ratio"],
    "label_key": "approve",
    "label_kind": "categorical",
    "rows": [
        {"income_band": "high", "credit_history": "good", "debt_ratio": "low", "approve": "yes"},
        {"income_band": "high", "credit_history": "good", "debt_ratio": "high", "approve": "yes"},
        {"income_band": "low", "credit_history": "bad", "debt_ratio": "high", "approve": "no"},
        {"income_band": "low", "credit_history": "bad", "debt_ratio": "low", "approve": "no"},
        {"income_band": "mid", "credit_history": "good", "debt_ratio": "mid", "approve": "yes"},
        {"income_band": "mid", "credit_history": "bad", "debt_ratio": "mid", "approve": "no"},
        {"income_band": "high", "credit_history": "bad", "debt_ratio": "low", "approve": "yes"},
        {"income_band": "low", "credit_history": "good", "debt_ratio": "high", "approve": "no"},
    ],
}


def _mock_chat(messages, **kwargs):  # noqa: ANN001, ANN003
    return {
        "choices": [
            {"message": {"content": "```json\n" + json.dumps(MOCK_PAYLOAD) + "\n```"}}
        ]
    }


def main() -> int:
    data = _REPO / "examples/data/bring_your_csv/loan_approve.csv"
    out = _REPO / "examples/artifacts/decide_demo"
    goal = "按是否批准贷款做决策树"
    result = run_decide(
        data,
        goal,
        out_dir=out,
        chat_fn=_mock_chat,
        max_depth=3,
        min_samples=1,
        seed=0,
        trace_examples=1,
        repo_root=_REPO,
    )
    print(
        f"decide_demo ok → {result['out_dir']}  "
        f"rows={result['n_rows']} features={result['n_features']} "
        f"label={result['label_key']}"
    )
    print(f"  tree → {result['tree_out']}")
    print(f"  sop  → {result['sop_out']}")
    print(result["story_zh"])
    if result.get("sop") is not None and result["table"].rows:
        row = result["table"].rows[0]
        obs = {k: row[k] for k in result["table"].feature_keys}
        traced = RuntimeEngine().run_sop_traced(result["sop"], obs)
        print(f"sample trace decision={traced.get('decision')!r}")
    print(
        "\nReal LLM path:\n"
        f'  meta-jev decide --data {data.relative_to(_REPO)} '
        f'--goal "{goal}" --out results/loan_decide'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
