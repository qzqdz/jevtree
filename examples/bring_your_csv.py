#!/usr/bin/env python3
"""Generic classmate path: point at a CSV → grow tree → SOP → trace one row.

Product spine demo (not Acc@budget). Default sample: loan approve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from meta_jev.data.csv_ingest import ingest_csv
from meta_jev.data.grow_pipeline import grow_from_table
from meta_jev.runtime.engine import RuntimeEngine


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bring your CSV → Meta-Jev decision tree")
    p.add_argument(
        "--csv",
        default=str(_REPO / "examples/data/bring_your_csv/loan_approve.csv"),
    )
    p.add_argument("--label", default="approve")
    p.add_argument("--feature", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default=str(_REPO / "examples/artifacts/bring_your_csv"),
    )
    args = p.parse_args(argv)

    features = None if args.feature in ("auto", "*") else [
        c.strip() for c in args.feature.split(",") if c.strip()
    ]
    ingested = ingest_csv(args.csv, label=args.label, features=features)
    table = ingested.table
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_path = out_dir / "tree.json"
    sop_path = out_dir / "tree.sop.json"
    result = grow_from_table(
        table,
        out=tree_path,
        sop_out=sop_path,
        min_samples=2,
        max_depth=4,
        seed=args.seed,
    )
    print(result["story_zh"])
    print(result["story_en"])
    print(f"tree → {tree_path}")
    print(f"sop  → {sop_path}")

    row = {k: table.rows[0][k] for k in table.feature_keys}
    engine = RuntimeEngine()
    traced = engine.run_sop_traced(result["sop"], row)
    print("trace one row:")
    for i, step in enumerate(traced["path"], 1):
        print(f"  Q{i}: {step['feature']} = {step['value']}")
    print(f"  Decision: {traced['decision']}")
    obs_path = out_dir / "sample_obs.json"
    obs_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print()
    print("CLI equivalent:")
    print(
        f"  meta-jev grow --csv {args.csv} --label {args.label} "
        f"--out {tree_path} --sop-out {sop_path} --seed {args.seed}"
    )
    print(f"  meta-jev run --sop {sop_path} --input {obs_path} --trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
