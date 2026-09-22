#!/usr/bin/env python3
"""Batch labeled texts OR scored homework/exam answers → keyword cues → tree.

Same universal spine for classification and ordinal scoring. Offline (no LLM).
Interpretable cues — not a SOTA text classifier or black-box auto-grader.
Try support_emails.csv, sms_spam_excerpt.csv, or homework_scoring/short_answers.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from meta_jev.data.grow_pipeline import grow_from_table
from meta_jev.data.text_batch import ingest_text_batch
from meta_jev.runtime.engine import RuntimeEngine


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch texts+labels/scores → Meta-Jev tree")
    p.add_argument(
        "--csv",
        default=str(_REPO / "examples/data/batch_texts/support_emails.csv"),
    )
    p.add_argument("--label", default="label", help="label or score column")
    p.add_argument(
        "--label-kind",
        default="categorical",
        choices=["categorical", "ordinal", "numeric_binned"],
    )
    p.add_argument("--text-key", default="text")
    p.add_argument("--goal", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default=str(_REPO / "examples/artifacts/batch_texts"),
    )
    args = p.parse_args(argv)

    label = args.label
    label_kind = args.label_kind
    if "homework" in args.csv or "scoring" in args.csv:
        if label == "label":
            label = "score"
        if args.label_kind == "categorical":
            label_kind = "ordinal"

    ingested = ingest_text_batch(
        csv_path=args.csv,
        text_key=args.text_key,
        label_key=label,
        label_kind=label_kind,
        goal=args.goal or None,
        max_features=12,
    )
    table = ingested.table
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_csv = out_dir / "features.csv"
    table.write_csv(feat_csv)
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
    print("我有一批要分类/打分的文本 + 标签，Meta-Jev 长出可审计决策树：")
    print(result["story_zh"])
    print(result["story_en"])
    print(f"features CSV → {feat_csv}")
    print(f"vocab → {ingested.provenance.get('vocab')}")

    row = {k: table.rows[0][k] for k in table.feature_keys}
    traced = RuntimeEngine().run_sop_traced(result["sop"], row)
    print("trace row0:")
    for i, step in enumerate(traced["path"], 1):
        print(f"  Q{i}: {step['feature']} = {step['value']}")
    print(f"  Decision: {traced['decision']}")
    print()
    print("CLI:")
    print(
        f"  meta-jev ingest-batch --csv {args.csv} --label {label} "
        f"--label-kind {label_kind} --out-csv {feat_csv} "
        f"--out {tree_path} --sop-out {sop_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
