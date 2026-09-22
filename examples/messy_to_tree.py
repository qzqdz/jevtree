#!/usr/bin/env python3
"""Messy notes + goal → LLM FeatureTable → same grow path.

Requires META_JEV_LLM_* in gitignored .env. On failure: clear error, no fake rows.
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
from meta_jev.data.messy_ingest import ingest_messy_text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Messy paste → Meta-Jev tree (LLM)")
    p.add_argument(
        "--input",
        default=str(_REPO / "examples/data/messy_notes_sample.txt"),
    )
    p.add_argument("--goal", default="分流到哪个团队")
    p.add_argument(
        "--out-dir",
        default=str(_REPO / "examples/artifacts/messy"),
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    notes = Path(args.input).read_text(encoding="utf-8")
    try:
        ingested = ingest_messy_text(notes, args.goal)
    except (RuntimeError, ValueError) as exc:
        print(f"messy_to_tree failed: {exc}", file=sys.stderr)
        print(
            "Fall back to offline paths:\n"
            "  python examples/bring_your_csv.py\n"
            "  python examples/batch_texts_to_tree.py",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "extracted.csv"
    ingested.table.write_csv(csv_path)
    result = grow_from_table(
        ingested.table,
        out=out_dir / "tree.json",
        sop_out=out_dir / "tree.sop.json",
        min_samples=1,
        max_depth=4,
        seed=args.seed,
    )
    print(f"extracted → {csv_path}")
    print(result["story_zh"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
