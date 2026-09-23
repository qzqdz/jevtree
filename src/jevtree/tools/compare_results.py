"""Aggregate existing eval-afa summary.json curves into comparison.json.

Does NOT rescore — only reads Acc/F1 already written by jevtree eval-afa.

Refuses to silently concatenate runs that disagree on dataset_hash,
predictor_name, split_seed, or (when present) train/test sizes. Fails if git
commits differ or any run has git_dirty=true while another claims clean.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ComparisonError(ValueError):
    """Incompatible eval-afa summaries — do not concatenate."""


def _load_summary(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _curve_map(summary: dict[str, Any]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for pt in summary.get("curve") or []:
        b = int(pt["budget"])
        out[b] = {"accuracy": float(pt["accuracy"]), "f1": float(pt["f1"])}
    return out


def _field(summary: dict[str, Any], key: str) -> Any:
    if key in summary and summary[key] is not None:
        return summary[key]
    prov = summary.get("provenance") or {}
    if isinstance(prov, dict) and key in prov and prov[key] is not None:
        return prov[key]
    return summary.get(key)


def validate_summaries_compatible(summaries: list[dict[str, Any]]) -> None:
    """Raise ComparisonError if summaries must not be concatenated."""
    if len(summaries) < 2:
        return

    def vals(key: str) -> list[Any]:
        return [_field(s, key) for s in summaries]

    must_match = ("dataset_hash", "predictor_name", "split_seed")
    for key in must_match:
        vs = vals(key)
        # Allow all-None only if truly absent everywhere; still require equality.
        first = vs[0]
        for i, v in enumerate(vs[1:], start=1):
            if v != first:
                raise ComparisonError(
                    f"refusing to compare: {key} mismatch "
                    f"(run0={first!r}, run{i}={v!r})"
                )

    # Train/test sizes when present on any run
    for key in ("n_train", "n_test"):
        vs = vals(key)
        present = [(i, v) for i, v in enumerate(vs) if v is not None]
        if len(present) >= 2:
            first_i, first_v = present[0]
            for i, v in present[1:]:
                if v != first_v:
                    raise ComparisonError(
                        f"refusing to compare: {key} mismatch "
                        f"(run{first_i}={first_v!r}, run{i}={v!r})"
                    )

    commits = vals("git_commit")
    if any(c is not None for c in commits):
        # Normalize None vs missing as distinct from differing SHAs
        nonzero = [c for c in commits if c is not None]
        if len(set(nonzero)) > 1:
            raise ComparisonError(
                f"refusing to compare: git_commit differs across runs: {commits!r}"
            )
        if any(c is None for c in commits) and nonzero:
            raise ComparisonError(
                f"refusing to compare: some runs missing git_commit: {commits!r}"
            )

    dirties = [_field(s, "git_dirty") for s in summaries]
    # Fail if any dirty=true while another claims clean (False).
    # All-dirty is allowed (same unclean tree); never mix dirty with claimed-clean.
    if any(d is True for d in dirties) and any(d is False for d in dirties):
        raise ComparisonError(
            "refusing to compare: mixed git_dirty flags "
            f"(dirty vs clean): {dirties!r}. "
            "Never treat a clean scaffold commit as rebuilding a dirty tree."
        )


def build_comparison(
    run_dirs: list[Path],
    *,
    labels: list[str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for i, d in enumerate(run_dirs):
        summary_path = d / "summary.json" if d.is_dir() else d
        if summary_path.name != "summary.json":
            summary_path = d / "summary.json"
        summary = _load_summary(summary_path)
        summaries.append(summary)
        label = (labels[i] if labels and i < len(labels) else None) or summary.get(
            "policy_name", f"run_{i}"
        )
        entries.append(
            {
                "label": label,
                "policy_name": summary.get("policy_name"),
                "dataset_id": summary.get("dataset_id"),
                "dataset_hash": _field(summary, "dataset_hash"),
                "split_seed": _field(summary, "split_seed"),
                "git_commit": _field(summary, "git_commit"),
                "git_dirty": _field(summary, "git_dirty"),
                "run_dir": str(summary_path.parent),
                "summary_path": str(summary_path),
                "curve": summary.get("curve"),
                "n_test": summary.get("n_test"),
                "n_train": summary.get("n_train"),
                "predictor_name": _field(summary, "predictor_name"),
            }
        )

    if strict:
        validate_summaries_compatible(summaries)

    budgets: set[int] = set()
    maps: dict[str, dict[int, dict[str, float]]] = {}
    for e in entries:
        m = _curve_map({"curve": e["curve"]})
        maps[e["label"]] = m
        budgets.update(m.keys())

    table_rows: list[dict[str, Any]] = []
    for b in sorted(budgets):
        row: dict[str, Any] = {"budget": b}
        for e in entries:
            lab = e["label"]
            pt = maps[lab].get(b)
            row[f"{lab}_acc"] = None if pt is None else pt["accuracy"]
            row[f"{lab}_f1"] = None if pt is None else pt["f1"]
        table_rows.append(row)

    return {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Aggregated from existing summary.json only; no new scoring.",
        "runs": entries,
        "table": table_rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m jevtree.tools.compare_results",
        description="Concatenate eval-afa summary Acc/F1 curves (non-scoring).",
    )
    p.add_argument(
        "run_dirs",
        nargs="+",
        help="Result run directories (each containing summary.json)",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching run_dirs order (default: policy_name)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output comparison.json path",
    )
    p.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="Dangerous: skip compatibility checks (not recommended).",
    )
    args = p.parse_args(argv)

    run_dirs = [Path(x) for x in args.run_dirs]
    for d in run_dirs:
        sp = d / "summary.json" if d.is_dir() else d
        if not sp.is_file() and not (d / "summary.json").is_file():
            print(f"missing summary.json under {d}", file=sys.stderr)
            return 2

    try:
        comp = build_comparison(
            run_dirs, labels=args.labels, strict=not args.allow_incompatible
        )
    except ComparisonError as exc:
        print(f"compare_results: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(comp, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # print table
    labels = [e["label"] for e in comp["runs"]]
    header = ["Budget"] + [f"{l} Acc" for l in labels] + [f"{l} F1" for l in labels]
    print(" | ".join(header))
    print("-+-".join("-" * len(h) for h in header))
    for row in comp["table"]:
        cells = [str(row["budget"])]
        for l in labels:
            v = row.get(f"{l}_acc")
            cells.append("" if v is None else f"{v:.4f}")
        for l in labels:
            v = row.get(f"{l}_f1")
            cells.append("" if v is None else f"{v:.4f}")
        print(" | ".join(cells))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
