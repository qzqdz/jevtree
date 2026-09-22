"""CSV → FeatureTable (offline main path)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from meta_jev.data.feature_table import (
    FeatureTable,
    IngestResult,
    LabelKind,
    coerce_labels_discrete,
)


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def ingest_csv(
    path: str | Path,
    *,
    label: str,
    features: Sequence[str] | None = None,
    label_kind: LabelKind | str = "categorical",
    n_bands: int = 5,
    exclude: Sequence[str] | None = None,
) -> IngestResult:
    """Load a user CSV into FeatureTable.

    *features*: explicit columns, or None/"auto" → all non-label columns
    except those in *exclude* (default: id, text, notes, raw_text).
    """
    path = Path(path)
    rows = load_csv_rows(path)
    cols = list(rows[0].keys())
    if label not in cols:
        raise ValueError(f"label column {label!r} not in CSV columns {cols}")

    default_exclude = {"id", "text", "notes", "raw_text", "answer", "document"}
    excl = set(exclude) if exclude is not None else default_exclude
    excl.add(label)

    if features is None or (len(features) == 1 and features[0] in ("auto", "*")):
        feature_keys = [c for c in cols if c not in excl]
    else:
        feature_keys = list(features)
        missing = [c for c in feature_keys if c not in cols]
        if missing:
            raise ValueError(f"feature columns missing from CSV: {missing}")

    if not feature_keys:
        raise ValueError(
            "no feature columns left after excluding label/id/text — "
            "pass --feature COL1,COL2 explicitly"
        )

    # stringify feature values for discrete IG (keep as-is strings)
    prepared = [dict(r) for r in rows]
    for r in prepared:
        for fk in feature_keys:
            if fk in r and r[fk] is not None:
                r[fk] = str(r[fk]).strip()

    prepared, kind = coerce_labels_discrete(
        prepared, label, label_kind=label_kind, n_bands=n_bands
    )
    table = FeatureTable(
        rows=prepared,
        feature_keys=feature_keys,
        label_key=label,
        label_kind=kind,
        source="csv",
        meta={"path": str(path), "n_bands": n_bands},
    )
    return IngestResult(
        table=table,
        provenance={"loader": "csv", "path": str(path), "label_kind": kind},
    )


__all__ = ["ingest_csv", "load_csv_rows"]
