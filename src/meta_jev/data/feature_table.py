"""Universal FeatureTable: normalized input for Meta-Jev grow.

All ingest doors (CSV, labeled text batch, scored answers, messy paste)
normalize into FeatureTable, then share one grow → SOP → run/trace path.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

LabelKind = Literal["categorical", "ordinal", "numeric_binned"]


@dataclass
class FeatureTable:
    """Tabular rows ready for IGDecisionTreeGrower.fit."""

    rows: list[dict[str, Any]]
    feature_keys: list[str]
    label_key: str
    label_kind: LabelKind = "categorical"
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("FeatureTable requires at least one row")
        if not self.label_key:
            raise ValueError("label_key is required")
        if self.label_key not in self.rows[0]:
            raise ValueError(
                f"label_key {self.label_key!r} missing from rows "
                f"(columns={list(self.rows[0].keys())})"
            )
        for fk in self.feature_keys:
            if fk == self.label_key:
                raise ValueError(f"feature key collides with label_key: {fk!r}")

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_features(self) -> int:
        return len(self.feature_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "feature_keys": list(self.feature_keys),
            "label_key": self.label_key,
            "label_kind": self.label_kind,
            "source": self.source,
            "meta": dict(self.meta),
        }

    def write_csv(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.feature_keys) + [self.label_key]
        # preserve any id / text columns in meta extras if present on rows
        extras = [
            k
            for k in self.rows[0].keys()
            if k not in fieldnames
        ]
        fieldnames = extras + fieldnames
        with open(dest, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        return dest

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureTable":
        return cls(
            rows=list(d["rows"]),
            feature_keys=list(d["feature_keys"]),
            label_key=str(d["label_key"]),
            label_kind=d.get("label_kind") or "categorical",  # type: ignore[arg-type]
            source=str(d.get("source") or "unknown"),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class IngestResult:
    """Outcome of any ingest door before grow."""

    table: FeatureTable
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table.to_dict(),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


def band_numeric_labels(
    rows: list[dict[str, Any]],
    label_key: str,
    *,
    bands: Sequence[tuple[float, float, str]] | None = None,
    n_bands: int = 3,
) -> tuple[list[dict[str, Any]], LabelKind]:
    """Copy rows with numeric labels mapped to ordinal band strings.

    If *bands* is None, split sorted unique values into up to *n_bands* equal-
    count groups labeled band0..band{k-1}. Non-numeric labels are left as str.
    """
    out = [dict(r) for r in rows]
    nums: list[tuple[int, float]] = []
    for i, r in enumerate(out):
        raw = r.get(label_key)
        try:
            nums.append((i, float(raw)))
        except (TypeError, ValueError):
            out[i][label_key] = str(raw)
    if not nums:
        return out, "categorical"

    if bands:
        for i, v in nums:
            assigned = None
            for lo, hi, name in bands:
                if lo <= v <= hi:
                    assigned = name
                    break
            out[i][label_key] = assigned if assigned is not None else f"raw_{v}"
        return out, "ordinal"

    vals_sorted = sorted(v for _, v in nums)
    # equal-count cuts
    n = len(vals_sorted)
    cuts: list[float] = []
    for b in range(1, n_bands):
        idx = min(n - 1, (b * n) // n_bands)
        cuts.append(vals_sorted[idx])
    for i, v in nums:
        band_id = 0
        for c in cuts:
            if v >= c:
                band_id += 1
            else:
                break
        band_id = min(band_id, n_bands - 1)
        out[i][label_key] = f"band{band_id}"
    return out, "numeric_binned"


def coerce_labels_discrete(
    rows: list[dict[str, Any]],
    label_key: str,
    *,
    label_kind: LabelKind | str = "categorical",
    n_bands: int = 5,
) -> tuple[list[dict[str, Any]], LabelKind]:
    """Ensure labels are discrete strings suitable for IG trees.

    - categorical: stringify
    - ordinal: stringify (keep A/B/C or 1..5 as class tokens)
    - numeric_binned: band floats into band0.. 
    """
    kind: LabelKind
    if label_kind in ("ordinal", "categorical"):
        kind = label_kind  # type: ignore[assignment]
        out = [dict(r) for r in rows]
        for r in out:
            r[label_key] = str(r.get(label_key))
        return out, kind
    if label_kind == "numeric_binned":
        return band_numeric_labels(rows, label_key, n_bands=n_bands)
    # auto: if all numeric and many unique → bin; else stringify
    uniq: set[str] = set()
    all_num = True
    for r in rows:
        raw = r.get(label_key)
        uniq.add(str(raw))
        try:
            float(raw)
        except (TypeError, ValueError):
            all_num = False
    if all_num and len(uniq) > n_bands:
        return band_numeric_labels(rows, label_key, n_bands=n_bands)
    out = [dict(r) for r in rows]
    for r in out:
        r[label_key] = str(r.get(label_key))
    return out, "categorical" if not all_num else "ordinal"


__all__ = [
    "FeatureTable",
    "IngestResult",
    "LabelKind",
    "band_numeric_labels",
    "coerce_labels_discrete",
]
