"""Information entropy & information gain for decision-tree split selection.

Shannon entropy (bits / log2), IG, split_info, gain_ratio, best_split.
Pure Python 3.11+, stdlib only (math).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal


def class_counts(labels: Sequence[Any]) -> dict[Any, int]:
    """统计各类别出现次数 / Count label frequencies."""
    return dict(Counter(labels))


def entropy(labels: Sequence[Any]) -> float:
    """Shannon 熵（比特，log2）。空集或单类 → 0.0。

    H(S) = -Σ p_i * log2(p_i)
    """
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    if len(counts) <= 1:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def partition(rows: list[dict], feature_key: str) -> dict[Any, list[dict]]:
    """按 feature_key 取值分组 / Partition rows by feature value."""
    groups: dict[Any, list[dict]] = {}
    for row in rows:
        v = row[feature_key]
        groups.setdefault(v, []).append(row)
    return groups


def information_gain(labels: Sequence[Any], feature_values: Sequence[Any]) -> float:
    """信息增益 IG = H(labels) - Σ (|S_v|/|S|) * H(S_v)。

    feature_values 与 labels 等长对齐。空 / 全同值特征 → 0.0。
    """
    n = len(labels)
    if n == 0 or len(feature_values) != n:
        return 0.0
    # all-one-value → residual entropy == parent → IG 0
    if len(set(feature_values)) <= 1:
        return 0.0

    parent_h = entropy(labels)
    # group labels by feature value
    buckets: dict[Any, list[Any]] = {}
    for lab, fv in zip(labels, feature_values):
        buckets.setdefault(fv, []).append(lab)

    residual = 0.0
    for subset in buckets.values():
        residual += (len(subset) / n) * entropy(subset)
    return parent_h - residual


def split_info(feature_values: Sequence[Any]) -> float:
    """分支固有值（intrinsic value），供增益率使用。

    SplitInfo(A) = -Σ (|S_v|/|S|) * log2(|S_v|/|S|)
    空或全同值 → 0.0。
    """
    n = len(feature_values)
    if n == 0:
        return 0.0
    counts = Counter(feature_values)
    if len(counts) <= 1:
        return 0.0
    si = 0.0
    for c in counts.values():
        p = c / n
        si -= p * math.log2(p)
    return si


def gain_ratio(labels: Sequence[Any], feature_values: Sequence[Any]) -> float:
    """增益率 = IG / SplitInfo；SplitInfo==0 → 0.0。"""
    si = split_info(feature_values)
    if si == 0.0:
        return 0.0
    return information_gain(labels, feature_values) / si


def best_split(
    rows: list[dict],
    label_key: str,
    feature_keys: list[str],
    criterion: Literal["gain", "gain_ratio"] = "gain",
) -> dict:
    """对每个特征打分，返回最优分裂。

    Returns:
        {
          "feature": str | None,
          "score": float,
          "ranking": list[{"feature", "score", "entropy_after"}],
          "parent_entropy": float,
        }
    无特征或空表：feature=None, score=0。
    """
    if not rows or not feature_keys:
        parent_h = entropy([r[label_key] for r in rows]) if rows else 0.0
        return {
            "feature": None,
            "score": 0.0,
            "ranking": [],
            "parent_entropy": parent_h,
        }

    labels = [r[label_key] for r in rows]
    parent_h = entropy(labels)
    n = len(rows)

    ranking: list[dict] = []
    for fk in feature_keys:
        fvals = [r[fk] for r in rows]
        if criterion == "gain_ratio":
            score = gain_ratio(labels, fvals)
        else:
            score = information_gain(labels, fvals)

        # weighted residual entropy after split
        buckets: dict[Any, list[Any]] = {}
        for lab, fv in zip(labels, fvals):
            buckets.setdefault(fv, []).append(lab)
        entropy_after = sum((len(s) / n) * entropy(s) for s in buckets.values()) if n else 0.0

        ranking.append(
            {
                "feature": fk,
                "score": score,
                "entropy_after": entropy_after,
            }
        )

    # Stable: higher score first, then feature name for ties (determinism).
    ranking.sort(key=lambda x: (-float(x["score"]), str(x["feature"])))
    best = ranking[0] if ranking else None
    return {
        "feature": best["feature"] if best and best["score"] > 0 else (best["feature"] if best else None),
        "score": best["score"] if best else 0.0,
        "ranking": ranking,
        "parent_entropy": parent_h,
    }


__all__ = [
    "class_counts",
    "entropy",
    "partition",
    "information_gain",
    "split_info",
    "gain_ratio",
    "best_split",
]
