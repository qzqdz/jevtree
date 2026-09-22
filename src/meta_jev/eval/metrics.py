"""Acc / F1 vs budget — SOLE metrics module for official scores.

Anti-reward-hacking: no other module or examples/ script may publish
leaderboard Acc/F1. Callers must be the eval-afa path only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence


def accuracy_at_budget(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    budget: int,  # noqa: ARG001 — kept for API symmetry / future filtering
) -> float:
    """Accuracy for episodes that used the given hard budget."""
    n = len(y_true)
    if n == 0 or n != len(y_pred):
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / n


def f1_at_budget(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    budget: int,  # noqa: ARG001
    average: str = "macro",
) -> float:
    """F1 for episodes that used the given hard budget (pure-Python macro F1)."""
    if average != "macro":
        raise ValueError("only average='macro' is supported in P1")
    n = len(y_true)
    if n == 0 or n != len(y_pred):
        return 0.0

    labels = sorted(set(y_true) | set(y_pred), key=lambda x: str(x))
    if not labels:
        return 0.0

    f1s: list[float] = []
    for lab in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp == lab)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lab and yp == lab)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if prec + rec == 0.0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s)


def summarize_curve(
    points: list[dict[str, Any]],
    *,
    dataset_id: str | None = None,
    dataset_hash: str | None = None,
    split_seed: int | None = None,
    budget_schedule: list[int] | None = None,
    git_commit: str | None = None,
    config_path: str | None = None,
    policy_name: str | None = None,
) -> dict[str, Any]:
    """Aggregate Acc/F1 vs budget curve + provenance fields for results/."""
    # allow provenance embedded in points[0] extras or kwargs
    prov_keys = (
        "dataset_id",
        "dataset_hash",
        "split_seed",
        "budget_schedule",
        "git_commit",
        "config_path",
        "policy_name",
    )
    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "split_seed": split_seed,
        "budget_schedule": list(budget_schedule) if budget_schedule is not None else [],
        "git_commit": git_commit,
        "config_path": config_path,
        "policy_name": policy_name,
        "curve": [],
    }
    # merge provenance from first point if kwargs missing
    if points:
        extra = points[0]
        for k in prov_keys:
            if summary[k] in (None, []) and k in extra:
                summary[k] = extra[k]

    for p in points:
        summary["curve"].append(
            {
                "budget": p["budget"],
                "accuracy": p["accuracy"],
                "f1": p["f1"],
                "n_episodes": p.get("n_episodes"),
            }
        )
    return summary


__all__ = [
    "accuracy_at_budget",
    "f1_at_budget",
    "summarize_curve",
]
