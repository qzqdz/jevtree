"""Decision tree grower: ID3-style IGDecisionTreeGrower (P1)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence


def quantile_bin_continuous(
    values: Sequence[float],
    *,
    n_bins: int = 4,
    seed: int = 0,
) -> list[str]:
    """Map continuous values to quantile-bin labels (shared with policy).

    Deterministic given *seed* (used to break ties stably via sort key).
    Returns labels like "q0", "q1", ... "q{n_bins-1}".
    Empty / constant series → all "q0".
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(
        enumerate(float(v) for v in values),
        key=lambda iv: (iv[1], (iv[0] * 1103515245 + seed) & 0x7FFFFFFF),
    )
    uniq = {v for _, v in indexed}
    if len(uniq) <= 1:
        return ["q0"] * n
    out = [""] * n
    for rank, (orig_i, _) in enumerate(indexed):
        bin_id = min(n_bins - 1, (rank * n_bins) // n)
        out[orig_i] = f"q{bin_id}"
    return out




def compute_quantile_edges(
    values: Sequence[float],
    *,
    n_bins: int = 4,
) -> list[float]:
    """Return ``n_bins - 1`` ascending interior edges for train→test binning.

    Empty / constant series → no edges (everything maps to ``q0``).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0 or n_bins == 1:
        return []
    uniq = set(vals)
    if len(uniq) <= 1:
        return []
    edges: list[float] = []
    for i in range(1, n_bins):
        # same rank cut as quantile_bin_continuous: (rank * n_bins) // n
        idx = (i * n) // n_bins
        idx = min(max(idx, 0), n - 1)
        edges.append(vals[idx])
    # ensure non-decreasing
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    return edges


def apply_quantile_edges(value: float, edges: Sequence[float]) -> str:
    """Map a scalar to ``q0``..``q{len(edges)}`` given interior edges."""
    if not edges:
        return "q0"
    v = float(value)
    for i, e in enumerate(edges):
        if v < e:
            return f"q{i}"
    return f"q{len(edges)}"


def bin_rows_with_edges(
    rows: list[dict[str, Any]],
    edges_by_key: dict[str, Sequence[float]],
) -> list[dict[str, Any]]:
    """Shallow-copy rows, replacing keys present in *edges_by_key* with bin labels."""
    out = [dict(r) for r in rows]
    for fk, edges in edges_by_key.items():
        for i, r in enumerate(rows):
            if fk in r and r[fk] is not None:
                try:
                    out[i][fk] = apply_quantile_edges(float(r[fk]), edges)
                except (TypeError, ValueError):
                    out[i][fk] = r[fk]
    return out

def bin_rows_continuous(
    rows: list[dict[str, Any]],
    feature_keys: list[str],
    *,
    continuous_keys: Sequence[str] | None = None,
    n_bins: int = 4,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return shallow-copied rows with continuous features replaced by quantile bins."""
    if not continuous_keys:
        return [dict(r) for r in rows]
    out = [dict(r) for r in rows]
    for fk in continuous_keys:
        if fk not in feature_keys:
            continue
        binned = quantile_bin_continuous(
            [float(r[fk]) for r in rows], n_bins=n_bins, seed=seed
        )
        for i, lab in enumerate(binned):
            out[i][fk] = lab
    return out


@dataclass
class TreeLeaf:
    """Terminal node: predicted majority label + support counts."""

    prediction: Any
    counts: dict[str, int] = field(default_factory=dict)
    n_samples: int = 0

    def predict(self, row: dict[str, Any]) -> Any:  # noqa: ARG002
        return self.prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "leaf",
            "prediction": self.prediction,
            "counts": self.counts,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TreeLeaf":
        return cls(
            prediction=d["prediction"],
            counts={str(k): int(v) for k, v in d.get("counts", {}).items()},
            n_samples=int(d.get("n_samples", 0)),
        )


@dataclass
class TreeNode:
    """Internal split on a discrete feature value."""

    feature: str
    children: dict[Any, "TreeNode | TreeLeaf"] = field(default_factory=dict)
    default: Any = None
    n_samples: int = 0

    def predict(self, row: dict[str, Any]) -> Any:
        v = row.get(self.feature)
        child = self.children.get(v)
        if child is None:
            return self.default
        return child.predict(row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "node",
            "feature": self.feature,
            "default": self.default,
            "n_samples": self.n_samples,
            "children": {
                json.dumps(k, default=str): c.to_dict() for k, c in self.children.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TreeNode":
        children: dict[Any, TreeNode | TreeLeaf] = {}
        for k_json, cd in d.get("children", {}).items():
            key = json.loads(k_json)
            if cd.get("kind") == "leaf":
                children[key] = TreeLeaf.from_dict(cd)
            else:
                children[key] = TreeNode.from_dict(cd)
        return cls(
            feature=d["feature"],
            children=children,
            default=d.get("default"),
            n_samples=int(d.get("n_samples", 0)),
        )


def _majority(labels: list[Any]) -> Any:
    if not labels:
        return None
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def _counts_str(labels: list[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(labels).items()}


class DecisionTreeGrower(ABC):
    """Grow a decision tree from tabular rows using information gain / gain ratio.

    Implementations should call meta_jev.core.entropy.best_split (or equivalents)
    and must NOT compute leaderboard metrics — scoring belongs in eval/.
    """

    @abstractmethod
    def fit(
        self,
        rows: list[dict[str, Any]],
        label_key: str,
        feature_keys: list[str],
        *,
        criterion: str = "gain",
        max_depth: int | None = None,
    ) -> Any:
        """Fit and return an opaque tree object."""

    @abstractmethod
    def export_sop(self, tree: Any) -> Any:
        """Optional: export grown tree to DecisionSOP (P3 parity)."""


class IGDecisionTreeGrower(DecisionTreeGrower):
    """ID3-style discrete-feature grower driven by meta_jev.core.entropy.best_split."""

    def __init__(
        self,
        *,
        min_samples: int = 1,
        continuous_keys: Sequence[str] | None = None,
        n_bins: int = 4,
        bin_seed: int = 0,
    ) -> None:
        self.min_samples = min_samples
        self.continuous_keys = list(continuous_keys) if continuous_keys else []
        self.n_bins = n_bins
        self.bin_seed = bin_seed
        self.tree_: TreeNode | TreeLeaf | None = None
        self.label_key_: str | None = None
        self.feature_keys_: list[str] = []

    def fit(
        self,
        rows: list[dict[str, Any]],
        label_key: str,
        feature_keys: list[str],
        *,
        criterion: str = "gain",
        max_depth: int | None = None,
    ) -> TreeNode | TreeLeaf:
        from meta_jev.core.entropy import best_split, partition

        prepared = bin_rows_continuous(
            rows,
            feature_keys,
            continuous_keys=self.continuous_keys,
            n_bins=self.n_bins,
            seed=self.bin_seed,
        )
        self.label_key_ = label_key
        self.feature_keys_ = list(feature_keys)

        def grow(
            subset: list[dict[str, Any]],
            remaining: list[str],
            depth: int,
        ) -> TreeNode | TreeLeaf:
            labels = [r[label_key] for r in subset]
            maj = _majority(labels)
            leaf = TreeLeaf(
                prediction=maj,
                counts=_counts_str(labels),
                n_samples=len(subset),
            )
            if not subset:
                return TreeLeaf(prediction=None, counts={}, n_samples=0)
            if len(set(labels)) <= 1:
                return leaf
            if not remaining:
                return leaf
            if max_depth is not None and depth >= max_depth:
                return leaf
            if len(subset) < self.min_samples:
                return leaf

            split = best_split(
                subset, label_key, remaining, criterion=criterion  # type: ignore[arg-type]
            )
            feat = split["feature"]
            if feat is None or split["score"] <= 0:
                return leaf

            groups = partition(subset, feat)
            node = TreeNode(feature=feat, default=maj, n_samples=len(subset))
            next_remaining = [f for f in remaining if f != feat]
            for val, group in groups.items():
                node.children[val] = grow(group, next_remaining, depth + 1)
            return node

        self.tree_ = grow(prepared, list(feature_keys), 0)
        return self.tree_

    def predict(self, row: dict[str, Any]) -> Any:
        if self.tree_ is None:
            raise RuntimeError("IGDecisionTreeGrower.fit must be called first")
        prepared = bin_rows_continuous(
            [row],
            self.feature_keys_,
            continuous_keys=self.continuous_keys,
            n_bins=self.n_bins,
            seed=self.bin_seed,
        )[0]
        return self.tree_.predict(prepared)

    def export_sop(self, tree: Any) -> Any:
        """Export the fitted tree as a deterministic, locally runnable SOP."""
        if tree is None:
            raise ValueError("tree is required")
        if tree is not self.tree_ and not isinstance(tree, (TreeNode, TreeLeaf)):
            raise TypeError("tree must be a TreeNode or TreeLeaf")
        from meta_jev.core.sop import DecisionSOP

        payload = {
            "kind": "IGDecisionTree",
            "label_key": self.label_key_,
            "feature_keys": list(self.feature_keys_),
            "min_samples": self.min_samples,
            "continuous_keys": list(self.continuous_keys),
            "n_bins": self.n_bins,
            "bin_seed": self.bin_seed,
            "tree": tree.to_dict(),
        }
        return DecisionSOP(
            name="meta-jev-tree",
            description="Deterministic IG decision tree exported as a local SOP",
            nodes=[],
            output="tree",
            tree=payload,
        )

    def to_json(self) -> dict[str, Any]:
        if self.tree_ is None:
            raise RuntimeError("no tree fitted")
        return {
            "type": "IGDecisionTree",
            "label_key": self.label_key_,
            "feature_keys": self.feature_keys_,
            "min_samples": self.min_samples,
            "continuous_keys": self.continuous_keys,
            "n_bins": self.n_bins,
            "bin_seed": self.bin_seed,
            "tree": self.tree_.to_dict(),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)
            f.write("\n")


    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "IGDecisionTreeGrower":
        """Restore a fitted grower from :meth:`to_json` output."""
        if payload.get("type") != "IGDecisionTree":
            raise ValueError("expected an IGDecisionTree payload")
        obj = cls(
            min_samples=int(payload.get("min_samples", 1)),
            continuous_keys=payload.get("continuous_keys") or [],
            n_bins=int(payload.get("n_bins", 4)),
            bin_seed=int(payload.get("bin_seed", 0)),
        )
        obj.label_key_ = payload.get("label_key")
        obj.feature_keys_ = list(payload.get("feature_keys") or [])
        tree = payload.get("tree")
        if not isinstance(tree, dict):
            raise ValueError("missing tree payload")
        obj.tree_ = TreeLeaf.from_dict(tree) if tree.get("kind") == "leaf" else TreeNode.from_dict(tree)
        return obj

    @classmethod
    def load(cls, path: str) -> "IGDecisionTreeGrower":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))


__all__ = [
    "DecisionTreeGrower",
    "IGDecisionTreeGrower",
    "TreeLeaf",
    "TreeNode",
    "apply_quantile_edges",
    "bin_rows_continuous",
    "bin_rows_with_edges",
    "compute_quantile_edges",
    "quantile_bin_continuous",
]
