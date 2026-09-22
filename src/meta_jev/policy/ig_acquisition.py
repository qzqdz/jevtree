"""IG-based AFA acquisition policy (P1 static ranking).

Aligns with AFABench AFAMethod (act / predict / save / load).
Hard budgets require force_acquisition so act never emits stop=0 while features remain.

predict() delegates to MatchMajorityPredictor (shared with Random/Sequential baselines).
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from meta_jev.core.entropy import best_split, information_gain
from meta_jev.policy.predictor import (
    MatchMajorityPredictor,
    build_predictor,
    configure_policy_predictor,
    prepare_binned_train,
)


class IGAcquisitionPolicy(ABC):
    """AFA policy: rank remaining features by information gain.

    Modes:
      - static: global IG order fitted on train (v0 / P1)
      - sequential: conditional IG given acquired values (v1)

    AFABench action convention: 0 = stop; i >= 1 → selection index i-1 (1-indexed).
    For hard_budget episodes, set force_acquisition=True so act never emits stop.
    """

    force_acquisition: bool = False
    has_builtin_classifier: bool = True

    @abstractmethod
    def fit(self, rows: list[dict[str, Any]], label_key: str, feature_keys: list[str]) -> None:
        """Fit static / conditional IG structures from train rows."""

    @abstractmethod
    def act(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        selection_mask: Sequence[bool] | None = None,
        label: Any = None,
        feature_shape: Any = None,
    ) -> int:
        """Choose next acquisition (AFAAction int)."""

    @abstractmethod
    def predict(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        label: Any = None,
        feature_shape: Any = None,
    ) -> Any:
        """Class distribution / label from currently observed features."""

    def save(self, path: str) -> None:
        raise NotImplementedError("IGAcquisitionPolicy.save — P1")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "IGAcquisitionPolicy":
        raise NotImplementedError("IGAcquisitionPolicy.load — P1")


class StaticIGAcquisitionPolicy(IGAcquisitionPolicy):
    """Global IG ranking fitted once on train; greedy acquire by rank."""

    force_acquisition: bool = False
    has_builtin_classifier: bool = True

    def __init__(
        self,
        *,
        criterion: str = "gain",
        continuous_keys: Sequence[str] | None = None,
        n_bins: int = 4,
        bin_seed: int = 0,
        force_acquisition: bool = False,
        predictor_name: str = "match_majority",
    ) -> None:
        self.criterion = criterion
        self.continuous_keys = list(continuous_keys) if continuous_keys else []
        self.n_bins = n_bins
        self.bin_seed = bin_seed
        self.force_acquisition = force_acquisition
        self.predictor_name = predictor_name
        self.feature_keys_: list[str] = []
        self.label_key_: str | None = None
        self.ranking_: list[str] = []  # feature names, best→worst
        self.ig_scores_: dict[str, float] = {}
        self.train_rows_: list[dict[str, Any]] = []
        self.raw_train_rows_: list[dict[str, Any]] = []
        self.global_majority_: Any = None
        self.bin_edges_: dict[str, list[float]] = {}
        self._predictor = build_predictor(predictor_name)

    def fit(self, rows: list[dict[str, Any]], label_key: str, feature_keys: list[str]) -> None:
        self.feature_keys_ = list(feature_keys)
        self.label_key_ = label_key
        self.raw_train_rows_ = list(rows)

        prepared, bin_edges, cont = prepare_binned_train(
            rows,
            feature_keys,
            continuous_keys=self.continuous_keys or None,
            n_bins=self.n_bins,
            bin_seed=self.bin_seed,
        )
        self.continuous_keys = cont
        self.bin_edges_ = bin_edges
        self.train_rows_ = prepared

        labels = [r[label_key] for r in prepared]

        split = best_split(
            prepared, label_key, feature_keys, criterion=self.criterion  # type: ignore[arg-type]
        )
        ranking_rows = split.get("ranking") or []
        if ranking_rows:
            self.ranking_ = [r["feature"] for r in ranking_rows]
            self.ig_scores_ = {r["feature"]: float(r["score"]) for r in ranking_rows}
        else:
            # fallback per-feature IG
            scores = {
                fk: information_gain(labels, [r[fk] for r in prepared]) for fk in feature_keys
            }
            self.ranking_ = sorted(feature_keys, key=lambda f: (-scores[f], f))
            self.ig_scores_ = scores

        self._predictor = build_predictor(self.predictor_name)
        configure_policy_predictor(
            self._predictor,
            feature_keys=self.feature_keys_,
            label_key=label_key,
            binned_rows=prepared,
            raw_rows=self.raw_train_rows_,
            bin_edges=bin_edges,
            ranking=self.ranking_,
        )
        self.global_majority_ = getattr(self._predictor, "global_majority_", None)

    def _remaining_indices(self, feature_mask: Sequence[bool]) -> list[int]:
        """Feature indices not yet acquired (False in mask = unobserved)."""
        return [i for i, seen in enumerate(feature_mask) if not seen]

    def act(
        self,
        masked_features: Sequence[Any],  # noqa: ARG002
        feature_mask: Sequence[bool],
        selection_mask: Sequence[bool] | None = None,
        label: Any = None,  # noqa: ARG002
        feature_shape: Any = None,  # noqa: ARG002
    ) -> int:
        if not self.ranking_:
            raise RuntimeError("StaticIGAcquisitionPolicy.fit must be called first")

        remaining = self._remaining_indices(feature_mask)
        if selection_mask is not None:
            remaining = [i for i in remaining if selection_mask[i]]

        if not remaining:
            return 0

        # pick highest-ranked remaining feature
        rank_pos = {fk: i for i, fk in enumerate(self.ranking_)}
        best_i = min(
            remaining,
            key=lambda i: (
                rank_pos.get(self.feature_keys_[i], len(self.ranking_)),
                i,
            ),
        )

        if self.force_acquisition:
            # never stop while features remain
            return best_i + 1

        # soft mode: still acquire if anything left (static policy always greedy)
        return best_i + 1

    def predict(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        label: Any = None,
        feature_shape: Any = None,
    ) -> Any:
        if self.label_key_ is None:
            raise RuntimeError("StaticIGAcquisitionPolicy.fit must be called first")
        return self._predictor.predict(
            masked_features, feature_mask, label=label, feature_shape=feature_shape
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            payload = {
                "type": "StaticIGAcquisitionPolicy",
                "criterion": self.criterion,
                "continuous_keys": self.continuous_keys,
                "n_bins": self.n_bins,
                "bin_seed": self.bin_seed,
                "force_acquisition": self.force_acquisition,
                "feature_keys": self.feature_keys_,
                "label_key": self.label_key_,
                "ranking": self.ranking_,
                "ig_scores": self.ig_scores_,
                "global_majority": self.global_majority_,
                "train_rows": self.train_rows_,
                "bin_edges": self.bin_edges_,
            }
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
        else:
            with open(p, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "StaticIGAcquisitionPolicy":  # noqa: ARG002
        p = Path(path)
        if p.suffix.lower() == ".json":
            with open(p, encoding="utf-8") as f:
                payload = json.load(f)
            obj = cls(
                criterion=payload.get("criterion", "gain"),
                continuous_keys=payload.get("continuous_keys"),
                n_bins=int(payload.get("n_bins", 4)),
                bin_seed=int(payload.get("bin_seed", 0)),
                force_acquisition=bool(payload.get("force_acquisition", False)),
            )
            obj.feature_keys_ = list(payload.get("feature_keys", []))
            obj.label_key_ = payload.get("label_key")
            obj.ranking_ = list(payload.get("ranking", []))
            obj.ig_scores_ = {str(k): float(v) for k, v in payload.get("ig_scores", {}).items()}
            obj.global_majority_ = payload.get("global_majority")
            obj.train_rows_ = list(payload.get("train_rows", []))
            raw_edges = payload.get("bin_edges") or {}
            obj.bin_edges_ = {
                str(k): [float(e) for e in v] for k, v in raw_edges.items()
            }
            obj._predictor = MatchMajorityPredictor()
            if obj.label_key_ is not None:
                obj._predictor.configure(
                    feature_keys=obj.feature_keys_,
                    label_key=obj.label_key_,
                    train_rows=obj.train_rows_,
                    bin_edges=obj.bin_edges_,
                    ranking=obj.ranking_,
                    global_majority=obj.global_majority_,
                )
            return obj
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, StaticIGAcquisitionPolicy):
            raise TypeError(f"expected StaticIGAcquisitionPolicy, got {type(obj)}")
        return obj


__all__ = [
    "IGAcquisitionPolicy",
    "StaticIGAcquisitionPolicy",
]
