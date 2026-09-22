"""Non-IG acquisition baselines sharing MatchMajorityPredictor with StaticIG.

RandomAcquisitionPolicy: uniform random among remaining features (seeded).
SequentialAcquisitionPolicy: fixed feature_keys order.

Both use the same MatchMajorityPredictor code path as StaticIG so Acc/F1
vs budget ablations isolate acquisition strategy, not the classifier.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from pathlib import Path
from typing import Any, Sequence

from meta_jev.core.entropy import best_split, information_gain
from meta_jev.policy.ig_acquisition import IGAcquisitionPolicy
from meta_jev.policy.predictor import (
    MatchMajorityPredictor,
    build_predictor,
    configure_policy_predictor,
    prepare_binned_train,
)


def _fit_shared_predictor(
    rows: list[dict[str, Any]],
    label_key: str,
    feature_keys: list[str],
    *,
    continuous_keys: Sequence[str] | None,
    n_bins: int,
    bin_seed: int,
    criterion: str,
    predictor_name: str = "match_majority",
) -> tuple[
    Any,
    list[str],
    dict[str, float],
    list[str],
    list[dict[str, Any]],
    dict[str, list[float]],
]:
    """Bin train + IG ranking; attach match-majority or logistic predictor."""
    prepared, bin_edges, cont = prepare_binned_train(
        rows,
        feature_keys,
        continuous_keys=continuous_keys,
        n_bins=n_bins,
        bin_seed=bin_seed,
    )
    labels = [r[label_key] for r in prepared]
    split = best_split(
        prepared, label_key, feature_keys, criterion=criterion  # type: ignore[arg-type]
    )
    ranking_rows = split.get("ranking") or []
    if ranking_rows:
        ranking = [r["feature"] for r in ranking_rows]
        ig_scores = {r["feature"]: float(r["score"]) for r in ranking_rows}
    else:
        scores = {
            fk: information_gain(labels, [r[fk] for r in prepared]) for fk in feature_keys
        }
        ranking = sorted(feature_keys, key=lambda f: (-scores[f], f))
        ig_scores = scores

    predictor = build_predictor(predictor_name)
    configure_policy_predictor(
        predictor,
        feature_keys=feature_keys,
        label_key=label_key,
        binned_rows=prepared,
        raw_rows=rows,
        bin_edges=bin_edges,
        ranking=ranking,
    )
    return predictor, ranking, ig_scores, cont, prepared, bin_edges


class RandomAcquisitionPolicy(IGAcquisitionPolicy):
    """Uniform random acquisition among remaining features; shared match-majority predict."""

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
        seed: int = 0,
        predictor_name: str = "match_majority",
    ) -> None:
        self.criterion = criterion
        self.continuous_keys = list(continuous_keys) if continuous_keys else []
        self.n_bins = n_bins
        self.bin_seed = bin_seed
        self.force_acquisition = force_acquisition
        self.seed = int(seed)
        self.predictor_name = predictor_name
        self._rng = random.Random(self.seed)
        self.feature_keys_: list[str] = []
        self.label_key_: str | None = None
        self.ranking_: list[str] = []
        self.ig_scores_: dict[str, float] = {}
        self.train_rows_: list[dict[str, Any]] = []
        self.raw_train_rows_: list[dict[str, Any]] = []
        self.global_majority_: Any = None
        self.bin_edges_: dict[str, list[float]] = {}
        self._predictor = build_predictor(predictor_name)

    def fit(self, rows: list[dict[str, Any]], label_key: str, feature_keys: list[str]) -> None:
        self.feature_keys_ = list(feature_keys)
        self.label_key_ = label_key
        self._rng = random.Random(self.seed)
        (
            self._predictor,
            self.ranking_,
            self.ig_scores_,
            cont,
            prepared,
            bin_edges,
        ) = _fit_shared_predictor(
            rows,
            label_key,
            feature_keys,
            continuous_keys=self.continuous_keys or None,
            n_bins=self.n_bins,
            bin_seed=self.bin_seed,
            criterion=self.criterion,
            predictor_name=self.predictor_name,
        )
        self.continuous_keys = cont
        self.train_rows_ = prepared
        self.raw_train_rows_ = list(rows)
        self.bin_edges_ = bin_edges
        self.global_majority_ = getattr(self._predictor, "global_majority_", None)

    def act(
        self,
        masked_features: Sequence[Any],  # noqa: ARG002
        feature_mask: Sequence[bool],
        selection_mask: Sequence[bool] | None = None,
        label: Any = None,  # noqa: ARG002
        feature_shape: Any = None,  # noqa: ARG002
    ) -> int:
        if not self.feature_keys_:
            raise RuntimeError("RandomAcquisitionPolicy.fit must be called first")

        remaining = [i for i, seen in enumerate(feature_mask) if not seen]
        if selection_mask is not None:
            remaining = [i for i in remaining if selection_mask[i]]

        if not remaining:
            return 0

        # Pure function of (seed, mask): schedule-/order-independent determinism.
        # Avoid consuming a shared RNG across budgets/episodes (historical Acc jitter).
        remaining_sorted = sorted(remaining)
        state = (
            self.seed,
            tuple(bool(x) for x in feature_mask),
            tuple(remaining_sorted),
        )
        digest = hashlib.sha256(repr(state).encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        pick = rng.choice(remaining_sorted)
        if self.force_acquisition:
            return pick + 1
        return pick + 1

    def predict(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        label: Any = None,
        feature_shape: Any = None,
    ) -> Any:
        return self._predictor.predict(
            masked_features, feature_mask, label=label, feature_shape=feature_shape
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            payload = {
                "type": "RandomAcquisitionPolicy",
                "criterion": self.criterion,
                "continuous_keys": self.continuous_keys,
                "n_bins": self.n_bins,
                "bin_seed": self.bin_seed,
                "force_acquisition": self.force_acquisition,
                "seed": self.seed,
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
    def load(cls, path: str, device: str = "cpu") -> "RandomAcquisitionPolicy":  # noqa: ARG002
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
                seed=int(payload.get("seed", 0)),
            )
            obj.feature_keys_ = list(payload.get("feature_keys", []))
            obj.label_key_ = payload.get("label_key")
            obj.ranking_ = list(payload.get("ranking", []))
            obj.ig_scores_ = {
                str(k): float(v) for k, v in payload.get("ig_scores", {}).items()
            }
            obj.global_majority_ = payload.get("global_majority")
            obj.train_rows_ = list(payload.get("train_rows", []))
            raw_edges = payload.get("bin_edges") or {}
            obj.bin_edges_ = {
                str(k): [float(e) for e in v] for k, v in raw_edges.items()
            }
            obj._predictor = MatchMajorityPredictor()
            obj._predictor.configure(
                feature_keys=obj.feature_keys_,
                label_key=obj.label_key_ or "y",
                train_rows=obj.train_rows_,
                bin_edges=obj.bin_edges_,
                ranking=obj.ranking_,
                global_majority=obj.global_majority_,
            )
            return obj
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, RandomAcquisitionPolicy):
            raise TypeError(f"expected RandomAcquisitionPolicy, got {type(obj)}")
        return obj


class SequentialAcquisitionPolicy(IGAcquisitionPolicy):
    """Acquire remaining features in fixed feature_keys order; shared predict."""

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
        seed: int = 0,
        predictor_name: str = "match_majority",
    ) -> None:
        self.criterion = criterion
        self.continuous_keys = list(continuous_keys) if continuous_keys else []
        self.n_bins = n_bins
        self.bin_seed = bin_seed
        self.force_acquisition = force_acquisition
        self.seed = int(seed)  # unused for act; kept for config symmetry
        self.predictor_name = predictor_name
        self.feature_keys_: list[str] = []
        self.label_key_: str | None = None
        self.ranking_: list[str] = []
        self.ig_scores_: dict[str, float] = {}
        self.train_rows_: list[dict[str, Any]] = []
        self.raw_train_rows_: list[dict[str, Any]] = []
        self.global_majority_: Any = None
        self.bin_edges_: dict[str, list[float]] = {}
        self._predictor = build_predictor(predictor_name)

    def fit(self, rows: list[dict[str, Any]], label_key: str, feature_keys: list[str]) -> None:
        self.feature_keys_ = list(feature_keys)
        self.label_key_ = label_key
        (
            self._predictor,
            self.ranking_,
            self.ig_scores_,
            cont,
            prepared,
            bin_edges,
        ) = _fit_shared_predictor(
            rows,
            label_key,
            feature_keys,
            continuous_keys=self.continuous_keys or None,
            n_bins=self.n_bins,
            bin_seed=self.bin_seed,
            criterion=self.criterion,
            predictor_name=self.predictor_name,
        )
        self.continuous_keys = cont
        self.train_rows_ = prepared
        self.raw_train_rows_ = list(rows)
        self.bin_edges_ = bin_edges
        self.global_majority_ = getattr(self._predictor, "global_majority_", None)

    def act(
        self,
        masked_features: Sequence[Any],  # noqa: ARG002
        feature_mask: Sequence[bool],
        selection_mask: Sequence[bool] | None = None,
        label: Any = None,  # noqa: ARG002
        feature_shape: Any = None,  # noqa: ARG002
    ) -> int:
        if not self.feature_keys_:
            raise RuntimeError("SequentialAcquisitionPolicy.fit must be called first")

        remaining = [i for i, seen in enumerate(feature_mask) if not seen]
        if selection_mask is not None:
            remaining = [i for i in remaining if selection_mask[i]]

        if not remaining:
            return 0

        # fixed order = feature_keys load order (lowest index first)
        best_i = min(remaining)
        if self.force_acquisition:
            return best_i + 1
        return best_i + 1

    def predict(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        label: Any = None,
        feature_shape: Any = None,
    ) -> Any:
        return self._predictor.predict(
            masked_features, feature_mask, label=label, feature_shape=feature_shape
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            payload = {
                "type": "SequentialAcquisitionPolicy",
                "criterion": self.criterion,
                "continuous_keys": self.continuous_keys,
                "n_bins": self.n_bins,
                "bin_seed": self.bin_seed,
                "force_acquisition": self.force_acquisition,
                "seed": self.seed,
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
    def load(cls, path: str, device: str = "cpu") -> "SequentialAcquisitionPolicy":  # noqa: ARG002
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
                seed=int(payload.get("seed", 0)),
            )
            obj.feature_keys_ = list(payload.get("feature_keys", []))
            obj.label_key_ = payload.get("label_key")
            obj.ranking_ = list(payload.get("ranking", []))
            obj.ig_scores_ = {
                str(k): float(v) for k, v in payload.get("ig_scores", {}).items()
            }
            obj.global_majority_ = payload.get("global_majority")
            obj.train_rows_ = list(payload.get("train_rows", []))
            raw_edges = payload.get("bin_edges") or {}
            obj.bin_edges_ = {
                str(k): [float(e) for e in v] for k, v in raw_edges.items()
            }
            obj._predictor = MatchMajorityPredictor()
            obj._predictor.configure(
                feature_keys=obj.feature_keys_,
                label_key=obj.label_key_ or "y",
                train_rows=obj.train_rows_,
                bin_edges=obj.bin_edges_,
                ranking=obj.ranking_,
                global_majority=obj.global_majority_,
            )
            return obj
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, SequentialAcquisitionPolicy):
            raise TypeError(f"expected SequentialAcquisitionPolicy, got {type(obj)}")
        return obj


__all__ = [
    "RandomAcquisitionPolicy",
    "SequentialAcquisitionPolicy",
]
