"""Conditional information-gain acquisition policy.

On each act(), restrict train rows to those matching currently observed
(binned) feature values — with the same progressive match relaxation as
MatchMajorityPredictor — then score remaining features by IG / gain_ratio
on that subset. Falls back to global static IG ranking when the matching
subset is smaller than min_samples.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

from meta_jev.core.entropy import best_split, gain_ratio, information_gain
from meta_jev.core.tree import apply_quantile_edges
from meta_jev.policy.ig_acquisition import IGAcquisitionPolicy
from meta_jev.policy.predictor import (
    build_predictor,
    configure_policy_predictor,
    prepare_binned_train,
)


class ConditionalIGAcquisitionPolicy(IGAcquisitionPolicy):
    """Conditional IG given acquired values; static IG fallback when sparse."""

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
        min_samples: int = 20,
        max_candidates: int = 15,
        predictor_name: str = "match_majority",
        seed: int = 0,  # config symmetry; unused in act
    ) -> None:
        self.criterion = criterion
        self.continuous_keys = list(continuous_keys) if continuous_keys else []
        self.n_bins = n_bins
        self.bin_seed = bin_seed
        self.force_acquisition = force_acquisition
        self.min_samples = int(min_samples)
        self.max_candidates = int(max_candidates)
        self.predictor_name = predictor_name
        self.seed = int(seed)
        self.feature_keys_: list[str] = []
        self.label_key_: str | None = None
        self.ranking_: list[str] = []
        self.ig_scores_: dict[str, float] = {}
        self.train_rows_: list[dict[str, Any]] = []
        self.raw_train_rows_: list[dict[str, Any]] = []
        self.global_majority_: Any = None
        self.bin_edges_: dict[str, list[float]] = {}
        self._predictor = build_predictor(predictor_name)
        self._X_codes: Any = None
        self._y_labels: list[Any] = []
        self._code_maps: list[dict[Any, int]] = []

    def fit(self, rows: list[dict[str, Any]], label_key: str, feature_keys: list[str]) -> None:
        import numpy as np

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
            scores = {
                fk: information_gain(labels, [r[fk] for r in prepared]) for fk in feature_keys
            }
            self.ranking_ = sorted(feature_keys, key=lambda f: (-scores[f], f))
            self.ig_scores_ = scores

        n = len(prepared)
        d = len(feature_keys)
        code_maps: list[dict[Any, int]] = []
        X = np.empty((n, d), dtype=np.int32)
        for j, fk in enumerate(feature_keys):
            vals = [r.get(fk) for r in prepared]
            uniq = sorted(set(vals), key=lambda x: str(x))
            cmap = {v: i for i, v in enumerate(uniq)}
            code_maps.append(cmap)
            X[:, j] = np.asarray([cmap[v] for v in vals], dtype=np.int32)
        self._X_codes = X
        self._code_maps = code_maps
        self._y_labels = labels

        self._predictor = build_predictor(self.predictor_name)
        configure_policy_predictor(
            self._predictor,
            feature_keys=self.feature_keys_,
            label_key=label_key,
            binned_rows=self.train_rows_,
            raw_rows=self.raw_train_rows_,
            bin_edges=self.bin_edges_,
            ranking=self.ranking_,
        )
        self.global_majority_ = getattr(self._predictor, "global_majority_", None)

    def bin_value(self, feature_key: str, value: Any) -> Any:
        if feature_key in self.bin_edges_:
            if value is None:
                return None
            try:
                return apply_quantile_edges(float(value), self.bin_edges_[feature_key])
            except (TypeError, ValueError):
                return value
        return value

    def _obs_code(self, feat_idx: int, raw_value: Any) -> int | None:
        fk = self.feature_keys_[feat_idx]
        binned = self.bin_value(fk, raw_value)
        return self._code_maps[feat_idx].get(binned)

    def _match_indices(self, masked_features: Sequence[Any], acquired: list[int]) -> list[int]:
        import numpy as np

        if self._X_codes is None or not acquired:
            return list(range(len(self.train_rows_)))

        rank_pos = {fk: j for j, fk in enumerate(self.ranking_)}
        ordered = sorted(
            acquired,
            key=lambda i: (rank_pos.get(self.feature_keys_[i], len(self.ranking_)), i),
        )
        obs_codes: dict[int, int] = {}
        for i in ordered:
            c = self._obs_code(i, masked_features[i])
            if c is None:
                continue
            obs_codes[i] = c

        if not obs_codes:
            return list(range(len(self.train_rows_)))

        usable = [i for i in ordered if i in obs_codes]
        X = self._X_codes
        n = X.shape[0]
        best: list[int] = []
        for keep in range(len(usable), 0, -1):
            use = usable[:keep]
            mask = np.ones(n, dtype=bool)
            for i in use:
                mask &= X[:, i] == obs_codes[i]
            idx = np.flatnonzero(mask).tolist()
            if len(idx) >= self.min_samples:
                return idx
            if idx:
                best = idx
        return best

    def _score_feature(self, feat_idx: int, subset_idx: Sequence[int]) -> float:
        import numpy as np

        if not subset_idx:
            return 0.0
        X = self._X_codes
        y = [self._y_labels[i] for i in subset_idx]
        fv = X[np.asarray(subset_idx, dtype=np.int64), feat_idx].tolist()
        if self.criterion == "gain_ratio":
            return float(gain_ratio(y, fv))
        return float(information_gain(y, fv))

    def _pick_static(self, remaining: list[int]) -> int:
        rank_pos = {fk: i for i, fk in enumerate(self.ranking_)}
        return min(
            remaining,
            key=lambda i: (
                rank_pos.get(self.feature_keys_[i], len(self.ranking_)),
                i,
            ),
        )

    def act(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        selection_mask: Sequence[bool] | None = None,
        label: Any = None,  # noqa: ARG002
        feature_shape: Any = None,  # noqa: ARG002
    ) -> int:
        if not self.feature_keys_ or self._X_codes is None:
            raise RuntimeError("ConditionalIGAcquisitionPolicy.fit must be called first")

        remaining = [i for i, seen in enumerate(feature_mask) if not seen]
        if selection_mask is not None:
            remaining = [i for i in remaining if selection_mask[i]]
        if not remaining:
            return 0

        acquired = [i for i, seen in enumerate(feature_mask) if seen]
        if not acquired:
            return self._pick_static(remaining) + 1

        subset = self._match_indices(masked_features, acquired)
        if len(subset) < self.min_samples:
            return self._pick_static(remaining) + 1

        candidates = list(remaining)
        if len(candidates) > self.max_candidates:
            rank_pos = {fk: j for j, fk in enumerate(self.ranking_)}
            candidates = sorted(
                candidates,
                key=lambda i: (
                    rank_pos.get(self.feature_keys_[i], len(self.ranking_)),
                    i,
                ),
            )[: self.max_candidates]

        best_i = candidates[0]
        best_score = float("-inf")
        for i in candidates:
            sc = self._score_feature(i, subset)
            if sc > best_score or (sc == best_score and i < best_i):
                best_score = sc
                best_i = i
        return best_i + 1

    def predict(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        label: Any = None,
        feature_shape: Any = None,
    ) -> Any:
        if self.label_key_ is None:
            raise RuntimeError("ConditionalIGAcquisitionPolicy.fit must be called first")
        return self._predictor.predict(
            masked_features, feature_mask, label=label, feature_shape=feature_shape
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ConditionalIGAcquisitionPolicy":  # noqa: ARG002
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ConditionalIGAcquisitionPolicy):
            raise TypeError(f"expected ConditionalIGAcquisitionPolicy, got {type(obj)}")
        return obj


__all__ = ["ConditionalIGAcquisitionPolicy"]
