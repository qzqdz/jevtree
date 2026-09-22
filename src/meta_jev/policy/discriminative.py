"""Discriminative / predictor-aligned acquisition (ELLG for logistic).

Selects the next feature by Expected Log-Likelihood Gain (ELLG) under the
shared MaskedLogisticPredictor — not by label entropy on matched subsets
(ConditionalIG). Practical GDFS/DIME-style proxy that is CPU-friendly for
MiniBooNE-scale (≈2000×50, max_candidates=15).

Algorithm (per act):
  1. Impute current observations under the fitted logistic (mean-fill missing).
  2. Soft-match train rows on acquired (binned) features; if |match|<min_samples
     fall back to the full train set (unconditional p(x_j)).
  3. Restrict candidates to top-`max_candidates` by static IG ranking.
  4. For each candidate j, estimate p(x_j | x_S) via quantile bins of raw j on
     the matched train subset; for each bin representative v, form the imputed
     vector with j:=v and score the logistic predictive distribution.
  5. ELLG(j) = H[p(y|x_S)] - E_v[ H[p(y|x_S, x_j=v)] ]  (expected entropy drop).
     Ties broken by static IG rank then feature index.
  6. Acquire argmax ELLG.

Score uses binary predictive entropy of MaskedLogisticPredictor.predict_proba
(no true labels required at acquisition time). Multiclass uses entropy of the
one-vs-rest softmax over class logits.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Sequence

from meta_jev.core.entropy import best_split, information_gain
from meta_jev.core.tree import apply_quantile_edges
from meta_jev.policy.ig_acquisition import IGAcquisitionPolicy
from meta_jev.policy.predictor import (
    MaskedLogisticPredictor,
    build_predictor,
    configure_policy_predictor,
    prepare_binned_train,
)


def _binary_entropy(p: float) -> float:
    """Binary entropy H(p) in nats; safe at 0/1."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))


def _multi_entropy(probs: Sequence[float]) -> float:
    h = 0.0
    for p in probs:
        pp = min(max(float(p), 1e-12), 1.0)
        h -= pp * math.log(pp)
    return float(h)


class DiscriminativeAcquisitionPolicy(IGAcquisitionPolicy):
    """ELLG acquisition aligned to MaskedLogisticPredictor (ig_discriminative).

    Candidate cap: score only the top-`max_candidates` remaining features by
    static IG (same pattern as ConditionalIG) to keep act() cheap on wide data.
    """

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
        predictor_name: str = "logistic_impute",
        seed: int = 0,  # config symmetry; unused in act
    ) -> None:
        self.criterion = criterion
        self.continuous_keys = list(continuous_keys) if continuous_keys else []
        self.n_bins = n_bins
        self.bin_seed = bin_seed
        self.force_acquisition = force_acquisition
        self.min_samples = int(min_samples)
        self.max_candidates = int(max_candidates)
        # Prefer logistic; fall back only if an explicit non-logistic name is passed.
        self.predictor_name = predictor_name or "logistic_impute"
        self.seed = int(seed)
        self.feature_keys_: list[str] = []
        self.label_key_: str | None = None
        self.ranking_: list[str] = []
        self.ig_scores_: dict[str, float] = {}
        self.train_rows_: list[dict[str, Any]] = []
        self.raw_train_rows_: list[dict[str, Any]] = []
        self.global_majority_: Any = None
        self.bin_edges_: dict[str, list[float]] = {}
        self._predictor = build_predictor(self.predictor_name)
        self._X_codes: Any = None
        self._X_raw: Any = None
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
        Xc = np.empty((n, d), dtype=np.int32)
        for j, fk in enumerate(feature_keys):
            vals = [r.get(fk) for r in prepared]
            uniq = sorted(set(vals), key=lambda x: str(x))
            cmap = {v: i for i, v in enumerate(uniq)}
            code_maps.append(cmap)
            Xc[:, j] = np.asarray([cmap[v] for v in vals], dtype=np.int32)
        self._X_codes = Xc
        self._code_maps = code_maps
        self._y_labels = labels

        # Raw numeric matrix for ELLG bin representatives (aligned with logistic).
        Xr = np.zeros((n, d), dtype=np.float64)
        for j, fk in enumerate(feature_keys):
            col = []
            for r in rows:
                v = r.get(fk)
                try:
                    col.append(float(v))
                except (TypeError, ValueError):
                    col.append(float("nan"))
            arr = np.asarray(col, dtype=np.float64)
            mean = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else 0.0
            Xr[:, j] = np.where(np.isnan(arr), mean, arr)
        self._X_raw = Xr

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

        n = int(self._X_codes.shape[0]) if self._X_codes is not None else 0
        if self._X_codes is None or not acquired:
            return list(range(n))

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
            return list(range(n))

        usable = [i for i in ordered if i in obs_codes]
        X = self._X_codes
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
        return best if best else list(range(n))

    def _value_distribution(
        self, feat_idx: int, subset_idx: Sequence[int]
    ) -> list[tuple[float, float]]:
        """Return [(bin_representative, mass), ...] for feature j on subset.

        Uses quantile bins of raw x_j; mass is empirical frequency. Falls back
        to a single (mean, 1.0) atom if the subset is empty / constant.
        """
        import numpy as np

        if self._X_raw is None or not subset_idx:
            if self._X_raw is None:
                return [(0.0, 1.0)]
            return [(float(self._X_raw[:, feat_idx].mean()), 1.0)]

        vals = self._X_raw[np.asarray(subset_idx, dtype=np.int64), feat_idx]
        if vals.size == 0:
            return [(float(self._X_raw[:, feat_idx].mean()), 1.0)]
        if float(np.std(vals)) < 1e-12 or vals.size < 2:
            return [(float(vals.mean()), 1.0)]

        n_bins = max(2, min(self.n_bins, int(vals.size)))
        # Quantile edges; unique to avoid empty bins on discrete columns.
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(vals, qs))
        if edges.size < 2:
            return [(float(vals.mean()), 1.0)]
        # Digitize into len(edges)-1 bins.
        codes = np.digitize(vals, edges[1:-1], right=False)
        out: list[tuple[float, float]] = []
        total = float(vals.size)
        for b in range(int(codes.max()) + 1):
            m = codes == b
            cnt = int(m.sum())
            if cnt <= 0:
                continue
            out.append((float(vals[m].mean()), cnt / total))
        if not out:
            return [(float(vals.mean()), 1.0)]
        # Renormalize in case of floating drift.
        s = sum(m for _, m in out)
        if s <= 0:
            return [(float(vals.mean()), 1.0)]
        return [(v, m / s) for v, m in out]

    def _predictive_entropy(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        *,
        override_idx: int | None = None,
        override_val: float | None = None,
    ) -> float:
        """Entropy of logistic predictive distribution under (optional) override."""
        if not isinstance(self._predictor, MaskedLogisticPredictor):
            # Non-logistic fallback: treat hard predict as a delta (entropy 0).
            return 0.0

        feats: list[Any] = list(masked_features)
        mask: list[bool] = list(feature_mask)
        if override_idx is not None and override_val is not None:
            feats[override_idx] = float(override_val)
            mask[override_idx] = True

        proba = self._predictor.predict_proba(feats, mask)
        if isinstance(proba, (list, tuple)):
            return _multi_entropy(proba)
        return _binary_entropy(float(proba))

    def _pick_static(self, remaining: list[int]) -> int:
        rank_pos = {fk: i for i, fk in enumerate(self.ranking_)}
        return min(
            remaining,
            key=lambda i: (
                rank_pos.get(self.feature_keys_[i], len(self.ranking_)),
                i,
            ),
        )

    def _ellg(
        self,
        feat_idx: int,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        subset_idx: Sequence[int],
        h_without: float,
    ) -> float:
        dist = self._value_distribution(feat_idx, subset_idx)
        expected_h = 0.0
        for v, mass in dist:
            expected_h += mass * self._predictive_entropy(
                masked_features,
                feature_mask,
                override_idx=feat_idx,
                override_val=v,
            )
        return float(h_without - expected_h)

    def act(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        selection_mask: Sequence[bool] | None = None,
        label: Any = None,  # noqa: ARG002
        feature_shape: Any = None,  # noqa: ARG002
    ) -> int:
        if not self.feature_keys_ or self._X_raw is None:
            raise RuntimeError("DiscriminativeAcquisitionPolicy.fit must be called first")

        remaining = [i for i, seen in enumerate(feature_mask) if not seen]
        if selection_mask is not None:
            remaining = [i for i in remaining if selection_mask[i]]
        if not remaining:
            return 0

        acquired = [i for i, seen in enumerate(feature_mask) if seen]
        # Empty observation: still score ELLG under unconditional p(x_j); no
        # special-case static pick — discriminative from step 0.
        subset = self._match_indices(masked_features, acquired)
        if len(subset) < self.min_samples:
            subset = list(range(int(self._X_raw.shape[0])))

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

        h_without = self._predictive_entropy(masked_features, feature_mask)
        rank_pos = {fk: j for j, fk in enumerate(self.ranking_)}

        best_i = candidates[0]
        best_score = float("-inf")
        best_rank = rank_pos.get(self.feature_keys_[best_i], len(self.ranking_))
        for i in candidates:
            sc = self._ellg(i, masked_features, feature_mask, subset, h_without)
            rnk = rank_pos.get(self.feature_keys_[i], len(self.ranking_))
            if sc > best_score or (
                sc == best_score and (rnk < best_rank or (rnk == best_rank and i < best_i))
            ):
                best_score = sc
                best_i = i
                best_rank = rnk
        return best_i + 1

    def predict(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        label: Any = None,
        feature_shape: Any = None,
    ) -> Any:
        if self.label_key_ is None:
            raise RuntimeError("DiscriminativeAcquisitionPolicy.fit must be called first")
        return self._predictor.predict(
            masked_features, feature_mask, label=label, feature_shape=feature_shape
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DiscriminativeAcquisitionPolicy":  # noqa: ARG002
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, DiscriminativeAcquisitionPolicy):
            raise TypeError(f"expected DiscriminativeAcquisitionPolicy, got {type(obj)}")
        return obj


__all__ = ["DiscriminativeAcquisitionPolicy"]
