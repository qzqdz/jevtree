"""Shared predictors for AFA policies.

MatchMajorityPredictor: equality-match majority vote over binned train rows
  (progressive drop of lowest-ranked features).

MaskedLogisticPredictor: full-feature L2 logistic at fit; mean-impute unobserved
  dims at predict (numpy GD; binary / one-vs-rest).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from jevtree.core.tree import (
    apply_quantile_edges,
    bin_rows_continuous,
    bin_rows_with_edges,
    compute_quantile_edges,
)


def majority(labels: Sequence[Any]) -> Any:
    if not labels:
        return None
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def detect_continuous_keys(
    rows: list[dict[str, Any]],
    feature_keys: list[str],
    *,
    n_bins: int = 4,
) -> list[str]:
    """Auto-detect float-like / high-cardinality numeric columns."""
    cont: list[str] = []
    for fk in feature_keys:
        vals = [r.get(fk) for r in rows if fk in r]
        if vals and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals
        ):
            if any(isinstance(v, float) and not float(v).is_integer() for v in vals):
                cont.append(fk)
            elif len(set(vals)) > max(4, n_bins):
                cont.append(fk)
    return cont


def prepare_binned_train(
    rows: list[dict[str, Any]],
    feature_keys: list[str],
    *,
    continuous_keys: Sequence[str] | None = None,
    n_bins: int = 4,
    bin_seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, list[float]], list[str]]:
    """Bin continuous features; return (prepared_rows, bin_edges, continuous_keys)."""
    cont = list(continuous_keys) if continuous_keys else []
    if not cont:
        cont = detect_continuous_keys(rows, feature_keys, n_bins=n_bins)

    bin_edges: dict[str, list[float]] = {}
    if cont:
        for fk in cont:
            if fk not in feature_keys:
                continue
            edges = compute_quantile_edges(
                [float(r[fk]) for r in rows], n_bins=n_bins
            )
            bin_edges[fk] = edges
        prepared = bin_rows_with_edges(rows, bin_edges)
    else:
        prepared = bin_rows_continuous(
            rows,
            feature_keys,
            continuous_keys=cont,
            n_bins=n_bins,
            seed=bin_seed,
        )
    return prepared, bin_edges, cont


class MatchMajorityPredictor:
    """Equality-match majority vote over binned train rows.

    Progressive relaxation drops lowest-ranked acquired features first until
    at least one train match appears (same logic formerly inlined in StaticIG).
    """

    def __init__(self) -> None:
        self.feature_keys_: list[str] = []
        self.label_key_: str | None = None
        self.train_rows_: list[dict[str, Any]] = []
        self.global_majority_: Any = None
        self.bin_edges_: dict[str, list[float]] = {}
        self.ranking_: list[str] = []

    def configure(
        self,
        *,
        feature_keys: list[str],
        label_key: str,
        train_rows: list[dict[str, Any]],
        bin_edges: dict[str, list[float]],
        ranking: list[str],
        global_majority: Any | None = None,
    ) -> None:
        self.feature_keys_ = list(feature_keys)
        self.label_key_ = label_key
        self.train_rows_ = list(train_rows)
        self.bin_edges_ = dict(bin_edges)
        self.ranking_ = list(ranking)
        if global_majority is None:
            labels = [r[label_key] for r in self.train_rows_]
            self.global_majority_ = majority(labels)
        else:
            self.global_majority_ = global_majority

    def bin_value(self, feature_key: str, value: Any) -> Any:
        if feature_key in self.bin_edges_:
            if value is None:
                return None
            try:
                return apply_quantile_edges(float(value), self.bin_edges_[feature_key])
            except (TypeError, ValueError):
                return value
        return value

    def predict(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        label: Any = None,  # noqa: ARG002
        feature_shape: Any = None,  # noqa: ARG002
    ) -> Any:
        if self.label_key_ is None:
            raise RuntimeError("MatchMajorityPredictor.configure must be called first")

        acquired = [i for i, seen in enumerate(feature_mask) if seen]
        if not acquired:
            return self.global_majority_

        obs: dict[int, Any] = {}
        for i in acquired:
            fk = self.feature_keys_[i]
            obs[i] = self.bin_value(fk, masked_features[i])

        rank_pos = {fk: j for j, fk in enumerate(self.ranking_)}
        ordered = sorted(
            acquired,
            key=lambda i: (rank_pos.get(self.feature_keys_[i], len(self.ranking_)), i),
        )

        for keep in range(len(ordered), 0, -1):
            use = ordered[:keep]
            matches: list[Any] = []
            for row in self.train_rows_:
                ok = True
                for i in use:
                    fk = self.feature_keys_[i]
                    if row.get(fk) != obs[i]:
                        ok = False
                        break
                if ok:
                    matches.append(row[self.label_key_])
            if matches:
                return majority(matches)

        return self.global_majority_

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_keys": self.feature_keys_,
            "label_key": self.label_key_,
            "train_rows": self.train_rows_,
            "global_majority": self.global_majority_,
            "bin_edges": self.bin_edges_,
            "ranking": self.ranking_,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.feature_keys_ = list(payload.get("feature_keys", []))
        self.label_key_ = payload.get("label_key")
        self.train_rows_ = list(payload.get("train_rows", []))
        self.global_majority_ = payload.get("global_majority")
        raw_edges = payload.get("bin_edges") or {}
        self.bin_edges_ = {
            str(k): [float(e) for e in v] for k, v in raw_edges.items()
        }
        self.ranking_ = list(payload.get("ranking", []))


def _sigmoid(z):  # type: ignore[no-untyped-def]
    import numpy as np

    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_binary_logistic(
    X,  # type: ignore[no-untyped-def]
    y,
    *,
    l2: float = 0.01,
    lr: float = 0.5,
    n_iter: int = 1500,
):
    """L2-regularized logistic regression via GD (numpy). Returns (w, b)."""
    import numpy as np

    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(n_iter):
        p = _sigmoid(X @ w + b)
        err = p - y
        gw = (X.T @ err) / n + l2 * w
        gb = float(err.mean())
        w = w - lr * gw
        b = b - lr * gb
    return w, b


class MaskedLogisticPredictor:
    """Full-feature logistic at fit; mean-impute unobserved features at predict.

    Uses raw numeric train rows (not IG bins). Unobserved dims → train mean.
    Binary MiniBooNE path is primary; multiclass uses one-vs-rest.
    """

    def __init__(
        self,
        *,
        l2: float = 0.01,
        lr: float = 0.5,
        n_iter: int = 1500,
    ) -> None:
        self.l2 = float(l2)
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.feature_keys_: list[str] = []
        self.label_key_: str | None = None
        self.ranking_: list[str] = []
        self.global_majority_: Any = None
        self.classes_: list[Any] = []
        self.feature_means_: list[float] = []
        self.feature_stds_: list[float] = []
        self.weights_: Any = None
        self.bias_: Any = None
        self._binary: bool = True

    def configure(
        self,
        *,
        feature_keys: list[str],
        label_key: str,
        train_rows: list[dict[str, Any]],
        ranking: list[str] | None = None,
        bin_edges: dict[str, list[float]] | None = None,  # noqa: ARG002
        global_majority: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        import numpy as np

        self.feature_keys_ = list(feature_keys)
        self.label_key_ = label_key
        self.ranking_ = list(ranking or [])
        d = len(feature_keys)
        n = len(train_rows)
        if n == 0 or d == 0:
            raise ValueError("MaskedLogisticPredictor needs non-empty train rows/features")

        X = np.zeros((n, d), dtype=np.float64)
        for j, fk in enumerate(feature_keys):
            col = []
            for r in train_rows:
                v = r.get(fk)
                try:
                    col.append(float(v))
                except (TypeError, ValueError):
                    col.append(float("nan"))
            arr = np.asarray(col, dtype=np.float64)
            mean = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else 0.0
            arr = np.where(np.isnan(arr), mean, arr)
            X[:, j] = arr

        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)
        self.feature_means_ = means.tolist()
        self.feature_stds_ = stds.tolist()
        Xn = (X - means) / stds

        labels = [r[label_key] for r in train_rows]
        self.global_majority_ = (
            global_majority if global_majority is not None else majority(labels)
        )
        classes = sorted(set(labels), key=lambda x: str(x))
        self.classes_ = classes

        if len(classes) <= 1:
            self._binary = True
            self.weights_ = np.zeros((1, d), dtype=np.float64)
            self.bias_ = np.zeros(1, dtype=np.float64)
            return

        if len(classes) == 2:
            self._binary = True
            y = np.asarray(
                [1.0 if lab == classes[1] else 0.0 for lab in labels], dtype=np.float64
            )
            w, b = _fit_binary_logistic(
                Xn, y, l2=self.l2, lr=self.lr, n_iter=self.n_iter
            )
            self.weights_ = w.reshape(1, -1)
            self.bias_ = np.asarray([b], dtype=np.float64)
        else:
            self._binary = False
            ws = []
            bs = []
            for c in classes:
                y = np.asarray(
                    [1.0 if lab == c else 0.0 for lab in labels], dtype=np.float64
                )
                w, b = _fit_binary_logistic(
                    Xn, y, l2=self.l2, lr=self.lr, n_iter=self.n_iter
                )
                ws.append(w)
                bs.append(b)
            self.weights_ = np.vstack(ws)
            self.bias_ = np.asarray(bs, dtype=np.float64)

    def _impute_vector(self, masked_features: Sequence[Any], feature_mask: Sequence[bool]):
        import numpy as np

        d = len(self.feature_keys_)
        x = np.asarray(self.feature_means_, dtype=np.float64).copy()
        for i, seen in enumerate(feature_mask):
            if not seen or i >= d:
                continue
            v = masked_features[i]
            if v is None:
                continue
            try:
                x[i] = float(v)
            except (TypeError, ValueError):
                continue
        means = np.asarray(self.feature_means_, dtype=np.float64)
        stds = np.asarray(self.feature_stds_, dtype=np.float64)
        return (x - means) / stds

    def predict_proba(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
    ) -> float | list[float]:
        """Return P(positive) for binary, or class-prob list for multiclass.

        Uses the same mean-impute + standardize path as predict(). For binary
        with a single class, returns 1.0. Multiclass probs come from a stable
        softmax over one-vs-rest logits.
        """
        import numpy as np

        if self.label_key_ is None or self.weights_ is None:
            raise RuntimeError("MaskedLogisticPredictor.configure must be called first")
        if not self.classes_:
            return 0.5

        x = self._impute_vector(masked_features, feature_mask)
        if self._binary:
            if len(self.classes_) == 1:
                return 1.0
            logit = float(x @ self.weights_[0] + self.bias_[0])
            return float(_sigmoid(logit))

        logits = np.asarray(self.weights_ @ x + self.bias_, dtype=np.float64)
        logits = logits - logits.max()
        ex = np.exp(logits)
        probs = ex / ex.sum()
        return [float(p) for p in probs]

    def predict(
        self,
        masked_features: Sequence[Any],
        feature_mask: Sequence[bool],
        label: Any = None,  # noqa: ARG002
        feature_shape: Any = None,  # noqa: ARG002
    ) -> Any:
        import numpy as np

        if self.label_key_ is None or self.weights_ is None:
            raise RuntimeError("MaskedLogisticPredictor.configure must be called first")
        if not self.classes_:
            return self.global_majority_

        if self._binary:
            if len(self.classes_) == 1:
                return self.classes_[0]
            p = float(self.predict_proba(masked_features, feature_mask))  # type: ignore[arg-type]
            return self.classes_[1] if p >= 0.5 else self.classes_[0]

        probs = self.predict_proba(masked_features, feature_mask)
        assert isinstance(probs, list)
        return self.classes_[int(np.argmax(np.asarray(probs)))]


def normalize_predictor_name(name: str | None) -> str:
    """Map config predictor.name → canonical key."""
    n = (name or "match_majority").strip().lower()
    if n in ("logistic_impute", "logistic", "masked_logistic"):
        return "logistic_impute"
    if n in ("match_majority", "policy.predict", "majority"):
        return "match_majority"
    return n


def build_predictor(name: str | None = "match_majority") -> Any:
    """Factory: match_majority | logistic_impute."""
    key = normalize_predictor_name(name)
    if key == "logistic_impute":
        return MaskedLogisticPredictor()
    return MatchMajorityPredictor()


def configure_policy_predictor(
    predictor: Any,
    *,
    feature_keys: list[str],
    label_key: str,
    binned_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    bin_edges: dict[str, list[float]],
    ranking: list[str],
    global_majority: Any | None = None,
) -> Any:
    """Configure predictor with the right row representation."""
    if isinstance(predictor, MaskedLogisticPredictor):
        predictor.configure(
            feature_keys=feature_keys,
            label_key=label_key,
            train_rows=raw_rows,
            ranking=ranking,
            global_majority=global_majority,
        )
    else:
        predictor.configure(
            feature_keys=feature_keys,
            label_key=label_key,
            train_rows=binned_rows,
            bin_edges=bin_edges,
            ranking=ranking,
            global_majority=global_majority,
        )
    return predictor


__all__ = [
    "MatchMajorityPredictor",
    "MaskedLogisticPredictor",
    "majority",
    "detect_continuous_keys",
    "prepare_binned_train",
    "normalize_predictor_name",
    "build_predictor",
    "configure_policy_predictor",
]
