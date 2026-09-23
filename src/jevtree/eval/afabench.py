"""AFABench-shaped thin adapter over jevtree IG policies (P2).

Wraps StaticIGAcquisitionPolicy with an AFAMethod-compatible surface:
  act / predict / save / load / to(device) / device
  force_acquisition, has_builtin_classifier
  optional cost_param / set_cost_param stubs

Does NOT vendor AFABench. Protocol-compatible without importing afabench;
optional to_afabench_method() exports only when afabench is importable.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence

from jevtree.policy.ig_acquisition import IGAcquisitionPolicy, StaticIGAcquisitionPolicy


def _as_list(x: Any) -> list[Any]:
    """Best-effort convert tensor / ndarray / sequence to a flat Python list."""
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        # squeeze singleton batch dim if present: [[...]] -> [...]
        if len(x) == 1 and isinstance(x[0], (list, tuple)) and (
            not x[0] or not isinstance(x[0][0], (list, tuple))
        ):
            # ambiguous; only squeeze if looks like batch of feature vector
            inner = x[0]
            if all(not isinstance(v, (list, tuple)) for v in inner):
                return list(inner)
        return list(x)
    return [x]


def _as_bool_mask(x: Any) -> list[bool]:
    return [bool(v) for v in _as_list(x)]


class AFABenchAdapter:
    """Thin adapter: jevtree StaticIGAcquisitionPolicy ↔ AFAMethod shape."""

    def __init__(
        self,
        policy: IGAcquisitionPolicy | None = None,
        *,
        force_acquisition: bool | None = None,
        has_builtin_classifier: bool = True,
        cost_param: float | None = None,
        device: str = "cpu",
        predictor_name: str = "policy.predict",
    ) -> None:
        self.policy = policy if policy is not None else StaticIGAcquisitionPolicy()
        if force_acquisition is not None:
            self.force_acquisition = bool(force_acquisition)
            self.policy.force_acquisition = self.force_acquisition
        else:
            self.force_acquisition = bool(getattr(self.policy, "force_acquisition", False))
        self.has_builtin_classifier = bool(has_builtin_classifier)
        self._cost_param = cost_param
        self._device = str(device)
        self.predictor_name = predictor_name

    # ------------------------------------------------------------------ act / predict

    def act(
        self,
        masked_features: Sequence[Any] | Any = (),
        feature_mask: Sequence[bool] | Any = (),
        selection_mask: Sequence[bool] | Any | None = None,
        label: Any = None,
        feature_shape: Any = None,
    ) -> int:
        self.policy.force_acquisition = bool(self.force_acquisition)
        mf = _as_list(masked_features)
        fm = _as_bool_mask(feature_mask)
        sm = _as_bool_mask(selection_mask) if selection_mask is not None else None
        action = self.policy.act(
            masked_features=mf,
            feature_mask=fm,
            selection_mask=sm,
            label=label,
            feature_shape=feature_shape,
        )
        return int(action)

    def predict(
        self,
        masked_features: Sequence[Any] | Any = (),
        feature_mask: Sequence[bool] | Any = (),
        label: Any = None,
        feature_shape: Any = None,
    ) -> Any:
        mf = _as_list(masked_features)
        fm = _as_bool_mask(feature_mask)
        return self.policy.predict(
            masked_features=mf,
            feature_mask=fm,
            label=label,
            feature_shape=feature_shape,
        )

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json" or p.is_dir() or p.suffix == "":
            # directory bundle preferred for AFABench-like layout
            out_dir = p if p.suffix == "" or p.is_dir() else p.with_suffix("")
            if p.suffix.lower() == ".json":
                out_dir = p.parent / p.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "type": "AFABenchAdapter",
                "force_acquisition": self.force_acquisition,
                "has_builtin_classifier": self.has_builtin_classifier,
                "cost_param": self._cost_param,
                "device": self._device,
                "predictor_name": self.predictor_name,
                "policy_file": "policy.json",
            }
            with open(out_dir / "adapter.json", "w", encoding="utf-8", newline="\n") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
                f.write("\n")
            self.policy.save(str(out_dir / "policy.json"))
        else:
            with open(p, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "AFABenchAdapter":
        p = Path(path)
        if p.is_dir() or (p.with_suffix("").is_dir() and not p.is_file()):
            out_dir = p if p.is_dir() else p.with_suffix("")
            meta_path = out_dir / "adapter.json"
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            policy = StaticIGAcquisitionPolicy.load(str(out_dir / "policy.json"), device=device)
            obj = cls(
                policy,
                force_acquisition=bool(meta.get("force_acquisition", False)),
                has_builtin_classifier=bool(meta.get("has_builtin_classifier", True)),
                cost_param=meta.get("cost_param"),
                device=str(device or meta.get("device", "cpu")),
                predictor_name=str(meta.get("predictor_name", "policy.predict")),
            )
            return obj
        if p.suffix.lower() == ".json" and p.is_file():
            # single-file: treat as policy.json wrapped with defaults
            # or adapter.json next to policy
            parent = p.parent
            if (parent / "adapter.json").is_file() and p.name == "policy.json":
                return cls.load(parent, device=device)
            policy = StaticIGAcquisitionPolicy.load(str(p), device=device)
            return cls(policy, device=device)
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, AFABenchAdapter):
            raise TypeError(f"expected AFABenchAdapter, got {type(obj)}")
        obj._device = str(device)
        return obj

    # ------------------------------------------------------------------ device / cost stubs

    def to(self, device: Any) -> "AFABenchAdapter":
        self._device = str(device)
        return self

    @property
    def device(self) -> str:
        return self._device

    @property
    def cost_param(self) -> float | None:
        return self._cost_param

    def set_cost_param(self, cost_param: float) -> None:
        self._cost_param = float(cost_param)

    def set_seed(self, seed: int | None) -> None:
        """No-op for deterministic static IG (AFAMethod stub)."""
        return None

    # ------------------------------------------------------------------ optional AFABench bridge

    def to_afabench_method(self) -> Any:
        """Return self if afabench is importable; else raise with guidance.

        jevtree keeps a thin protocol-compatible adapter and does not vendor
        AFABench. Full snakemake registration remains deferred.
        """
        try:
            import afabench  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "afabench is not installed in this environment. "
                "AFABenchAdapter is protocol-compatible (act/predict/save/load/"
                "to/device/force_acquisition/has_builtin_classifier/cost_param) "
                "without importing afabench. Full AFABench snakemake wiring is deferred."
            ) from exc
        return self


__all__ = ["AFABenchAdapter"]
