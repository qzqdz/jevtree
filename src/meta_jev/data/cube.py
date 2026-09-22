"""Synthetic cube_without_noise-like tabular data (no external AFABench dep).

Label is a deterministic function of the first features so static IG has a
clear ranking (f0, f1 matter most; distractors near zero IG).
"""

from __future__ import annotations

import hashlib
import itertools
import random
from typing import Any


DATASET_ID = "cube_without_noise"
DEFAULT_FEATURES = ["f0", "f1", "f2", "f3", "f4"]
LABEL_KEY = "y"


def _label_from_features(f0: int, f1: int, f2: int) -> int:
    """Deterministic rule: y = (f0 XOR f1) OR (f0 AND f2).

    Makes f0 the strongest single feature, then f1/f2, with f3/f4 irrelevant.
    """
    return int((f0 ^ f1) or (f0 & f2))


def generate_cube(
    *,
    n_samples: int = 256,
    n_features: int = 5,
    seed: int = 0,
    exhaust_binary: bool = True,
) -> list[dict[str, Any]]:
    """Generate binary/categorical cube-like rows.

    If *exhaust_binary* and n_features <= 5, include all 2^{min(3,n_features)}
    core combinations repeated, plus random fills to reach n_samples.
    """
    if n_features < 3:
        raise ValueError("n_features must be >= 3")
    feature_keys = [f"f{i}" for i in range(n_features)]
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    core_bits = min(3, n_features)
    if exhaust_binary:
        for combo in itertools.product([0, 1], repeat=core_bits):
            row = {fk: 0 for fk in feature_keys}
            for i, bit in enumerate(combo):
                row[feature_keys[i]] = bit
            for i in range(core_bits, n_features):
                row[feature_keys[i]] = rng.randint(0, 1)
            row[LABEL_KEY] = _label_from_features(row["f0"], row["f1"], row["f2"])
            rows.append(row)

    while len(rows) < n_samples:
        row = {fk: rng.randint(0, 1) for fk in feature_keys}
        row[LABEL_KEY] = _label_from_features(row["f0"], row["f1"], row["f2"])
        rows.append(row)

    rng.shuffle(rows)
    return rows[:n_samples]


def dataset_hash(rows: list[dict[str, Any]]) -> str:
    """Stable short hash over sorted JSON-ish row content."""
    blob = repr(sorted((tuple(sorted(r.items())) for r in rows))).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_cube_split(
    *,
    n_samples: int = 256,
    n_features: int = 5,
    seed: int = 0,
    train_ratio: float = 0.7,
) -> dict[str, Any]:
    """Train/test split for cube_without_noise smoke.

    Returns dict with train/test rows, feature_keys, label_key, dataset_id, hash.
    """
    rows = generate_cube(n_samples=n_samples, n_features=n_features, seed=seed)
    feature_keys = [f"f{i}" for i in range(n_features)]
    rng = random.Random(seed + 17)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = max(1, int(len(idx) * train_ratio))
    train = [rows[i] for i in idx[:cut]]
    test = [rows[i] for i in idx[cut:]]
    if not test:
        test = train[-1:]
        train = train[:-1] or train
    return {
        "dataset_id": DATASET_ID,
        "dataset_hash": dataset_hash(rows),
        "feature_keys": feature_keys,
        "label_key": LABEL_KEY,
        "train": train,
        "test": test,
        "split_seed": seed,
        "n_samples": n_samples,
        "n_features": n_features,
    }


__all__ = [
    "DATASET_ID",
    "DEFAULT_FEATURES",
    "LABEL_KEY",
    "dataset_hash",
    "generate_cube",
    "load_cube_split",
]
