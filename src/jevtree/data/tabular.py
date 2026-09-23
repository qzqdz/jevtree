"""Real tabular dataset loaders for jevtree hard-budget eval (P2).

Prefer MiniBooNE (UCI PID text / OpenML 41150). Cache under data/cache/.
Fallback: OpenML diabetes (id=37) with clear provenance if MiniBooNE fails.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from jevtree.data.cube import dataset_hash as rows_dataset_hash

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path("data/cache")

MINIBOONE_DATASET_ID = "miniboone"
MINIBOONE_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt"
)
MINIBOONE_OPENML_ARFF_URL = (
    "https://www.openml.org/data/v1/download/19335523/MiniBooNE.arff"
)
MINIBOONE_OPENML_ID = 41150
MINIBOONE_N_FEATURES = 50
MINIBOONE_LABEL_KEY = "y"
# UCI MiniBooNE uses -999 as an all-feature missing sentinel; those rows are
# exact duplicates of each other and leak across train/test if kept.
MINIBOONE_MISSING_SENTINEL = -999.0

DIABETES_DATASET_ID = "diabetes"
DIABETES_OPENML_CSV_URL = (
    "https://www.openml.org/data/get_csv/37/dataset_37_diabetes"
)
DIABETES_OPENML_ID = 37
DIABETES_LABEL_KEY = "class"

BANK_DATASET_ID = "bank_marketing"
# OpenML bank-marketing (45252 is common; use UCI bank CSV alternative via OpenML 1461)
BANK_OPENML_CSV_URL = "https://www.openml.org/data/get_csv/1586217/phpkIxskf"
BANK_OPENML_ID = 1461
BANK_LABEL_KEY = "Class"


def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").is_file() and (p / "src").is_dir():
            return p
    return Path.cwd()


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        d = Path(cache_dir)
    else:
        d = _repo_root_from_here() / DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, *, timeout: float = 180.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": "jevtree/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# MiniBooNE
# ---------------------------------------------------------------------------

def _parse_miniboone_uci_txt(text: str) -> list[dict[str, Any]]:
    """Parse UCI MiniBooNE_PID.txt (first line: n_signal n_background)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty MiniBooNE file")
    header = lines[0].split()
    if len(header) != 2:
        raise ValueError(f"unexpected MiniBooNE header: {lines[0]!r}")
    n_signal, n_background = int(header[0]), int(header[1])
    data_lines = lines[1:]
    expected = n_signal + n_background
    if len(data_lines) < expected:
        raise ValueError(
            f"MiniBooNE rows {len(data_lines)} < expected {expected}"
        )
    rows: list[dict[str, Any]] = []
    feature_keys = [f"f{i}" for i in range(MINIBOONE_N_FEATURES)]
    for i, ln in enumerate(data_lines[:expected]):
        parts = ln.split()
        if len(parts) != MINIBOONE_N_FEATURES:
            raise ValueError(
                f"MiniBooNE row {i}: expected {MINIBOONE_N_FEATURES} feats, got {len(parts)}"
            )
        row: dict[str, Any] = {
            feature_keys[j]: float(parts[j]) for j in range(MINIBOONE_N_FEATURES)
        }
        # first n_signal are signal (1), rest background (0)
        row[MINIBOONE_LABEL_KEY] = 1 if i < n_signal else 0
        rows.append(row)
    return rows


def _parse_miniboone_openml_arff(text: str) -> list[dict[str, Any]]:
    """Parse OpenML 41150 MiniBooNE ARFF (signal + 50 numeric attrs)."""
    lines = text.splitlines()
    attrs: list[str] = []
    data_start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        low = s.lower()
        if low.startswith("@attribute"):
            # @attribute name type
            rest = s.split(None, 2)
            if len(rest) >= 2:
                name = rest[1].strip().strip("'\"")
                attrs.append(name)
        elif low.startswith("@data"):
            data_start = i + 1
            break
    if data_start is None or not attrs:
        raise ValueError("failed to parse MiniBooNE ARFF header")
    # expect signal + 50 features (or last is signal)
    label_key = "signal" if "signal" in attrs else attrs[0]
    feature_attrs = [a for a in attrs if a != label_key]
    if len(feature_attrs) != MINIBOONE_N_FEATURES:
        # still accept if close; rename to f0..f49 when exactly 50
        pass
    rows: list[dict[str, Any]] = []
    for ln in lines[data_start:]:
        s = ln.strip()
        if not s or s.startswith("%"):
            continue
        parts = [p.strip().strip("'\"") for p in s.split(",")]
        if len(parts) != len(attrs):
            continue
        raw = dict(zip(attrs, parts))
        row: dict[str, Any] = {}
        if len(feature_attrs) == MINIBOONE_N_FEATURES:
            for j, a in enumerate(feature_attrs):
                row[f"f{j}"] = float(raw[a])
        else:
            for a in feature_attrs:
                row[a] = float(raw[a])
        lab = raw[label_key]
        # normalize label to 0/1
        if lab in ("True", "true", "signal", "1", 1, "1.0"):
            row[MINIBOONE_LABEL_KEY] = 1
        elif lab in ("False", "false", "background", "0", 0, "0.0"):
            row[MINIBOONE_LABEL_KEY] = 0
        else:
            row[MINIBOONE_LABEL_KEY] = int(float(lab))
        rows.append(row)
    if not rows:
        raise ValueError("no MiniBooNE ARFF data rows parsed")
    return rows


def load_miniboone_raw(
    *,
    cache_dir: str | Path | None = None,
    timeout: float = 180.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Download (or load cache) full MiniBooNE; return rows + provenance."""
    cache = resolve_cache_dir(cache_dir)
    uci_path = cache / "miniboone_uci_pid.txt"
    arff_path = cache / "miniboone_openml_41150.arff"
    meta_path = cache / "miniboone_source.json"

    provenance: dict[str, Any] = {
        "dataset_id": MINIBOONE_DATASET_ID,
        "label_key": MINIBOONE_LABEL_KEY,
        "n_features_expected": MINIBOONE_N_FEATURES,
    }

    errors: list[str] = []

    # Prefer UCI PID text (classic 50-feature layout)
    try:
        if not uci_path.is_file():
            _download(MINIBOONE_UCI_URL, uci_path, timeout=timeout)
        text = uci_path.read_text(encoding="utf-8", errors="replace")
        rows = _parse_miniboone_uci_txt(text)
        provenance.update(
            {
                "source": "uci",
                "url": MINIBOONE_UCI_URL,
                "cache_path": str(uci_path),
                "file_sha256": file_sha256(uci_path),
                "n_rows_full": len(rows),
            }
        )
        meta_path.write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        return rows, provenance
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        errors.append(f"UCI: {exc}")

    # Fallback: OpenML 41150 ARFF
    try:
        if not arff_path.is_file():
            _download(MINIBOONE_OPENML_ARFF_URL, arff_path, timeout=timeout)
        text = arff_path.read_text(encoding="utf-8", errors="replace")
        rows = _parse_miniboone_openml_arff(text)
        provenance.update(
            {
                "source": "openml",
                "openml_id": MINIBOONE_OPENML_ID,
                "url": MINIBOONE_OPENML_ARFF_URL,
                "cache_path": str(arff_path),
                "file_sha256": file_sha256(arff_path),
                "n_rows_full": len(rows),
            }
        )
        meta_path.write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        return rows, provenance
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        errors.append(f"OpenML: {exc}")

    raise RuntimeError(
        "MiniBooNE download/parse failed:\n  - " + "\n  - ".join(errors)
    )


# ---------------------------------------------------------------------------
# Fallbacks: diabetes / bank marketing
# ---------------------------------------------------------------------------

def _parse_openml_csv(text: str, label_key: str | None = None) -> tuple[list[dict[str, Any]], list[str], str]:
    reader = csv.DictReader(io.StringIO(text))
    rows = [dict(r) for r in reader]
    if not rows:
        raise ValueError("empty OpenML CSV")
    cols = list(rows[0].keys())
    # strip BOM / quotes from keys
    cols = [c.strip().strip('"') for c in cols]
    remapped: list[dict[str, Any]] = []
    for r in rows:
        nr = {k.strip().strip('"'): v for k, v in r.items()}
        remapped.append(nr)
    rows = remapped
    cols = list(rows[0].keys())
    if label_key is None or label_key not in cols:
        # last column as label
        label_key = cols[-1]
    feature_keys = [c for c in cols if c != label_key]
    # coerce numerics where possible
    out: list[dict[str, Any]] = []
    for r in rows:
        nr: dict[str, Any] = {}
        for k in feature_keys:
            v = r[k]
            try:
                nr[k] = float(v)
            except (TypeError, ValueError):
                nr[k] = v
        lab = r[label_key]
        try:
            # diabetes often 'tested_positive'/'tested_negative'
            nr[label_key] = lab
        except Exception:
            nr[label_key] = lab
        out.append(nr)
    return out, feature_keys, label_key


def load_diabetes_raw(
    *,
    cache_dir: str | Path | None = None,
    timeout: float = 60.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = resolve_cache_dir(cache_dir)
    path = cache / "diabetes_openml_37.csv"
    if not path.is_file():
        _download(DIABETES_OPENML_CSV_URL, path, timeout=timeout)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows, feature_keys, label_key = _parse_openml_csv(text, label_key=None)
    # normalize common OpenML diabetes label column names
    for cand in ("class", "Class", "Outcome", "outcome"):
        if cand in rows[0]:
            label_key = cand
            feature_keys = [k for k in rows[0].keys() if k != label_key]
            break
    provenance = {
        "dataset_id": DIABETES_DATASET_ID,
        "source": "openml",
        "openml_id": DIABETES_OPENML_ID,
        "url": DIABETES_OPENML_CSV_URL,
        "cache_path": str(path),
        "file_sha256": file_sha256(path),
        "n_rows_full": len(rows),
        "label_key": label_key,
        "feature_keys": feature_keys,
    }
    return rows, provenance


def load_bank_marketing_raw(
    *,
    cache_dir: str | Path | None = None,
    timeout: float = 120.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = resolve_cache_dir(cache_dir)
    path = cache / "bank_marketing_openml_1461.csv"
    if not path.is_file():
        _download(BANK_OPENML_CSV_URL, path, timeout=timeout)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows, feature_keys, label_key = _parse_openml_csv(text, label_key=None)
    for cand in ("Class", "class", "y", "Target"):
        if cand in rows[0]:
            label_key = cand
            feature_keys = [k for k in rows[0].keys() if k != label_key]
            break
    provenance = {
        "dataset_id": BANK_DATASET_ID,
        "source": "openml",
        "openml_id": BANK_OPENML_ID,
        "url": BANK_OPENML_CSV_URL,
        "cache_path": str(path),
        "file_sha256": file_sha256(path),
        "n_rows_full": len(rows),
        "label_key": label_key,
        "feature_keys": feature_keys,
    }
    return rows, provenance


# ---------------------------------------------------------------------------
# Split + public API
# ---------------------------------------------------------------------------

def _is_all_sentinel_row(
    row: dict[str, Any],
    feature_keys: Sequence[str],
    *,
    sentinel: float = MINIBOONE_MISSING_SENTINEL,
    atol: float = 1e-9,
) -> bool:
    """True if every feature equals the missing sentinel (e.g. all -999)."""
    if not feature_keys:
        return False
    for fk in feature_keys:
        v = row.get(fk)
        try:
            if abs(float(v) - float(sentinel)) > atol:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _feature_vector_key(
    row: dict[str, Any], feature_keys: Sequence[str]
) -> tuple[Any, ...]:
    return tuple(row.get(fk) for fk in feature_keys)


def drop_all_sentinel_rows(
    rows: list[dict[str, Any]],
    feature_keys: Sequence[str],
    *,
    sentinel: float = MINIBOONE_MISSING_SENTINEL,
) -> tuple[list[dict[str, Any]], int]:
    """Drop rows whose features are entirely the missing sentinel."""
    kept = [
        r
        for r in rows
        if not _is_all_sentinel_row(r, feature_keys, sentinel=sentinel)
    ]
    return kept, len(rows) - len(kept)


def ensure_train_test_disjoint_features(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    feature_keys: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Remove test rows whose feature vectors are exact duplicates of any train row.

    Train is left unchanged. Returns (train, filtered_test, n_removed_from_test).
    """
    train_keys = {_feature_vector_key(r, feature_keys) for r in train}
    filtered: list[dict[str, Any]] = []
    removed = 0
    for r in test:
        if _feature_vector_key(r, feature_keys) in train_keys:
            removed += 1
        else:
            filtered.append(r)
    return train, filtered, removed



def _deterministic_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    n_train: int | None,
    n_test: int | None,
    train_ratio: float = 0.7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    if n_train is not None and n_test is not None:
        need = n_train + n_test
        if need > len(idx):
            # take all, split proportionally
            cut = max(1, int(len(idx) * train_ratio))
            train_idx = idx[:cut]
            test_idx = idx[cut:] or idx[-1:]
            # then subsample
            rng2 = random.Random(seed + 91)
            if len(train_idx) > n_train:
                train_idx = rng2.sample(train_idx, n_train)
            if len(test_idx) > n_test:
                test_idx = rng2.sample(test_idx, n_test)
        else:
            train_idx = idx[:n_train]
            test_idx = idx[n_train : n_train + n_test]
        subsample = {"n_train": n_train, "n_test": n_test, "mode": "fixed_counts"}
    else:
        cut = max(1, int(len(idx) * train_ratio))
        train_idx = idx[:cut]
        test_idx = idx[cut:] or idx[-1:]
        subsample = {"n_train": len(train_idx), "n_test": len(test_idx), "mode": "ratio"}
    train = [rows[i] for i in train_idx]
    test = [rows[i] for i in test_idx]
    return train, test, subsample


def load_tabular_split(
    dataset_id: str,
    *,
    seed: int = 0,
    n_train: int | None = None,
    n_test: int | None = None,
    train_ratio: float = 0.7,
    cache_dir: str | Path | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Load real tabular split by dataset_id with provenance.

    Supported ids: miniboone, diabetes, bank_marketing.
    If miniboone fails and allow_fallback, tries diabetes then bank_marketing.
    """
    wanted = dataset_id
    loaders = {
        MINIBOONE_DATASET_ID: load_miniboone_raw,
        DIABETES_DATASET_ID: load_diabetes_raw,
        BANK_DATASET_ID: load_bank_marketing_raw,
    }
    fallback_order = [DIABETES_DATASET_ID, BANK_DATASET_ID]

    tried: list[str] = []
    last_err: Exception | None = None
    rows: list[dict[str, Any]] | None = None
    provenance: dict[str, Any] = {}
    used_id = wanted

    candidates = [wanted]
    if allow_fallback and wanted == MINIBOONE_DATASET_ID:
        candidates = [wanted] + fallback_order
    elif wanted not in loaders:
        raise ValueError(
            f"unknown dataset_id={wanted!r}; "
            f"supported: {sorted(loaders)} | cube_without_noise"
        )

    for cand in candidates:
        tried.append(cand)
        try:
            rows, provenance = loaders[cand](cache_dir=cache_dir)
            used_id = cand
            break
        except Exception as exc:  # noqa: BLE001 — collect and try next
            last_err = exc
            continue
    if rows is None:
        raise RuntimeError(
            f"failed to load dataset_id={wanted!r} (tried={tried}): {last_err}"
        )

    # feature / label keys
    if used_id == MINIBOONE_DATASET_ID:
        feature_keys = [f"f{i}" for i in range(MINIBOONE_N_FEATURES)]
        label_key = MINIBOONE_LABEL_KEY
        # defaults for first P2 curve
        if n_train is None:
            n_train = 2000
        if n_test is None:
            n_test = 500
    else:
        feature_keys = list(provenance.get("feature_keys") or [])
        label_key = str(provenance.get("label_key") or "y")
        if not feature_keys:
            feature_keys = [k for k in rows[0].keys() if k != label_key]

    hygiene: dict[str, Any] = {
        "dropped_all_sentinel": 0,
        "sentinel": None,
        "removed_test_feature_leaks": 0,
    }
    if used_id == MINIBOONE_DATASET_ID:
        rows, n_drop = drop_all_sentinel_rows(
            rows, feature_keys, sentinel=MINIBOONE_MISSING_SENTINEL
        )
        hygiene["dropped_all_sentinel"] = int(n_drop)
        hygiene["sentinel"] = MINIBOONE_MISSING_SENTINEL
        provenance = {
            **provenance,
            "n_rows_after_sentinel_drop": len(rows),
            "dropped_all_sentinel": int(n_drop),
        }

    train, test, subsample = _deterministic_split(
        rows,
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        train_ratio=train_ratio,
    )

    # Ensure train/test disjoint by exact feature-vector identity (catches residual
    # duplicates that are not all-sentinel, and any sentinel rows that slipped through).
    train, test, n_leak = ensure_train_test_disjoint_features(train, test, feature_keys)
    hygiene["removed_test_feature_leaks"] = int(n_leak)
    # If MiniBooNE fixed counts and we removed leaks, refill test from unused pool.
    if (
        used_id == MINIBOONE_DATASET_ID
        and n_test is not None
        and len(test) < int(n_test)
    ):
        need = int(n_test) - len(test)
        train_keys = {_feature_vector_key(r, feature_keys) for r in train}
        test_keys = {_feature_vector_key(r, feature_keys) for r in test}
        used = train_keys | test_keys
        # rebuild index order with same RNG stream as split for stability
        rng_refill = random.Random(seed + 907)
        candidates = [
            r
            for r in rows
            if _feature_vector_key(r, feature_keys) not in used
        ]
        rng_refill.shuffle(candidates)
        for r in candidates[:need]:
            test.append(r)
            used.add(_feature_vector_key(r, feature_keys))
        hygiene["refilled_test"] = int(min(need, len(candidates)))
        subsample = {
            **subsample,
            "n_test_after_hygiene": len(test),
            "hygiene": hygiene,
        }

    # hash of the *eval subsample* AFTER hygiene (honest provenance)
    split_hash = rows_dataset_hash(train + test)
    full_hash = provenance.get("file_sha256") or rows_dataset_hash(rows)

    continuous_keys = list(feature_keys)  # all numeric for these real sets

    result = {
        "dataset_id": used_id,
        "requested_dataset_id": wanted,
        "fallback_used": used_id != wanted,
        "dataset_hash": split_hash,
        "full_file_sha256": full_hash,
        "feature_keys": feature_keys,
        "label_key": label_key,
        "continuous_keys": continuous_keys,
        "train": train,
        "test": test,
        "split_seed": seed,
        "subsample": subsample,
        "provenance": {
            **provenance,
            "tried": tried,
            "subsample": subsample,
            "split_seed": seed,
            "dataset_hash_split": split_hash,
            "hygiene": hygiene,
        },
        "hygiene": hygiene,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(feature_keys),
    }
    return result


__all__ = [
    "BANK_DATASET_ID",
    "DIABETES_DATASET_ID",
    "MINIBOONE_DATASET_ID",
    "MINIBOONE_MISSING_SENTINEL",
    "drop_all_sentinel_rows",
    "ensure_train_test_disjoint_features",
    "load_bank_marketing_raw",
    "load_diabetes_raw",
    "load_miniboone_raw",
    "load_tabular_split",
    "resolve_cache_dir",
]
