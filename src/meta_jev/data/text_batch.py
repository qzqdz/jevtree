"""Batch labeled/scored texts → FeatureTable via keyword presence features.

Offline-first: mine a small vocabulary from the batch and emit yes/no
presence cues. Optional LLM enrichment is a separate helper.
Works for topic classification AND grading/scoring (label = class or score).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from meta_jev.data.feature_table import (
    FeatureTable,
    IngestResult,
    LabelKind,
    coerce_labels_discrete,
)

_TOKEN_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]{2,}")

# Very small bilingual stop-ish set (not a full NLP stack).
_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "you", "your", "our", "their", "they",
    "will", "can", "may", "also", "into", "about", "been", "being", "which",
    "what", "when", "where", "there", "here", "than", "then", "them", "its",
    "his", "her", "she", "him", "who", "how", "why", "all", "any", "some",
    "more", "most", "other", "such", "only", "over", "after", "before",
    "in", "on", "at", "to", "of", "as", "by", "or", "an", "be", "is", "it",
    "my", "me", "we", "us", "do", "did", "does", "if", "so", "no", "yes",
    "just", "very", "too", "out", "up", "down", "off", "via", "per",
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "那", "他", "她", "我们", "你们", "他们",
}


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for t in _TOKEN_RE.findall(text or ""):
        low = t.lower()
        if low in _STOP:
            continue
        # Prefer contentful tokens: ASCII length >= 3, CJK keep >= 2
        if all(ord(c) < 128 for c in low) and len(low) < 3:
            continue
        out.append(low)
    return out


def mine_vocabulary(
    texts: Sequence[str],
    *,
    max_features: int = 12,
    min_df: int = 2,
) -> list[str]:
    """Pick frequent tokens as presence-feature names (kw_*)."""
    df: Counter[str] = Counter()
    for t in texts:
        df.update(set(tokenize(t)))
    candidates = [(tok, c) for tok, c in df.items() if c >= min_df]
    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    vocab = [tok for tok, _ in candidates[:max_features]]
    # fallback: allow min_df=1 if too sparse
    if len(vocab) < 3:
        candidates = list(df.items())
        candidates.sort(key=lambda kv: (-kv[1], kv[0]))
        vocab = [tok for tok, _ in candidates[:max_features]]
    return vocab


def texts_to_feature_rows(
    items: Sequence[dict[str, Any]],
    *,
    text_key: str = "text",
    label_key: str = "label",
    vocab: Sequence[str] | None = None,
    max_features: int = 12,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Convert {text, label, ...} items to discrete presence-feature rows."""
    texts = [str(it.get(text_key, "")) for it in items]
    used_vocab = list(vocab) if vocab is not None else mine_vocabulary(
        texts, max_features=max_features
    )
    if not used_vocab:
        raise ValueError("could not mine any keyword features from texts")
    feature_keys = [f"kw_{_safe(tok)}" for tok in used_vocab]
    rows: list[dict[str, Any]] = []
    for it, text in zip(items, texts):
        toks = set(tokenize(text))
        row: dict[str, Any] = {}
        if "id" in it:
            row["id"] = it["id"]
        row[text_key] = text
        for tok, fk in zip(used_vocab, feature_keys):
            row[fk] = "yes" if tok in toks else "no"
        row[label_key] = it.get(label_key)
        rows.append(row)
    return rows, feature_keys, list(used_vocab)


def _safe(tok: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", tok)
    return s[:40] or "tok"


def load_text_batch_csv(
    path: str | Path,
    *,
    text_key: str = "text",
    label_key: str = "label",
) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"empty text-batch CSV: {path}")
    # accept aliases
    if text_key not in rows[0]:
        for alt in ("answer", "document", "content", "body", "raw_text"):
            if alt in rows[0]:
                text_key = alt
                break
    if label_key not in rows[0]:
        for alt in ("score", "grade", "route", "class", "y"):
            if alt in rows[0]:
                label_key = alt
                break
    if text_key not in rows[0] or label_key not in rows[0]:
        raise ValueError(
            f"need text+label columns; got {list(rows[0].keys())} "
            f"(tried text_key={text_key!r} label_key={label_key!r})"
        )
    out = []
    for r in rows:
        item = dict(r)
        item["_text_key"] = text_key
        item["_label_key"] = label_key
        out.append(item)
    return out


def load_text_batch_jsonl(
    path: str | Path,
    *,
    text_key: str = "text",
    label_key: str = "label",
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(obj)
    if not items:
        raise ValueError(f"empty JSONL: {path}")
    return items


def load_text_folder(
    folder: str | Path,
    labels_path: str | Path,
    *,
    label_key: str = "label",
) -> list[dict[str, Any]]:
    """Folder of .txt files + labels CSV/JSONL with columns id,label (id=stem)."""
    folder = Path(folder)
    labels_path = Path(labels_path)
    label_map: dict[str, Any] = {}
    if labels_path.suffix.lower() == ".jsonl":
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            label_map[str(obj.get("id"))] = obj.get(label_key, obj.get("score", obj.get("grade")))
    else:
        with open(labels_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lid = str(r.get("id") or r.get("file") or "")
                label_map[lid] = r.get(label_key) or r.get("score") or r.get("grade")
    items: list[dict[str, Any]] = []
    for p in sorted(folder.glob("*.txt")):
        stem = p.stem
        if stem not in label_map:
            continue
        items.append({"id": stem, "text": p.read_text(encoding="utf-8"), label_key: label_map[stem]})
    if not items:
        raise ValueError(f"no labeled .txt files matched under {folder}")
    return items


def ingest_text_batch(
    *,
    csv_path: str | Path | None = None,
    jsonl_path: str | Path | None = None,
    folder: str | Path | None = None,
    labels_path: str | Path | None = None,
    text_key: str = "text",
    label_key: str = "label",
    label_kind: LabelKind | str = "categorical",
    max_features: int = 12,
    n_bands: int = 5,
    goal: str | None = None,
) -> IngestResult:
    """Normalize a labeled/scored text batch into FeatureTable (keyword cues)."""
    if csv_path:
        raw = load_text_batch_csv(csv_path, text_key=text_key, label_key=label_key)
        # resolve aliased keys from loader
        text_key = raw[0].get("_text_key", text_key)
        label_key = raw[0].get("_label_key", label_key)
        items = [{**{k: v for k, v in r.items() if not k.startswith("_")}, "text": r[text_key], label_key: r[label_key]} for r in raw]
        # normalize to text/label keys used below
        norm = []
        for r in raw:
            norm.append({
                "id": r.get("id"),
                "text": r[text_key],
                "label": r[label_key],
            })
        items = norm
        use_label = "label"
    elif jsonl_path:
        raw = load_text_batch_jsonl(jsonl_path, text_key=text_key, label_key=label_key)
        items = [{"id": r.get("id"), "text": r.get(text_key, r.get("answer", "")), "label": r.get(label_key, r.get("score"))} for r in raw]
        use_label = "label"
    elif folder and labels_path:
        raw = load_text_folder(folder, labels_path, label_key=label_key)
        items = [{"id": r.get("id"), "text": r["text"], "label": r.get(label_key)} for r in raw]
        use_label = "label"
    else:
        raise ValueError("provide csv_path, jsonl_path, or folder+labels_path")

    rows, feature_keys, vocab = texts_to_feature_rows(
        items, text_key="text", label_key=use_label, max_features=max_features
    )
    rows, kind = coerce_labels_discrete(
        rows, use_label, label_kind=label_kind, n_bands=n_bands
    )
    table = FeatureTable(
        rows=rows,
        feature_keys=feature_keys,
        label_key=use_label,
        label_kind=kind,
        source="text_batch",
        meta={
            "vocab": vocab,
            "goal": goal,
            "max_features": max_features,
            "text_key": "text",
        },
    )
    return IngestResult(
        table=table,
        provenance={
            "loader": "text_batch_keywords",
            "vocab": vocab,
            "goal": goal,
            "n_items": len(items),
            "label_kind": kind,
        },
    )


__all__ = [
    "ingest_text_batch",
    "load_text_batch_csv",
    "load_text_batch_jsonl",
    "load_text_folder",
    "mine_vocabulary",
    "texts_to_feature_rows",
    "tokenize",
]
