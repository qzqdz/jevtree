"""Synthetic support-ticket routing dataset for the five-minute story demo.

Labels are routing destinations under a question-budget metaphor:
ask fewer features -> clear, auditable route.
Not an AFABench / Acc@budget scoring dataset.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from meta_jev.data.cube import dataset_hash

DATASET_ID = "ticket_routing"
LABEL_KEY = "route"
FEATURE_KEYS = [
    "product_area",
    "urgency_keywords",
    "account_tier",
    "error_code_present",
    "prior_escalation",
    "refund_mentioned",
]
ROUTES = ("billing", "engineering", "trust_safety", "L1_general")

PRODUCT_AREAS = ("billing", "api", "account", "content", "general")
YES_NO = ("yes", "no")
TIERS = ("free", "pro", "enterprise")

FEATURE_LABELS_ZH = {
    "product_area": "产品线",
    "urgency_keywords": "含紧急词",
    "account_tier": "账户等级",
    "error_code_present": "有错误码",
    "prior_escalation": "曾升级",
    "refund_mentioned": "提退款",
}
VALUE_LABELS_ZH = {
    "billing": "计费",
    "api": "API",
    "account": "账号",
    "content": "内容",
    "general": "综合",
    "yes": "是",
    "no": "否",
    "free": "免费",
    "pro": "专业版",
    "enterprise": "企业版",
    "engineering": "工程",
    "trust_safety": "信任安全",
    "L1_general": "一线综合",
}
ROUTE_LABELS_ZH = {
    "billing": "计费",
    "engineering": "工程",
    "trust_safety": "信任安全",
    "L1_general": "一线综合",
}


def _route_from_features(row: dict[str, Any]) -> str:
    """Deterministic routing rule (high-signal for IG demo trees)."""
    if row["refund_mentioned"] == "yes" or (
        row["product_area"] == "billing" and row["error_code_present"] == "no"
    ):
        return "billing"
    if row["error_code_present"] == "yes" or row["product_area"] == "api":
        return "engineering"
    if row["product_area"] == "content" and row["urgency_keywords"] == "yes":
        return "trust_safety"
    if row["prior_escalation"] == "yes" and row["account_tier"] == "enterprise":
        return "engineering"
    return "L1_general"


def generate_ticket_routing(
    *,
    n_samples: int = 120,
    seed: int = 42,
    noise_rate: float = 0.05,
) -> list[dict[str, Any]]:
    """Generate tiny categorical ticket rows with a fixed seed."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    prototypes = [
        {
            "product_area": "billing",
            "urgency_keywords": "no",
            "account_tier": "pro",
            "error_code_present": "no",
            "prior_escalation": "no",
            "refund_mentioned": "yes",
        },
        {
            "product_area": "api",
            "urgency_keywords": "yes",
            "account_tier": "enterprise",
            "error_code_present": "yes",
            "prior_escalation": "no",
            "refund_mentioned": "no",
        },
        {
            "product_area": "content",
            "urgency_keywords": "yes",
            "account_tier": "free",
            "error_code_present": "no",
            "prior_escalation": "no",
            "refund_mentioned": "no",
        },
        {
            "product_area": "general",
            "urgency_keywords": "no",
            "account_tier": "free",
            "error_code_present": "no",
            "prior_escalation": "no",
            "refund_mentioned": "no",
        },
        {
            "product_area": "account",
            "urgency_keywords": "no",
            "account_tier": "enterprise",
            "error_code_present": "no",
            "prior_escalation": "yes",
            "refund_mentioned": "no",
        },
    ]
    for proto in prototypes:
        row = dict(proto)
        row[LABEL_KEY] = _route_from_features(row)
        rows.append(row)

    while len(rows) < n_samples:
        row = {
            "product_area": rng.choice(PRODUCT_AREAS),
            "urgency_keywords": rng.choice(YES_NO),
            "account_tier": rng.choice(TIERS),
            "error_code_present": rng.choice(YES_NO),
            "prior_escalation": rng.choice(YES_NO),
            "refund_mentioned": rng.choice(YES_NO),
        }
        route = _route_from_features(row)
        if noise_rate > 0 and rng.random() < noise_rate:
            others = [r for r in ROUTES if r != route] or list(ROUTES)
            route = rng.choice(others)
        row[LABEL_KEY] = route
        rows.append(row)

    rng.shuffle(rows)
    return rows[:n_samples]


def write_ticket_routing_csv(
    path: str | Path,
    *,
    n_samples: int = 120,
    seed: int = 42,
    noise_rate: float = 0.05,
) -> Path:
    """Write CSV under examples/data/ (or any path)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_ticket_routing(n_samples=n_samples, seed=seed, noise_rate=noise_rate)
    fieldnames = list(FEATURE_KEYS) + [LABEL_KEY]
    with open(dest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return dest


def load_ticket_routing_split(
    *,
    n_samples: int = 120,
    seed: int = 42,
    train_ratio: float = 0.75,
    noise_rate: float = 0.05,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train/test split for grow / demo / tests.

    If *csv_path* is given and exists, load that CSV instead of regenerating.
    """
    if csv_path is not None and Path(csv_path).is_file():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    else:
        rows = generate_ticket_routing(
            n_samples=n_samples, seed=seed, noise_rate=noise_rate
        )
    feature_keys = [k for k in FEATURE_KEYS if k in rows[0]]
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
        "n_samples": len(rows),
        "n_features": len(feature_keys),
        "routes": list(ROUTES),
    }


DEMO_OBSERVATIONS: list[dict[str, Any]] = [
    {
        "id": "invoice_refund",
        "story_zh": "企业客户说发票金额不对，要求退款重开",
        "story_en": "Enterprise customer: wrong invoice amount, wants refund",
        "obs": {
            "product_area": "billing",
            "urgency_keywords": "no",
            "account_tier": "enterprise",
            "error_code_present": "no",
            "prior_escalation": "no",
            "refund_mentioned": "yes",
        },
        "expected_route": "billing",
    },
    {
        "id": "api_500",
        "story_zh": "企业 API 调用持续 500，日志里有错误码",
        "story_en": "Enterprise API keep returning 500 with an error code",
        "obs": {
            "product_area": "api",
            "urgency_keywords": "yes",
            "account_tier": "enterprise",
            "error_code_present": "yes",
            "prior_escalation": "no",
            "refund_mentioned": "no",
        },
        "expected_route": "engineering",
    },
    {
        "id": "ugc_report",
        "story_zh": "用户紧急举报不当内容",
        "story_en": "Urgent user report of unsafe UGC",
        "obs": {
            "product_area": "content",
            "urgency_keywords": "yes",
            "account_tier": "free",
            "error_code_present": "no",
            "prior_escalation": "no",
            "refund_mentioned": "no",
        },
        "expected_route": "trust_safety",
    },
]


__all__ = [
    "DATASET_ID",
    "DEMO_OBSERVATIONS",
    "FEATURE_KEYS",
    "FEATURE_LABELS_ZH",
    "LABEL_KEY",
    "ROUTE_LABELS_ZH",
    "ROUTES",
    "VALUE_LABELS_ZH",
    "generate_ticket_routing",
    "load_ticket_routing_split",
    "write_ticket_routing_csv",
]
