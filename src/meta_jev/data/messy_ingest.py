"""Messy free-form paste + goal → FeatureTable via LLM, then same grow path.

If no API key / LLM fails: raise a clear error (do not silently fake rows).
"""

from __future__ import annotations

import json
import re
from typing import Any

from meta_jev.data.feature_table import FeatureTable, IngestResult, coerce_labels_discrete


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    # find first { ... } span
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def build_messy_prompt(goal: str, notes: str) -> list[dict[str, str]]:
    system = (
        "You extract a small discrete feature table for decision-tree growth. "
        "Return ONLY JSON with keys: "
        "feature_keys (list of short snake_case names), "
        "label_key (string), "
        "label_kind (categorical|ordinal|numeric_binned), "
        "rows (list of objects with those features + label). "
        "Use short categorical values (yes/no, low/mid/high, or small enums). "
        "Invent only features justified by the notes. 6-40 rows if possible. "
        "No markdown commentary."
    )
    user = (
        f"Goal / requirement:\n{goal}\n\n"
        f"Messy notes / emails / logs:\n{notes}\n\n"
        "Extract feature_keys, label_key, label_kind, and labeled rows as JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_extracted_table(payload: dict[str, Any]) -> FeatureTable:
    feature_keys = list(payload.get("feature_keys") or [])
    label_key = str(payload.get("label_key") or "").strip()
    rows = list(payload.get("rows") or [])
    if not feature_keys:
        raise ValueError("extracted JSON missing feature_keys")
    if not label_key:
        raise ValueError("extracted JSON missing label_key")
    if not rows:
        raise ValueError("extracted JSON has no rows")
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"row {i} is not an object")
        for fk in feature_keys:
            if fk not in r:
                raise ValueError(f"row {i} missing feature {fk!r}")
            r[fk] = str(r[fk]).strip()
        if label_key not in r:
            raise ValueError(f"row {i} missing label {label_key!r}")
    kind = payload.get("label_kind") or "categorical"
    rows, kind = coerce_labels_discrete(rows, label_key, label_kind=kind)
    return FeatureTable(
        rows=rows,
        feature_keys=feature_keys,
        label_key=label_key,
        label_kind=kind,  # type: ignore[arg-type]
        source="messy_llm",
        meta={"n_rows_extracted": len(rows)},
    )


def ingest_messy_text(
    notes: str,
    goal: str,
    *,
    chat_fn: Any | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> IngestResult:
    """Call LLM (or injected *chat_fn*) to structure messy notes into FeatureTable.

    *chat_fn(messages, **kw) -> dict* should match meta_jev.llm.mimo.chat_completion.
    """
    if not (notes or "").strip():
        raise ValueError("notes are empty")
    if not (goal or "").strip():
        raise ValueError("goal is empty")

    if chat_fn is None:
        try:
            from meta_jev.llm.mimo import chat_completion, load_dotenv_env
            import os

            load_dotenv_env()
            if not os.environ.get("META_JEV_LLM_API_KEY"):
                raise RuntimeError(
                    "No META_JEV_LLM_API_KEY in .env. "
                    "Use `meta-jev decide --data ... --goal ...` with META_JEV_LLM_* set."
                )
            chat_fn = chat_completion
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"LLM helper unavailable ({exc}). "
                "Set META_JEV_LLM_* for `meta-jev decide`."
            ) from exc

    messages = build_messy_prompt(goal, notes)
    try:
        resp = chat_fn(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"LLM call failed: {exc}."
        ) from exc

    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError("LLM returned no choices; use CSV path instead.")
    content = str((choices[0].get("message") or {}).get("content") or "")
    try:
        payload = _extract_json_object(content)
        table = validate_extracted_table(payload)
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"LLM JSON invalid ({exc}). Refusing to invent rows."
        ) from exc

    return IngestResult(
        table=table,
        provenance={"loader": "messy_llm", "goal": goal, "n_chars": len(notes)},
    )


__all__ = [
    "build_messy_prompt",
    "ingest_messy_text",
    "validate_extracted_table",
]
