"""Universal LLM-first ingest: any data path + NL goal → FeatureTable.

Single product door for jevtree. Schema is validated in code; invalid LLM
output raises a clear error (no silent fake rows).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jevtree.data.feature_table import FeatureTable, IngestResult
from jevtree.data.messy_ingest import (
    _extract_json_object,
    validate_extracted_table,
)

# Caps keep prompts bounded for local demos / flash models.
_MAX_CHARS = 48_000
_MAX_DIR_FILES = 40
_MAX_CSV_ROWS_PREVIEW = 80
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".csv", ".json", ".jsonl", ".tsv"}


def _truncate(text: str, limit: int = _MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 80] + f"\n\n…[truncated {len(text) - limit + 80} chars]…"


def read_data_blob(path: str | Path) -> dict[str, Any]:
    """Load *path* (file or directory) into a preview blob for the LLM."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"data path not found: {p}")

    if p.is_dir():
        parts: list[str] = []
        files = sorted(
            f
            for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in _TEXT_SUFFIXES
        )[:_MAX_DIR_FILES]
        if not files:
            # fall back: any small text-ish files
            files = sorted(f for f in p.iterdir() if f.is_file())[:_MAX_DIR_FILES]
        if not files:
            raise ValueError(f"directory has no readable files: {p}")
        for f in files:
            try:
                body = f.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ValueError(f"cannot read {f}: {exc}") from exc
            parts.append(f"--- file: {f.relative_to(p)} ---\n{body}")
        text = "\n\n".join(parts)
        kind = "directory"
        meta = {"n_files": len(files), "files": [str(f.relative_to(p)) for f in files]}
    else:
        suffix = p.suffix.lower()
        raw = p.read_text(encoding="utf-8", errors="replace")
        if suffix == ".csv" or suffix == ".tsv":
            lines = raw.splitlines()
            head = lines[: _MAX_CSV_ROWS_PREVIEW + 1]
            text = "\n".join(head)
            if len(lines) > len(head):
                text += f"\n…[{len(lines) - len(head)} more rows omitted]…"
            kind = "csv"
            meta = {"n_lines": len(lines), "delimiter": "\t" if suffix == ".tsv" else ","}
        elif suffix == ".jsonl":
            rows_preview: list[str] = []
            for i, line in enumerate(raw.splitlines()):
                if i >= _MAX_CSV_ROWS_PREVIEW:
                    rows_preview.append(f"…[{len(raw.splitlines()) - i} more lines]…")
                    break
                rows_preview.append(line)
            text = "\n".join(rows_preview)
            kind = "jsonl"
            meta = {"n_lines": len(raw.splitlines())}
        elif suffix == ".json":
            text = raw
            kind = "json"
            meta = {}
        else:
            text = raw
            kind = "text"
            meta = {"suffix": suffix or "(none)"}

    text = _truncate(text)
    if not text.strip():
        raise ValueError(f"data is empty: {p}")
    return {
        "path": str(p),
        "kind": kind,
        "preview": text,
        "n_chars": len(text),
        "meta": meta,
    }


def build_decide_prompt(goal: str, blob: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You normalize arbitrary user data into a small discrete feature table "
        "for decision-tree growth. Return ONLY JSON with keys: "
        "feature_keys (list of short snake_case names), "
        "label_key (string), "
        "label_kind (categorical|ordinal|numeric_binned), "
        "rows (list of objects with those features + label). "
        "Use short categorical values (yes/no, low/mid/high, or small enums). "
        "Respect the user's natural-language goal when choosing the label and features. "
        "Invent only features justified by the data. Prefer 8-60 rows when possible. "
        "If the input is already a labeled table, map columns to feature_keys/label_key "
        "and keep row values (discretize continuous columns into short bins if needed). "
        "No markdown commentary outside a JSON object."
    )
    user = (
        f"Goal / requirement (natural language):\n{goal}\n\n"
        f"Data kind: {blob.get('kind')}\n"
        f"Data path: {blob.get('path')}\n\n"
        f"Data preview:\n{blob.get('preview')}\n\n"
        "Produce feature_keys, label_key, label_kind, and labeled rows as JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def ingest_data_with_goal(
    data_path: str | Path,
    goal: str,
    *,
    chat_fn: Any | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    blob: dict[str, Any] | None = None,
) -> IngestResult:
    """LLM-normalize *data_path* + *goal* into a validated FeatureTable."""
    if not (goal or "").strip():
        raise ValueError("goal is empty — pass a natural-language requirement via --goal")

    blob = blob or read_data_blob(data_path)

    if chat_fn is None:
        try:
            from jevtree.llm.mimo import chat_completion, load_dotenv_env
            import os

            load_dotenv_env()
            if not os.environ.get("JEVTREE_LLM_API_KEY"):
                raise RuntimeError(
                    "No JEVTREE_LLM_API_KEY in .env. "
                    "The unified entry `jevtree decide` needs an LLM key to "
                    "normalize data + goal into a FeatureTable."
                )
            chat_fn = chat_completion
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"LLM helper unavailable ({exc}). "
                "Set JEVTREE_LLM_* in .env for `jevtree decide`."
            ) from exc

    messages = build_decide_prompt(goal, blob)
    try:
        resp = chat_fn(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError("LLM returned no choices")
    content = str((choices[0].get("message") or {}).get("content") or "")
    try:
        payload = _extract_json_object(content)
        table = validate_extracted_table(payload)
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"LLM FeatureTable JSON invalid ({exc}). "
            "Refusing to invent rows — fix the goal/data or retry."
        ) from exc

    # Mark universal source (validate_extracted_table sets messy_llm)
    table = FeatureTable(
        rows=table.rows,
        feature_keys=table.feature_keys,
        label_key=table.label_key,
        label_kind=table.label_kind,
        source="decide_llm",
        meta={
            **dict(table.meta),
            "data_kind": blob.get("kind"),
            "data_path": blob.get("path"),
            "goal": goal,
        },
    )
    return IngestResult(
        table=table,
        provenance={
            "loader": "decide_llm",
            "goal": goal,
            "data_kind": blob.get("kind"),
            "data_path": blob.get("path"),
            "n_chars": blob.get("n_chars"),
            "blob_meta": blob.get("meta") or {},
        },
    )


__all__ = [
    "build_decide_prompt",
    "ingest_data_with_goal",
    "read_data_blob",
]
