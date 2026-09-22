"""Messy-text ingest with mocked LLM (no live key / network)."""

from __future__ import annotations

import json
from pathlib import Path

from meta_jev.data.grow_pipeline import grow_from_table
from meta_jev.data.messy_ingest import ingest_messy_text, validate_extracted_table
from meta_jev.runtime.engine import RuntimeEngine


MOCK_PAYLOAD = {
    "feature_keys": ["mentions_refund", "has_error_code", "urgency"],
    "label_key": "team",
    "label_kind": "categorical",
    "rows": [
        {"mentions_refund": "yes", "has_error_code": "no", "urgency": "low", "team": "billing"},
        {"mentions_refund": "yes", "has_error_code": "no", "urgency": "high", "team": "billing"},
        {"mentions_refund": "no", "has_error_code": "yes", "urgency": "high", "team": "engineering"},
        {"mentions_refund": "no", "has_error_code": "yes", "urgency": "mid", "team": "engineering"},
        {"mentions_refund": "no", "has_error_code": "no", "urgency": "high", "team": "trust_safety"},
        {"mentions_refund": "no", "has_error_code": "no", "urgency": "low", "team": "general"},
        {"mentions_refund": "yes", "has_error_code": "yes", "urgency": "mid", "team": "billing"},
        {"mentions_refund": "no", "has_error_code": "no", "urgency": "mid", "team": "general"},
    ],
}


def _mock_chat(messages, **kwargs):  # noqa: ANN001, ANN003
    return {
        "choices": [
            {"message": {"content": "```json\n" + json.dumps(MOCK_PAYLOAD) + "\n```"}}
        ]
    }


def test_validate_extracted_table() -> None:
    table = validate_extracted_table(MOCK_PAYLOAD)
    assert table.n_features == 3
    assert table.label_key == "team"


def test_messy_ingest_mocked_then_grow(tmp_path: Path) -> None:
    ingested = ingest_messy_text(
        "customer wants refund; api 500 ERR_TIMEOUT; nsfw report",
        "分流到哪个团队",
        chat_fn=_mock_chat,
    )
    assert ingested.table.source == "messy_llm"
    result = grow_from_table(
        ingested.table,
        out=tmp_path / "m.json",
        sop_out=tmp_path / "m.sop.json",
        min_samples=1,
        max_depth=3,
        seed=0,
    )
    row = {k: ingested.table.rows[0][k] for k in ingested.table.feature_keys}
    traced = RuntimeEngine().run_sop_traced(result["sop"], row)
    assert traced["decision"] in {"billing", "engineering", "trust_safety", "general"}


def test_messy_ingest_no_key_clear_error(monkeypatch) -> None:  # noqa: ANN001
    import meta_jev.data.messy_ingest as mi
    import os

    monkeypatch.setattr(os.environ, "get", lambda *a, **k: "" if a and a[0] == "META_JEV_LLM_API_KEY" else os.environ.get(*a, **k))

    def boom(*_a, **_k):
        raise RuntimeError(
            "No META_JEV_LLM_API_KEY in .env. "
            "Messy-text ingest needs an LLM key, or use the CSV / "
            "text-batch path: meta-jev grow --csv ... / "
            "meta-jev ingest-batch --csv ..."
        )

    # Force the no-key path by injecting failure when chat_fn is None
    # and dotenv has no key — call with chat_fn that we don't pass, mock load
    from meta_jev.llm import mimo

    monkeypatch.setattr(mimo, "load_dotenv_env", lambda *a, **k: {})
    monkeypatch.setenv("META_JEV_LLM_API_KEY", "")
    monkeypatch.delenv("META_JEV_LLM_API_KEY", raising=False)
    # ensure empty
    os.environ.pop("META_JEV_LLM_API_KEY", None)

    try:
        ingest_messy_text("notes here", "goal here")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        msg = str(exc)
        assert "CSV" in msg or "API_KEY" in msg or "Fall back" in msg or "key" in msg.lower()
