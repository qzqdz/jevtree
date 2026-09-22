"""Batch text classification + homework scoring → grow (offline keywords)."""

from __future__ import annotations

from pathlib import Path

from meta_jev.data.grow_pipeline import grow_from_table
from meta_jev.data.text_batch import ingest_text_batch, mine_vocabulary, texts_to_feature_rows
from meta_jev.runtime.engine import RuntimeEngine

REPO = Path(__file__).resolve().parents[1]
EMAILS = REPO / "examples/data/batch_texts/support_emails.csv"
HOMEWORK = REPO / "examples/data/homework_scoring/short_answers.csv"


def test_mine_vocabulary_deterministic() -> None:
    texts = ["refund invoice billing", "refund charge invoice", "api error timeout"]
    v1 = mine_vocabulary(texts, max_features=5, min_df=1)
    v2 = mine_vocabulary(texts, max_features=5, min_df=1)
    assert v1 == v2
    assert len(v1) >= 3


def test_support_emails_batch_grow(tmp_path: Path) -> None:
    ingested = ingest_text_batch(csv_path=EMAILS, label_key="label", max_features=10)
    table = ingested.table
    assert table.source == "text_batch"
    assert all(k.startswith("kw_") for k in table.feature_keys)
    result = grow_from_table(
        table, out=tmp_path / "t.json", sop_out=tmp_path / "s.json",
        min_samples=2, max_depth=4, seed=42,
    )
    row = {k: table.rows[0][k] for k in table.feature_keys}
    traced = RuntimeEngine().run_sop_traced(result["sop"], row)
    assert traced["decision"] in {"billing", "engineering", "trust_safety", "general"}
    assert "先问" in result["story_zh"] or "直接决定" in result["story_zh"]


def test_homework_scoring_ordinal(tmp_path: Path) -> None:
    ingested = ingest_text_batch(
        csv_path=HOMEWORK,
        label_key="score",
        label_kind="ordinal",
        max_features=12,
        goal="作业打分 1-5",
    )
    table = ingested.table
    assert table.label_kind == "ordinal"
    # scores stringified as discrete classes
    labels = {r["label"] for r in table.rows}
    assert labels <= {"1", "2", "3", "4", "5"}
    result = grow_from_table(
        table, out=tmp_path / "hw.json", sop_out=tmp_path / "hw.sop.json",
        min_samples=2, max_depth=4, seed=7,
    )
    row = {k: table.rows[0][k] for k in table.feature_keys}
    traced = RuntimeEngine().run_sop_traced(result["sop"], row)
    assert traced["decision"] in labels
