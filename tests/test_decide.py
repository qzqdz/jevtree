"""Unified decide entry with mocked LLM (no live key / network)."""

from __future__ import annotations

import json
from pathlib import Path

from jevtree.cli.main import main
from jevtree.data.decide_pipeline import run_decide
from jevtree.data.feature_table import FeatureTable, IngestResult
from jevtree.data.messy_ingest import validate_extracted_table
from jevtree.data.universal_ingest import ingest_data_with_goal, read_data_blob
from jevtree.runtime.engine import RuntimeEngine


MOCK_PAYLOAD = {
    "feature_keys": ["income_band", "credit_history", "debt_ratio"],
    "label_key": "approve",
    "label of_kind_placeholder": None,  # kept out of JSON via rebuild below
    "label_kind": "categorical",
    "rows": [
        {"income_band": "high", "credit_history": "good", "debt_ratio": "low", "approve": "yes"},
        {"income_band": "high", "credit_history": "good", "debt_ratio": "high", "approve": "yes"},
        {"income_band": "low", "credit_history": "bad", "debt_ratio": "high", "approve": "no"},
        {"income_band": "low", "credit_history": "bad", "debt_ratio": "low", "approve": "no"},
        {"income_band": "mid", "credit_history": "good", "debt_ratio": "mid", "approve": "yes"},
        {"income_band": "mid", "credit_history": "bad", "debt_ratio": "mid", "approve": "no"},
        {"income_band": "high", "credit_history": "bad", "debt_ratio": "low", "approve": "yes"},
        {"income_band": "low", "credit_history": "good", "debt_ratio": "high", "approve": "no"},
    ],
}
# drop accidental key if present
MOCK_PAYLOAD.pop("label of_kind_placeholder", None)


def _mock_chat(messages, **kwargs):  # noqa: ANN001, ANN003
    return {
        "choices": [
            {"message": {"content": "```json\n" + json.dumps(MOCK_PAYLOAD) + "\n```"}}
        ]
    }


def _mock_table() -> FeatureTable:
    t = validate_extracted_table(MOCK_PAYLOAD)
    return FeatureTable(
        rows=t.rows,
        feature_keys=t.feature_keys,
        label_key=t.label_key,
        label_kind=t.label_kind,
        source="decide_llm",
    )


def test_read_data_blob_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("a,b,y\n1,2,yes\n3,4,no\n", encoding="utf-8")
    blob = read_data_blob(csv_path)
    assert blob["kind"] == "csv"
    assert "a,b,y" in blob["preview"]


def test_ingest_data_with_goal_mocked(tmp_path: Path) -> None:
    csv_path = tmp_path / "loan.csv"
    csv_path.write_text(
        "income_band,credit_history,debt_ratio,approve\nhigh,good,low,yes\n",
        encoding="utf-8",
    )
    ingested = ingest_data_with_goal(
        csv_path,
        "按是否批准贷款做决策树",
        chat_fn=_mock_chat,
    )
    assert ingested.table.source == "decide_llm"
    assert ingested.table.label_key == "approve"
    assert ingested.table.n_rows == 8


def test_run_decide_mocked_writes_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "loan.csv"
    csv_path.write_text(
        "income_band,credit_history,debt_ratio,approve\n"
        "high,good,low,yes\nlow,bad,high,no\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    result = run_decide(
        csv_path,
        "按是否批准贷款做决策树",
        out_dir=out,
        chat_fn=_mock_chat,
        max_depth=3,
        seed=0,
        trace_examples=2,
    )
    assert (out / "feature_table.json").is_file()
    assert (out / "feature_table.csv").is_file()
    assert (out / "tree.json").is_file()
    assert (out / "sop.json").is_file()
    assert (out / "story.md").is_file()
    assert (out / "traces.json").is_file()
    assert len(result["traces"]) == 2
    row = {k: result["table"].rows[0][k] for k in result["table"].feature_keys}
    traced = RuntimeEngine().run_sop_traced(result["sop"], row)
    assert traced["decision"] in {"yes", "no"}


def test_cli_decide_with_mocked_llm(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import jevtree.data.decide_pipeline as dp

    csv_path = tmp_path / "loan.csv"
    csv_path.write_text(
        "income_band,credit_history,debt_ratio,approve\nhigh,good,low,yes\n",
        encoding="utf-8",
    )
    out = tmp_path / "cli_out"
    table = _mock_table()

    def fake_ingest(data_path, goal, **kwargs):  # noqa: ANN001, ANN003
        return IngestResult(
            table=table,
            provenance={"loader": "decide_llm", "goal": goal},
        )

    monkeypatch.setattr(dp, "ingest_data_with_goal", fake_ingest)
    rc = main(
        [
            "decide",
            "--data",
            str(csv_path),
            "--goal",
            "按是否批准贷款做决策树",
            "--out",
            str(out),
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    assert (out / "tree.json").is_file()
    assert (out / "sop.json").is_file()


def test_cli_from_data_alias(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import jevtree.data.decide_pipeline as dp

    csv_path = tmp_path / "x.csv"
    csv_path.write_text("a,y\n1,yes\n", encoding="utf-8")
    table = _mock_table()
    monkeypatch.setattr(
        dp,
        "ingest_data_with_goal",
        lambda *a, **k: IngestResult(table=table, provenance={}),
    )
    out = tmp_path / "alias_out"
    rc = main(
        [
            "from-data",
            "--data",
            str(csv_path),
            "--goal",
            "approve loan",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert (out / "feature_table.csv").is_file()


def test_invalid_llm_json_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("a,y\n1,yes\n", encoding="utf-8")

    def bad_chat(messages, **kwargs):  # noqa: ANN001, ANN003
        return {"choices": [{"message": {"content": "not json at all"}}]}

    try:
        ingest_data_with_goal(csv_path, "goal", chat_fn=bad_chat)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "invalid" in str(exc).lower() or "JSON" in str(exc)
