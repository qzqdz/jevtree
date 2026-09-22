"""CSV grow + trace is deterministic offline (universal spine)."""

from __future__ import annotations

import json
from pathlib import Path

from meta_jev.data.csv_ingest import ingest_csv
from meta_jev.data.grow_pipeline import grow_from_table
from meta_jev.runtime.engine import RuntimeEngine

REPO = Path(__file__).resolve().parents[1]
LOAN = REPO / "examples/data/bring_your_csv/loan_approve.csv"
COURSE = REPO / "examples/data/bring_your_csv/course_pass.csv"


def test_loan_csv_grow_deterministic(tmp_path: Path) -> None:
    assert LOAN.is_file()
    table = ingest_csv(LOAN, label="approve").table
    r1 = grow_from_table(
        table, out=tmp_path / "t1.json", sop_out=tmp_path / "s1.json",
        min_samples=2, max_depth=4, seed=42,
    )
    r2 = grow_from_table(
        table, out=tmp_path / "t2.json", sop_out=tmp_path / "s2.json",
        min_samples=2, max_depth=4, seed=42,
    )
    assert r1["story_zh"] == r2["story_zh"]
    assert "先问" in r1["story_zh"] or "直接决定" in r1["story_zh"]
    row = {k: table.rows[0][k] for k in table.feature_keys}
    t1 = RuntimeEngine().run_sop_traced(r1["sop"], row)
    t2 = RuntimeEngine().run_sop_traced(r2["sop"], row)
    assert t1["decision"] == t2["decision"]
    assert t1["path"] == t2["path"]
    assert isinstance(t1["questions_used"], int)


def test_course_pass_csv_roundtrip(tmp_path: Path) -> None:
    table = ingest_csv(COURSE, label="pass").table
    result = grow_from_table(
        table, out=tmp_path / "tree.json", sop_out=tmp_path / "sop.json",
        min_samples=2, max_depth=4, seed=0,
    )
    payload = json.loads((tmp_path / "sop.json").read_text(encoding="utf-8"))
    row = {k: table.rows[3][k] for k in table.feature_keys}
    assert RuntimeEngine().run_sop(payload, row) == result["grower"].predict(row)
