"""End-to-end decide: data + NL goal → LLM FeatureTable → grow → artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meta_jev.data.feature_table import FeatureTable, IngestResult
from meta_jev.data.grow_pipeline import grow_from_table
from meta_jev.data.universal_ingest import ingest_data_with_goal


def default_out_dir(repo_root: Path | None = None) -> Path:
    base = (repo_root or Path.cwd()) / "results"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base / f"decide_{ts}"


def write_story_md(path: Path, *, story_zh: str, story_en: str, goal: str, table: FeatureTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f"# Meta-Jev decide story\n\n"
        f"**Goal:** {goal}\n\n"
        f"**Source:** `{table.source}` · rows={table.n_rows} · "
        f"features={table.n_features} · label=`{table.label_key}` "
        f"({table.label_kind})\n\n"
        f"## 中文\n\n{story_zh}\n\n"
        f"## English\n\n{story_en}\n"
    )
    path.write_text(body, encoding="utf-8")


def run_decide(
    data_path: str | Path,
    goal: str,
    *,
    out_dir: str | Path | None = None,
    chat_fn: Any | None = None,
    criterion: str = "gain",
    max_depth: int | None = 4,
    min_samples: int = 1,
    seed: int = 0,
    trace_examples: int = 0,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """LLM ingest + grow; write feature_table / tree / sop / story under *out_dir*."""
    ingested: IngestResult = ingest_data_with_goal(
        data_path, goal, chat_fn=chat_fn
    )
    table = ingested.table
    dest = Path(out_dir) if out_dir else default_out_dir(repo_root)
    dest.mkdir(parents=True, exist_ok=True)

    table_json = dest / "feature_table.json"
    table_csv = dest / "feature_table.csv"
    tree_path = dest / "tree.json"
    sop_path = dest / "sop.json"
    story_path = dest / "story.md"

    table_json.write_text(
        json.dumps(table.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    table.write_csv(table_csv)

    result = grow_from_table(
        table,
        out=tree_path,
        sop_out=sop_path,
        criterion=criterion,
        max_depth=max_depth,
        min_samples=min_samples,
        seed=seed,
    )
    write_story_md(
        story_path,
        story_zh=result["story_zh"],
        story_en=result["story_en"],
        goal=goal,
        table=table,
    )

    traces: list[dict[str, Any]] = []
    n_trace = max(0, int(trace_examples or 0))
    # Also auto-trace a couple if the goal mentions 示例/trace/演示
    goal_l = goal.lower()
    if n_trace == 0 and any(
        k in goal_l for k in ("trace", "示例", "演示", "举例", "example")
    ):
        n_trace = 2
    if n_trace > 0 and result.get("sop") is not None:
        from meta_jev.runtime.engine import RuntimeEngine

        engine = RuntimeEngine()
        for row in table.rows[:n_trace]:
            obs = {k: row[k] for k in table.feature_keys if k in row}
            traced = engine.run_sop_traced(result["sop"], obs)
            traces.append({"obs": obs, "trace": traced})
        (dest / "traces.json").write_text(
            json.dumps(traces, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    provenance_path = dest / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "goal": goal,
                "data_path": str(data_path),
                "ingest": ingested.provenance,
                "n_rows": table.n_rows,
                "n_features": table.n_features,
                "label_key": table.label_key,
                "label_kind": table.label_kind,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        **result,
        "out_dir": str(dest),
        "feature_table_json": str(table_json),
        "feature_table_csv": str(table_csv),
        "tree_out": str(tree_path),
        "sop_out": str(sop_path),
        "story_out": str(story_path),
        "traces": traces,
        "table": table,
        "ingest": ingested,
    }


__all__ = ["default_out_dir", "run_decide", "write_story_md"]
