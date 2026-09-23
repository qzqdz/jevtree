"""Shared grow pipeline: FeatureTable → tree JSON + SOP + story."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jevtree.core.tree import IGDecisionTreeGrower
from jevtree.data.feature_table import FeatureTable
from jevtree.data.story import format_tree_story


def grow_from_table(
    table: FeatureTable,
    *,
    out: str | Path | None = None,
    sop_out: str | Path | None = None,
    criterion: str = "gain",
    max_depth: int | None = None,
    min_samples: int = 1,
    continuous_keys: list[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit IG tree on *table*; optionally write artifacts. Returns summary dict."""
    grower = IGDecisionTreeGrower(
        min_samples=min_samples,
        continuous_keys=continuous_keys or [],
        bin_seed=seed,
    )
    tree = grower.fit(
        table.rows,
        table.label_key,
        table.feature_keys,
        criterion=criterion,
        max_depth=max_depth,
    )
    story_zh = format_tree_story(tree, label_key=table.label_key, lang="zh")
    story_en = format_tree_story(tree, label_key=table.label_key, lang="en")
    result: dict[str, Any] = {
        "grower": grower,
        "tree": tree,
        "story_zh": story_zh,
        "story_en": story_en,
        "n_rows": table.n_rows,
        "n_features": table.n_features,
        "label_key": table.label_key,
        "label_kind": table.label_kind,
        "source": table.source,
    }
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        grower.save(str(out_path))
        result["out"] = str(out_path)
    if sop_out:
        sop = grower.export_sop(tree)
        sop_path = Path(sop_out)
        sop_path.parent.mkdir(parents=True, exist_ok=True)
        sop_path.write_text(
            json.dumps(sop.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result["sop_out"] = str(sop_path)
        result["sop"] = sop
    return result


__all__ = ["grow_from_table"]
