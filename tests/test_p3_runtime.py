from __future__ import annotations

import json
from pathlib import Path

from jevtree.core.sop import DecisionSOP
from jevtree.core.tree import IGDecisionTreeGrower
from jevtree.data.cube import load_cube_split
from jevtree.runtime import RuntimeEngine
from jevtree.sdk import decide


def test_tree_export_sop_roundtrip_and_sdk() -> None:
    split = load_cube_split(n_samples=128, n_features=5, seed=0)
    grower = IGDecisionTreeGrower(continuous_keys=[])
    tree = grower.fit(split["train"], split["label_key"], split["feature_keys"])
    sop = grower.export_sop(tree)
    assert sop.validate() == []

    row = split["test"][0]
    expected = grower.predict(row)
    engine = RuntimeEngine()
    assert engine.run_sop(sop, row) == expected
    assert decide(row, sop) == expected

    restored = DecisionSOP.from_dict(sop.to_dict())
    assert engine.run_sop(restored, row) == expected


def test_tree_and_sop_files_can_be_reloaded(tmp_path: Path) -> None:
    split = load_cube_split(n_samples=96, n_features=5, seed=1)
    grower = IGDecisionTreeGrower(continuous_keys=[])
    tree = grower.fit(split["train"], split["label_key"], split["feature_keys"])
    tree_path = tmp_path / "tree.json"
    sop_path = tmp_path / "tree.sop.json"
    grower.save(str(tree_path))
    sop_path.write_text(json.dumps(grower.export_sop(tree).to_dict()), encoding="utf-8")

    restored = IGDecisionTreeGrower.load(str(tree_path))
    row = split["test"][0]
    assert restored.predict(row) == grower.predict(row)
    payload = json.loads(sop_path.read_text(encoding="utf-8"))
    assert RuntimeEngine().run_sop(payload, row) == grower.predict(row)
