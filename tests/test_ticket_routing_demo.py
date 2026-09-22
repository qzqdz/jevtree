"""Ticket-routing five-minute demo: grow + traced run is deterministic."""

from __future__ import annotations

import json
from pathlib import Path

from meta_jev.core.tree import IGDecisionTreeGrower
from meta_jev.data.ticket_routing import (
    DEMO_OBSERVATIONS,
    generate_ticket_routing,
    load_ticket_routing_split,
    write_ticket_routing_csv,
)
from meta_jev.runtime.engine import RuntimeEngine


SEED = 42


def test_ticket_routing_known_obs_routes() -> None:
    split = load_ticket_routing_split(n_samples=120, seed=SEED, noise_rate=0.05)
    rows = split["train"] + split["test"]
    grower = IGDecisionTreeGrower(min_samples=2, continuous_keys=[])
    tree = grower.fit(
        rows,
        split["label_key"],
        split["feature_keys"],
        criterion="gain",
        max_depth=4,
    )
    engine = RuntimeEngine()
    sop = grower.export_sop(tree)
    for case in DEMO_OBSERVATIONS:
        traced = engine.run_sop_traced(sop, case["obs"])
        assert traced["decision"] == case["expected_route"], case["id"]
        assert isinstance(traced["path"], list)
        assert traced["questions_used"] == len(traced["path"])
        # Same path via grower API
        pred, path = grower.predict_path(case["obs"])
        assert pred == case["expected_route"]
        assert [s["feature"] for s in path] == [s["feature"] for s in traced["path"]]


def test_ticket_routing_csv_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "ticket_routing.csv"
    write_ticket_routing_csv(csv_path, n_samples=80, seed=SEED)
    rows = generate_ticket_routing(n_samples=80, seed=SEED)
    assert csv_path.is_file()
    split = load_ticket_routing_split(csv_path=csv_path, seed=SEED)
    assert split["n_samples"] == 80
    assert set(split["feature_keys"])  # non-empty
    # grow from CSV-backed split still routes demo obs
    grower = IGDecisionTreeGrower(min_samples=2, continuous_keys=[])
    all_rows = split["train"] + split["test"]
    tree = grower.fit(all_rows, split["label_key"], split["feature_keys"], max_depth=4)
    sop = grower.export_sop(tree)
    case = DEMO_OBSERVATIONS[1]  # api_500 -> engineering
    assert RuntimeEngine().run_sop(sop, case["obs"]) == case["expected_route"]
    # SOP JSON roundtrip preserves traced decision
    payload = json.loads(json.dumps(sop.to_dict()))
    traced = RuntimeEngine().run_sop_traced(payload, case["obs"])
    assert traced["decision"] == "engineering"
