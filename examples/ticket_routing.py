#!/usr/bin/env python3
"""Five-minute classmate demo: ticket routing grow → SOP → run → auditable trace.

Story: under a question budget, ask fewer features and still get a clear,
auditable routing decision (billing / engineering / trust_safety / L1_general).

This is a *demo SOP*, not a production classifier scoreboard.
Do not report Acc@budget here — scoring stays in `meta-jev eval-afa`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python examples/ticket_routing.py` without install.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from meta_jev.core.tree import IGDecisionTreeGrower
from meta_jev.data.ticket_routing import (
    DEMO_OBSERVATIONS,
    FEATURE_LABELS_ZH,
    ROUTE_LABELS_ZH,
    VALUE_LABELS_ZH,
    load_ticket_routing_split,
    write_ticket_routing_csv,
)
from meta_jev.runtime.engine import RuntimeEngine


def _repo_root() -> Path:
    return _REPO


def _fmt_value(v: object) -> str:
    s = str(v)
    zh = VALUE_LABELS_ZH.get(s)
    return f"{s} ({zh})" if zh else s


def _fmt_feature(name: str) -> str:
    zh = FEATURE_LABELS_ZH.get(name)
    return f"{name} / {zh}" if zh else name


def _fmt_route(route: object) -> str:
    s = str(route)
    zh = ROUTE_LABELS_ZH.get(s)
    return f"{s} ({zh})" if zh else s


def print_trace(case: dict, result: dict) -> None:
    print()
    print("=" * 60)
    print(f"Case: {case['id']}")
    print(f"  ZH: {case['story_zh']}")
    print(f"  EN: {case['story_en']}")
    print("-" * 60)
    path = result.get("path") or []
    for i, step in enumerate(path, start=1):
        feat = step.get("feature")
        val = step.get("value")
        print(f"  Q{i}: {_fmt_feature(str(feat))} = {_fmt_value(val)}")
    decision = result.get("decision")
    used = result.get("questions_used", len(path))
    print(f"  Decision / 决策: {_fmt_route(decision)}")
    print(f"  Questions used / 已问: {used}  (demo question budget metaphor)")
    print("=" * 60)


def run_demo(
    *,
    out_dir: Path,
    seed: int = 42,
    max_depth: int | None = 4,
    write_csv: bool = True,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = _repo_root() / "examples" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "ticket_routing.csv"
    if write_csv:
        write_ticket_routing_csv(csv_path, n_samples=120, seed=seed, noise_rate=0.05)
        print(f"[demo] wrote dataset → {csv_path}")

    split = load_ticket_routing_split(
        n_samples=120, seed=seed, noise_rate=0.05, csv_path=csv_path
    )
    rows = split["train"] + split["test"]
    grower = IGDecisionTreeGrower(min_samples=2, continuous_keys=[])
    tree = grower.fit(
        rows,
        split["label_key"],
        split["feature_keys"],
        criterion="gain",
        max_depth=max_depth,
    )

    tree_path = out_dir / "ticket_routing_tree.json"
    sop_path = out_dir / "ticket_routing.sop.json"
    grower.save(str(tree_path))
    sop = grower.export_sop(tree)
    sop_path.write_text(
        json.dumps(sop.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[demo] grew IG tree → {tree_path}")
    print(f"[demo] exported SOP  → {sop_path}")
    print(
        "[demo] This is a demo SOP for classmates — not a production Acc@budget claim."
    )

    engine = RuntimeEngine()
    payload = sop.to_dict()
    ok = True
    for case in DEMO_OBSERVATIONS:
        traced = engine.run_sop_traced(payload, case["obs"])
        print_trace(case, traced)
        expected = case["expected_route"]
        if traced["decision"] != expected:
            print(
                f"  !! unexpected route: got {traced['decision']!r}, "
                f"expected {expected!r} (demo still ran; check seed/data)"
            )
            ok = False

    print()
    print("CLI equivalent (after `pip install -e .`):")
    print(
        "  meta-jev grow --dataset ticket_routing "
        f"--out {tree_path} --sop-out {sop_path} --seed {seed}"
    )
    print(
        "  meta-jev run --sop "
        f"{sop_path} --input /tmp/ticket_obs.json --trace"
    )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Meta-Jev ticket-routing five-minute demo")
    p.add_argument(
        "--out-dir",
        default=str(_repo_root() / "examples" / "artifacts"),
        help="Where to write tree/SOP JSON (default: examples/artifacts)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--no-csv", action="store_true", help="Skip rewriting examples/data CSV")
    args = p.parse_args(argv)
    return run_demo(
        out_dir=Path(args.out_dir),
        seed=int(args.seed),
        max_depth=args.max_depth,
        write_csv=not args.no_csv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
