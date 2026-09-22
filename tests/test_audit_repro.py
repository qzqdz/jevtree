"""Audit / reproducibility regressions for Meta-Jev eval-afa."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_jev.data.cube import load_cube_split
from meta_jev.data.tabular import (
    MINIBOONE_MISSING_SENTINEL,
    ensure_train_test_disjoint_features,
    load_tabular_split,
)
from meta_jev.eval.afabench import AFABenchAdapter
from meta_jev.eval.metrics import summarize_curve
from meta_jev.eval.provenance import collect_provenance, git_dirty
from meta_jev.eval.protocol import HardBudgetEpisodeConfig, HardBudgetProtocol
from meta_jev.policy.ig_acquisition import StaticIGAcquisitionPolicy
from meta_jev.tools.compare_results import ComparisonError, build_comparison, validate_summaries_compatible


def test_compare_results_rejects_mismatched_predictor_hash_seed(tmp_path: Path) -> None:
    base = {
        "dataset_hash": "abc123",
        "predictor_name": "logistic_impute",
        "split_seed": 0,
        "n_train": 100,
        "n_test": 50,
        "git_commit": "deadbeef",
        "git_dirty": False,
        "policy_name": "ig_static",
        "curve": [{"budget": 3, "accuracy": 0.9, "f1": 0.8, "n_episodes": 50}],
    }
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "summary.json").write_text(json.dumps(base) + "\n", encoding="utf-8")

    # predictor mismatch
    bad = dict(base, predictor_name="match_majority")
    (b / "summary.json").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ComparisonError, match="predictor_name"):
        build_comparison([a, b])

    # hash mismatch
    bad = dict(base, predictor_name="logistic_impute", dataset_hash="other")
    (b / "summary.json").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ComparisonError, match="dataset_hash"):
        build_comparison([a, b])

    # split_seed mismatch
    bad = dict(base, dataset_hash="abc123", split_seed=1)
    (b / "summary.json").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ComparisonError, match="split_seed"):
        build_comparison([a, b])

    # n_test mismatch
    bad = dict(base, split_seed=0, n_test=51)
    (b / "summary.json").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ComparisonError, match="n_test"):
        build_comparison([a, b])

    # mixed dirty/clean
    bad = dict(base, n_test=50, git_dirty=True)
    (b / "summary.json").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ComparisonError, match="git_dirty"):
        build_comparison([a, b])

    # commit mismatch
    bad = dict(base, git_dirty=False, git_commit="cafebabe")
    (b / "summary.json").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ComparisonError, match="git_commit"):
        build_comparison([a, b])

    # compatible pair OK
    (b / "summary.json").write_text(json.dumps(base) + "\n", encoding="utf-8")
    comp = build_comparison([a, b], labels=["A", "B"])
    assert len(comp["runs"]) == 2
    assert comp["table"][0]["A_acc"] == 0.9


def test_summary_provenance_fields_present() -> None:
    prov = collect_provenance(
        predictor_name="match_majority",
        policy_name="ig_static",
        dataset_hash="deadbeefdeadbeef",
        split_seed=0,
        budgets=[1, 2, 3],
        config_path="configs/eval_cube_hard.json",
    )
    for key in (
        "git_commit",
        "git_dirty",
        "python_version",
        "numpy_version",
        "predictor_name",
        "policy_name",
        "dataset_hash",
        "split_seed",
        "budgets",
        "config_path",
    ):
        assert key in prov, f"missing provenance key {key}"
    assert isinstance(prov["git_dirty"], bool)
    assert isinstance(prov["python_version"], str)
    assert prov["predictor_name"] == "match_majority"
    # Working tree in this repo is expected dirty until committed; never claim clean wrongly.
    # Just assert the flag is a real bool from git status (or True on VCS failure).
    assert prov["git_dirty"] in (True, False)

    # summarize_curve still carries core fields
    summary = summarize_curve(
        [{"budget": 1, "accuracy": 1.0, "f1": 1.0, "n_episodes": 1}],
        dataset_id="cube_without_noise",
        dataset_hash="h",
        split_seed=0,
        budget_schedule=[1],
        git_commit=prov["git_commit"],
        config_path="configs/eval_cube_hard.json",
        policy_name="ig_static",
    )
    assert summary["git_commit"] == prov["git_commit"]
    assert summary["dataset_hash"] == "h"


def test_miniboone_split_no_train_test_feature_duplicates() -> None:
    split = load_tabular_split("miniboone", seed=0, n_train=2000, n_test=500)
    fk = split["feature_keys"]
    train, test = split["train"], split["test"]
    train_keys = {tuple(r[k] for k in fk) for r in train}
    leaks = [i for i, r in enumerate(test) if tuple(r[k] for k in fk) in train_keys]
    assert leaks == [], f"train/test feature leaks at test indices {leaks}"

    # No all-sentinel rows remain in the split
    def all_sent(r: dict) -> bool:
        return all(abs(float(r[k]) - MINIBOONE_MISSING_SENTINEL) < 1e-9 for k in fk)

    assert not any(all_sent(r) for r in train)
    assert not any(all_sent(r) for r in test)
    assert split["hygiene"]["dropped_all_sentinel"] > 0
    # Hash must reflect post-hygiene rows (changed from pre-hygiene c9de6ed766fcee87)
    assert split["dataset_hash"]
    assert split["dataset_hash"] != "c9de6ed766fcee87"


def test_eval_determinism_same_seed_cube() -> None:
    """Two evals with same seed produce identical Acc table (cube, CI-fast)."""

    def once() -> list[tuple[int, float]]:
        split = load_cube_split(n_samples=128, n_features=5, seed=0)
        train, test = split["train"], split["test"]
        fkeys, lkey = split["feature_keys"], split["label_key"]
        policy = StaticIGAcquisitionPolicy(
            force_acquisition=True,
            predictor_name="match_majority",
            bin_seed=0,
        )
        policy.fit(train, lkey, fkeys)
        adapter = AFABenchAdapter(policy, force_acquisition=True, predictor_name="match_majority")
        proto = HardBudgetProtocol()
        cfg = HardBudgetEpisodeConfig(
            dataset_id=split["dataset_id"],
            hard_budget=1,
            split_seed=0,
            budget_schedule=[1, 2, 3],
            dataset_hash=split["dataset_hash"],
            policy_name="ig_static",
            feature_keys=fkeys,
            label_key=lkey,
        )
        summary = proto.run_eval(
            adapter,
            cfg,
            test_rows=test,
            feature_keys=fkeys,
            label_key=lkey,
            budget_schedule=[1, 2, 3],
        )
        return [(int(p["budget"]), float(p["accuracy"])) for p in summary["curve"]]

    a = once()
    b = once()
    assert a == b
    assert len(a) == 3


def test_ensure_disjoint_helper() -> None:
    fk = ["f0", "f1"]
    train = [{"f0": 1.0, "f1": 2.0, "y": 0}, {"f0": 3.0, "f1": 4.0, "y": 1}]
    test = [
        {"f0": 1.0, "f1": 2.0, "y": 0},  # leak
        {"f0": 9.0, "f1": 8.0, "y": 1},
    ]
    tr, te, n = ensure_train_test_disjoint_features(train, test, fk)
    assert n == 1
    assert len(te) == 1
    assert te[0]["f0"] == 9.0
    assert len(tr) == 2


def test_git_dirty_helper_returns_bool() -> None:
    assert isinstance(git_dirty(), bool)


def test_random_acquisition_same_seed_mask_same_act() -> None:
    """Same seed + mask → same act; order/budget history must not leak shared RNG."""
    from meta_jev.policy.baselines import RandomAcquisitionPolicy

    fkeys = [f"f{i}" for i in range(8)]
    # Tiny balanced table so fit succeeds; acquisition ignores labels for act.
    train = [
        {**{fk: float((i + j) % 3) for j, fk in enumerate(fkeys)}, "y": i % 2}
        for i in range(24)
    ]

    def make(seed: int = 0) -> RandomAcquisitionPolicy:
        p = RandomAcquisitionPolicy(
            force_acquisition=True,
            seed=seed,
            predictor_name="match_majority",
            n_bins=3,
            bin_seed=0,
        )
        p.fit(train, "y", fkeys)
        return p

    mask0 = [False] * 8
    p1 = make(0)
    p2 = make(0)
    a1 = p1.act([0.0] * 8, mask0)
    a2 = p2.act([0.0] * 8, mask0)
    assert a1 == a2 and a1 >= 1

    # Different histories ending at the SAME mask must agree (no shared RNG leak).
    p_hist = make(0)
    # Simulate budget-5 path: acquire five features in an arbitrary order.
    mask = [False] * 8
    feats = [0.0] * 8
    for _ in range(5):
        act = p_hist.act(feats, mask)
        idx = act - 1
        mask[idx] = True
    mid_mask = list(mask)
    act_after_budget5 = p_hist.act(feats, mid_mask)

    p_fresh = make(0)
    act_fresh = p_fresh.act(feats, mid_mask)
    assert act_after_budget5 == act_fresh, (
        f"budget history leaked RNG: hist={act_after_budget5} fresh={act_fresh}"
    )

    # Different seed → different (or at least independently seeded) choice stream.
    # Across many masks, seeds should not be identical everywhere.
    disagreements = 0
    for k in range(8):
        m = [False] * 8
        m[k] = True  # one observed
        if make(0).act(feats, m) != make(1).act(feats, m):
            disagreements += 1
    assert disagreements >= 1


def test_ig_ranking_stable_tie_break() -> None:
    """Construct exact IG ties; ranking order must be deterministic by feature name."""
    from meta_jev.core.entropy import best_split, information_gain

    # Two features with identical partitions of labels → identical IG.
    # f_a and f_b both split as L/L/R/R vs labels neg/neg/pos/pos → equal IG.
    # f_z is constant → IG 0, ranks last.
    rows = [
        {"f_b": "L", "f_a": "L", "f_z": "C", "y": "neg"},
        {"f_b": "L", "f_a": "L", "f_z": "C", "y": "neg"},
        {"f_b": "R", "f_a": "R", "f_z": "C", "y": "pos"},
        {"f_b": "R", "f_a": "R", "f_z": "C", "y": "pos"},
    ]
    feature_keys = ["f_b", "f_a", "f_z"]  # intentional non-alpha input order
    labels = [r["y"] for r in rows]
    ig_a = information_gain(labels, [r["f_a"] for r in rows])
    ig_b = information_gain(labels, [r["f_b"] for r in rows])
    ig_z = information_gain(labels, [r["f_z"] for r in rows])
    assert ig_a == ig_b > ig_z

    split = best_split(rows, "y", feature_keys, criterion="gain")
    ranking = [r["feature"] for r in split["ranking"]]
    # Tie-break: higher score first, then feature name ascending.
    assert ranking == ["f_a", "f_b", "f_z"], ranking

    # Repeat: order must be stable across calls.
    ranking2 = [
        r["feature"]
        for r in best_split(rows, "y", feature_keys, criterion="gain")["ranking"]
    ]
    assert ranking2 == ranking

    # StaticIG policy ranking must match the same tie-break.
    from meta_jev.policy.ig_acquisition import StaticIGAcquisitionPolicy

    pol = StaticIGAcquisitionPolicy(
        force_acquisition=True,
        predictor_name="match_majority",
        n_bins=4,
        bin_seed=0,
        continuous_keys=[],
    )
    pol.fit(rows, "y", feature_keys)
    assert pol.ranking_[:2] == ["f_a", "f_b"], pol.ranking_
    assert pol.ranking_[-1] == "f_z"
