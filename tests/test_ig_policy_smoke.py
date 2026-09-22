"""Smoke: StaticIG + HardBudgetProtocol on synthetic cube."""

from __future__ import annotations

from meta_jev.data.cube import load_cube_split
from meta_jev.eval.protocol import HardBudgetEpisodeConfig, HardBudgetProtocol
from meta_jev.policy.ig_acquisition import StaticIGAcquisitionPolicy


def test_ig_policy_hard_budget_smoke() -> None:
    split = load_cube_split(n_samples=256, n_features=5, seed=0)
    train, test = split["train"], split["test"]
    fkeys, lkey = split["feature_keys"], split["label_key"]

    policy = StaticIGAcquisitionPolicy(force_acquisition=True)
    policy.fit(train, lkey, fkeys)

    # ranking should prefer informative features over distractors
    assert policy.ranking_
    assert "f0" in policy.ranking_[:3]

    proto = HardBudgetProtocol()
    cfg = HardBudgetEpisodeConfig(
        dataset_id=split["dataset_id"],
        hard_budget=3,
        split_seed=split["split_seed"],
        budget_schedule=[3],
        dataset_hash=split["dataset_hash"],
        policy_name="ig_static",
        feature_keys=fkeys,
        label_key=lkey,
    )
    summary = proto.run_eval(
        policy,
        cfg,
        test_rows=test,
        feature_keys=fkeys,
        label_key=lkey,
        budget_schedule=[3],
    )
    acc = summary["curve"][0]["accuracy"]
    # deterministic cube: should beat random (~0.5); allow margin
    assert acc > 0.6, f"expected acc>0.6 at budget=3, got {acc}"


def test_force_acquisition_never_stops_early() -> None:
    split = load_cube_split(n_samples=64, n_features=5, seed=1)
    policy = StaticIGAcquisitionPolicy(force_acquisition=True)
    policy.fit(split["train"], split["label_key"], split["feature_keys"])
    mask = [False] * 5
    for _ in range(5):
        a = policy.act([None] * 5, mask)
        assert a != 0
        mask[a - 1] = True
    assert policy.act([None] * 5, mask) == 0


if __name__ == "__main__":
    test_ig_policy_hard_budget_smoke()
    test_force_acquisition_never_stops_early()
    print("ok")
