"""Smoke: ConditionalIG + MaskedLogistic on synthetic cube."""

from __future__ import annotations

from meta_jev.data.cube import load_cube_split
from meta_jev.eval.protocol import HardBudgetEpisodeConfig, HardBudgetProtocol
from meta_jev.policy.conditional_ig import ConditionalIGAcquisitionPolicy


def test_conditional_ig_cube_acc_at_3() -> None:
    split = load_cube_split(n_samples=256, n_features=5, seed=0)
    train, test = split["train"], split["test"]
    fkeys, lkey = split["feature_keys"], split["label_key"]

    policy = ConditionalIGAcquisitionPolicy(
        force_acquisition=True,
        predictor_name="logistic_impute",
        min_samples=10,
        max_candidates=5,
    )
    policy.fit(train, lkey, fkeys)
    assert policy.ranking_
    assert "f0" in policy.ranking_[:3]

    proto = HardBudgetProtocol()
    cfg = HardBudgetEpisodeConfig(
        dataset_id=split["dataset_id"],
        hard_budget=3,
        split_seed=split["split_seed"],
        budget_schedule=[3],
        dataset_hash=split["dataset_hash"],
        policy_name="ig_conditional",
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
    assert acc > 0.6, f"expected Acc@3 > 0.6, got {acc}"


def test_conditional_force_acquisition() -> None:
    split = load_cube_split(n_samples=64, n_features=5, seed=1)
    policy = ConditionalIGAcquisitionPolicy(force_acquisition=True, min_samples=5)
    policy.fit(split["train"], split["label_key"], split["feature_keys"])
    mask = [False] * 5
    feats = [0.0] * 5
    for _ in range(5):
        a = policy.act(feats, mask)
        assert a != 0
        mask[a - 1] = True
        feats[a - 1] = 1.0
    assert policy.act(feats, mask) == 0


if __name__ == "__main__":
    test_conditional_ig_cube_acc_at_3()
    test_conditional_force_acquisition()
    print("ok")
