"""Hard-budget episode API (research eval — not product UX).

Spec summary (aligned with AFABench):
  - Episode: start with empty mask; repeatedly call policy.act until stop or
    next acquisition would exceed hard_budget; then policy.predict.
  - hard_budget: int | None — unit costs by default.
  - When hard_budget is set, force acquisition (SupportsForcedAcquisition):
    policy.force_acquisition = True so act never emits stop=0.
  - First smoke: dataset cube_without_noise, hard_budget=3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from meta_jev.eval.metrics import accuracy_at_budget, f1_at_budget, summarize_curve


@dataclass
class HardBudgetEpisodeConfig:
    """Config for one hard-budget evaluation episode / run."""

    dataset_id: str
    hard_budget: int
    split_seed: int = 0
    budget_schedule: list[int] = field(default_factory=list)
    config_path: str | None = None
    git_commit: str | None = None
    dataset_hash: str | None = None
    policy_name: str | None = None
    feature_keys: list[str] = field(default_factory=list)
    label_key: str = "y"


@dataclass
class EpisodeResult:
    """Raw episode outcome before metrics aggregation."""

    acquired: list[int]
    prediction: Any
    label: Any
    budget_used: int
    stopped_reason: str  # "stop" | "budget" | "exhausted"


class HardBudgetProtocol:
    """Run hard-budget episodes for an AFA-shaped policy.

    Logging gate: implementations MUST record dataset id/hash, split seed,
    budget schedule, git commit, and config path alongside results/.
    """

    def run_episode(
        self,
        policy: Any,
        features: Sequence[Any],
        label: Any,
        *,
        hard_budget: int,
    ) -> EpisodeResult:
        n = len(features)
        feature_mask = [False] * n
        masked = [None] * n
        acquired: list[int] = []
        stopped_reason = "exhausted"
        # force acquisition under hard budget
        prev_force = getattr(policy, "force_acquisition", False)
        policy.force_acquisition = True

        try:
            while True:
                if len(acquired) >= hard_budget:
                    stopped_reason = "budget"
                    break
                if all(feature_mask):
                    stopped_reason = "exhausted"
                    break

                action = int(
                    policy.act(
                        masked_features=masked,
                        feature_mask=feature_mask,
                    )
                )
                if action == 0:
                    stopped_reason = "stop"
                    break
                feat_idx = action - 1
                if feat_idx < 0 or feat_idx >= n:
                    stopped_reason = "stop"
                    break
                if feature_mask[feat_idx]:
                    # already acquired — treat as stop to avoid loops
                    stopped_reason = "stop"
                    break
                # would this acquisition exceed budget?
                if len(acquired) + 1 > hard_budget:
                    stopped_reason = "budget"
                    break

                feature_mask[feat_idx] = True
                masked[feat_idx] = features[feat_idx]
                acquired.append(feat_idx)
        finally:
            policy.force_acquisition = prev_force

        prediction = policy.predict(
            masked_features=masked,
            feature_mask=feature_mask,
        )
        return EpisodeResult(
            acquired=acquired,
            prediction=prediction,
            label=label,
            budget_used=len(acquired),
            stopped_reason=stopped_reason,
        )

    def run_eval(
        self,
        policy: Any,
        cfg: HardBudgetEpisodeConfig,
        *,
        test_rows: list[dict[str, Any]] | None = None,
        feature_keys: list[str] | None = None,
        label_key: str | None = None,
        budget_schedule: list[int] | None = None,
    ) -> dict[str, Any]:
        """Evaluate policy on test rows for each hard budget in the schedule."""
        fkeys = list(feature_keys or cfg.feature_keys)
        lkey = label_key or cfg.label_key
        rows = test_rows if test_rows is not None else []
        schedule = list(budget_schedule or cfg.budget_schedule or [cfg.hard_budget])

        curve_points: list[dict[str, Any]] = []
        all_episodes: list[dict[str, Any]] = []

        for budget in schedule:
            y_true: list[Any] = []
            y_pred: list[Any] = []
            for ep_i, row in enumerate(rows):
                feats = [row[k] for k in fkeys]
                lab = row[lkey]
                result = self.run_episode(policy, feats, lab, hard_budget=int(budget))
                y_true.append(result.label)
                y_pred.append(result.prediction)
                all_episodes.append(
                    {
                        "budget": int(budget),
                        "episode_index": ep_i,
                        "acquired": result.acquired,
                        "prediction": result.prediction,
                        "label": result.label,
                        "budget_used": result.budget_used,
                        "stopped_reason": result.stopped_reason,
                    }
                )
            acc = accuracy_at_budget(y_true, y_pred, budget=int(budget))
            f1 = f1_at_budget(y_true, y_pred, budget=int(budget))
            curve_points.append(
                {
                    "budget": int(budget),
                    "accuracy": acc,
                    "f1": f1,
                    "n_episodes": len(rows),
                }
            )

        summary = summarize_curve(
            curve_points,
            dataset_id=cfg.dataset_id,
            dataset_hash=cfg.dataset_hash,
            split_seed=cfg.split_seed,
            budget_schedule=schedule,
            git_commit=cfg.git_commit,
            config_path=cfg.config_path,
            policy_name=cfg.policy_name,
        )
        summary["episodes"] = all_episodes
        return summary


__all__ = [
    "EpisodeResult",
    "HardBudgetEpisodeConfig",
    "HardBudgetProtocol",
]
