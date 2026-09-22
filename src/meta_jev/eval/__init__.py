"""Hard-budget AFA evaluation protocol (research path; not product UX)."""

from meta_jev.eval.afabench import AFABenchAdapter
from meta_jev.eval.metrics import accuracy_at_budget, f1_at_budget, summarize_curve
from meta_jev.eval.protocol import EpisodeResult, HardBudgetEpisodeConfig, HardBudgetProtocol

__all__ = [
    "AFABenchAdapter",
    "EpisodeResult",
    "HardBudgetEpisodeConfig",
    "HardBudgetProtocol",
    "accuracy_at_budget",
    "f1_at_budget",
    "summarize_curve",
]
