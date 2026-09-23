"""Hard-budget AFA evaluation protocol (research path; not product UX)."""

from jevtree.eval.afabench import AFABenchAdapter
from jevtree.eval.metrics import accuracy_at_budget, f1_at_budget, summarize_curve
from jevtree.eval.protocol import EpisodeResult, HardBudgetEpisodeConfig, HardBudgetProtocol

__all__ = [
    "AFABenchAdapter",
    "EpisodeResult",
    "HardBudgetEpisodeConfig",
    "HardBudgetProtocol",
    "accuracy_at_budget",
    "f1_at_budget",
    "summarize_curve",
]
