"""AFA acquisition policies (IG and baselines)."""

from jevtree.policy.baselines import RandomAcquisitionPolicy, SequentialAcquisitionPolicy
from jevtree.policy.conditional_ig import ConditionalIGAcquisitionPolicy
from jevtree.policy.discriminative import DiscriminativeAcquisitionPolicy
from jevtree.policy.ig_acquisition import IGAcquisitionPolicy, StaticIGAcquisitionPolicy
from jevtree.policy.predictor import MaskedLogisticPredictor, MatchMajorityPredictor

__all__ = [
    "IGAcquisitionPolicy",
    "StaticIGAcquisitionPolicy",
    "ConditionalIGAcquisitionPolicy",
    "DiscriminativeAcquisitionPolicy",
    "RandomAcquisitionPolicy",
    "SequentialAcquisitionPolicy",
    "MatchMajorityPredictor",
    "MaskedLogisticPredictor",
]
