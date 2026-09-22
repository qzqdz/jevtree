"""AFA acquisition policies (IG and baselines)."""

from meta_jev.policy.baselines import RandomAcquisitionPolicy, SequentialAcquisitionPolicy
from meta_jev.policy.conditional_ig import ConditionalIGAcquisitionPolicy
from meta_jev.policy.discriminative import DiscriminativeAcquisitionPolicy
from meta_jev.policy.ig_acquisition import IGAcquisitionPolicy, StaticIGAcquisitionPolicy
from meta_jev.policy.predictor import MaskedLogisticPredictor, MatchMajorityPredictor

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
