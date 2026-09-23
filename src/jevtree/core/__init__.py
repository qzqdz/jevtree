"""Core: entropy / IG, SOP structures, decision-tree grower interface."""

from jevtree.core.entropy import (
    best_split,
    class_counts,
    entropy,
    gain_ratio,
    information_gain,
    partition,
    split_info,
)
from jevtree.core.sop import (
    DecisionSOP,
    FeatureNode,
    JevNode,
    JevQuestion,
    bool_sop,
    classify_sop,
    score_sop,
)
from jevtree.core.tree import (
    DecisionTreeGrower,
    IGDecisionTreeGrower,
    TreeLeaf,
    TreeNode,
    bin_rows_continuous,
    quantile_bin_continuous,
)

__all__ = [
    "DecisionTreeGrower",
    "IGDecisionTreeGrower",
    "TreeLeaf",
    "TreeNode",
    "best_split",
    "bin_rows_continuous",
    "bool_sop",
    "class_counts",
    "classify_sop",
    "DecisionSOP",
    "entropy",
    "FeatureNode",
    "gain_ratio",
    "information_gain",
    "JevNode",
    "JevQuestion",
    "partition",
    "quantile_bin_continuous",
    "score_sop",
    "split_info",
]
