"""The hierarchical arm: a per-task `p_vector` projected into A, W, h, C, D.

`mtl.hierarchical.joint` holds the joint-training stack (one model over the whole
battery, task identity carried by the p_vector). The network, its projection
schemes, its config and its BPTT trainer are all specific to this arm.
"""

from mtl.hierarchical.config import HierarchicalModelConfig
from mtl.hierarchical.model import HierarchicalPLRNN

__all__ = ["HierarchicalModelConfig", "HierarchicalPLRNN"]
