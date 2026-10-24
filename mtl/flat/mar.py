"""The MAR regulariser, and which units it covers.

Moved here from `multi_task_training/MAR.py` (2026-10-21). Only the flat arm applies
this as a loss term: the hierarchical arm carries its own copy inside
`BPTT.regularization_loss`, with identical math.
"""

import torch


def regularization_loss(model, tau, M_reg):
    """
    Compute the regularization loss for the given model. Regularizes the first M_reg units per default.
    Args:
        model: The model object with attributes M, L, W, A, h.
        tau: Regularization strength (float).
        M_reg: Number of units to regularize (int).
    Returns:
        The combined regularization loss (torch scalar).
    """
    M, L = model.M, model.L
    linear = M - L
    loss = 0.0
    for i in range(M_reg):
        # effective diagonal term 𝛁A_ii
        if i < linear:  # linear unit
            diag_eff = model.W[i, i]
        else:  # nonlinear unit
            diag_eff = model.W[i, i] + model.A[i - linear]
        loss += (diag_eff - 1) ** 2  # first sum
        # off-diagonal W
        loss += torch.sum(model.W[i, :] ** 2) - model.W[i, i] ** 2

        # bias term
        loss += model.h[i] ** 2
    return tau * loss


def mar_units(hidden_size):
    """Units the MAR regulariser covers: the first half of the state (bptt convention)."""
    return hidden_size // 2


__all__ = ["mar_units", "regularization_loss"]
