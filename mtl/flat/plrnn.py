"""The flat PLRNN and the thin subclass the joint scorers expect.

`PLRNN` moved here from `multi_task_training/rnn_model.py` (2026-10-21) -- just the
class, 72 of that file's 991 lines. The rest stayed behind on purpose: it holds an
older `compute_accuracies` and a `train_multitask` with an extra `fix_loss` term
that none of the trainers use, and leaving them there retires both for good.

`FlatPLRNN` is `PLRNN` with the joint pipeline's call signature. The scoring and
loss helpers call `model(inputs, task_ids)` because the hierarchical model needs
the subject index to select its p_vector; the flat model has no per-task
parameters, so it accepts and ignores that argument. Nothing else differs -- same
A/W/h/C/D, same dynamics, same native initialisation, same parameter count.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim):
        super().__init__()
        self.M = M
        self.L = L
        self.N = N
        self.input_dim = input_dim

        # Diagonal matrix A for latent dynamics
        self.A = nn.Parameter(torch.randn(L) * 0.1 + 0.9)

        # Recurrent weight matrix
        self.W = nn.Parameter(torch.randn(M, M) * 0.1 / np.sqrt(M))

        # Bias
        self.h = nn.Parameter(torch.zeros(M))

        # Input weights
        self.C = nn.Parameter(torch.randn(M, input_dim) * 0.1)

        # Output weights
        self.D = nn.Parameter(torch.randn(N, M) * 0.1)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.M)

    def forward(self, inputs):
        """
        Args:
            inputs: (batch, T, input_dim)
        Returns:
            outputs: (batch, N)
        """
        batch_size, T, _ = inputs.shape

        # Initialize hidden state
        z = self.init_hidden(batch_size).to(inputs.device)

        traj = []

        # Process sequence
        for t in range(T):
            # Split into non-latent and latent parts
            if self.L > 0:
                z_non_latent = z[:, : -self.L]  # excluim les L últimes dimensions
                z_latent = z[:, -self.L :]  # només les L últimes dimensions

                # Apply diagonal A to latent part only
                z_latent_scaled = self.A * z_latent

                # Apply ReLU to latent part
                z_latent_act = F.relu(z_latent)

                # Reconstruct z for matrix multiplication
                z_combined = torch.cat([z_non_latent, z_latent_act], dim=1)

                # Update
                z_update = z_latent_scaled
                z_update = torch.cat([torch.zeros_like(z_non_latent), z_update], dim=1)
            else:
                z_combined = z
                z_update = torch.zeros_like(z)

            # RNN update
            z = z_update + z_combined @ self.W.t() + inputs[:, t] @ self.C.t() + self.h

            # Generate output at this timestep
            output_t = z @ self.D.t()
            traj.append(output_t)

        # Stack all outputs: (batch, T, N)
        return torch.stack(traj, dim=1)


class FlatPLRNN(PLRNN):
    """PLRNN with the joint pipeline's call signature: forward(inputs, subject=None)."""

    def forward(self, inputs, subject=None):
        return super().forward(inputs)


__all__ = ["PLRNN", "FlatPLRNN"]
