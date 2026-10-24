"""Bitcode distributions for the FLAT joint models (data/20260827_flat_joint_100tasks).

Companion to foundation_model/notebooks/compute_bitcode_distributions.py (imported for the battery,
bitcode, stats and JS code) -- needed because the flat runs are FlatPLRNN checkpoints
(raw A/W/h/C/D, one-hot task input appended to the 7 channels; tasks/dataset.py:118-120)
which load_model() there cannot rebuild. Same battery cache (battery_K50.npz), same output
conventions, tagged P0_N{n}_seed{s} into the same counts_K50/ and js_K50/ dirs, so the
routing notebook reads them exactly like the P8/P16 cells.

    python compute_flat_bitcodes.py            # all N in {0,1,2,4,8,16} x seeds 0-4
    python compute_flat_bitcodes.py --n 4 --seeds 0
"""


import torch

from mtl.paths import ROOT

FLAT = ROOT / "data/20260827_flat_joint_100tasks"
OUT = ROOT / "data/bitcodes_NPsweep_varnorm"


@torch.no_grad()
def rollout_flat(state, inputs, L):
    """PLRNN.forward (multi_task_training/rnn_model.py:49-88) returning pre-activation z_all."""
    A, W, h, C = state["A"], state["W"], state["h"], state["C"]
    B, T, _ = inputs.shape
    M = W.shape[0]
    z = torch.zeros(B, M)
    z_all = torch.empty(B, T, M)
    for t in range(T):
        if L > 0:
            z_nl, z_lat = z[:, :-L], z[:, -L:]
            z_combined = torch.cat([z_nl, torch.relu(z_lat)], dim=1)
            z_update = torch.cat([torch.zeros_like(z_nl), A * z_lat], dim=1)
        else:
            z_combined, z_update = z, torch.zeros_like(z)
        z = z_update + z_combined @ W.T + inputs[:, t] @ C.T + h
        z_all[:, t] = z
    return z_all
