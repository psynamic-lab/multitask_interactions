#!/usr/bin/env python
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

import argparse
import json
import logging
import time

import numpy as np
import torch

from mtl.analysis import bitcodes as cbd
from mtl.analysis.flat_bitcodes import FLAT, OUT, rollout_flat
from mtl.log import add_log_level_argument
from mtl.log import setup as setup_logging

logger = logging.getLogger(__name__)


def build_parser():
    """Command-line interface. Separate from main() so the parser can be
    imported and inspected (defaults, docs, tests) without running anything.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--no-js", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    add_log_level_argument(ap)
    return ap


def main():
    a = build_parser().parse_args()
    setup_logging(a.log_level)

    # any flat metadata fixes the task order; must match the cached battery
    meta0 = json.loads(next(FLAT.glob("*N0_P0*seed0")).joinpath("metadata.json").read_text())
    names = meta0["task_names"]
    battery = cbd.build_battery(names, a.trials, OUT / f"battery_K{a.trials}.npz")
    n_tasks = len(names)
    eye = torch.eye(n_tasks)

    for n in a.n:
        for seed in a.seeds:
            tag = f"P0_N{n}_seed{seed}"
            out_npz = OUT / f"counts_K{a.trials}" / f"{tag}.npz"
            if a.skip_existing and out_npz.exists():
                continue
            t0 = time.time()
            folder = next(FLAT.glob(f"flat_joint_M234_N{n}_P0_*seed{seed}"))
            state = torch.load(folder / "model.pt", map_location="cpu")
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            M, L = state["W"].shape[0], int(state["A"].shape[0])
            assert L == n and state["C"].shape[1] == 7 + n_tasks, (tag, L, state["C"].shape)

            codes_cat, counts_cat, offsets = [], [], [0]
            codes_list, probs_list = [], []
            for t_idx in range(n_tasks):
                item = battery[t_idx]
                inp = torch.from_numpy(item["inputs"])
                onehot = eye[t_idx].expand(inp.shape[0], inp.shape[1], n_tasks)
                z_all = rollout_flat(state, torch.cat([inp, onehot], dim=2), L)
                stats = cbd.task_bitcode_stats(cbd.bitcodes(z_all, L), item["lengths"])
                codes_cat.append(stats["codes"])
                counts_cat.append(stats["counts"])
                offsets.append(offsets[-1] + stats["codes"].size)
                codes_list.append(stats["codes"])
                probs_list.append(stats["probs"])

            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_npz,
                codes=np.concatenate(codes_cat), counts=np.concatenate(counts_cat),
                task_offsets=np.array(offsets, dtype=np.int64),
                task_names=np.array(names), n_bits=np.int32(n), P=np.int32(0),
                seed=np.int32(seed), hidden_size=np.int32(M),
                n_trials=np.int32(a.trials), run_dir=str(folder))
            if not a.no_js:
                js_dir = OUT / f"js_K{a.trials}"
                js_dir.mkdir(parents=True, exist_ok=True)
                np.save(js_dir / f"{tag}.npy", cbd.js_similarity(codes_list, probs_list))
            logger.info(f"{tag}: done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
