#!/usr/bin/env python
"""Dump the latent trajectories of the N x P sweep next to the existing bitcodes.

compute_bitcode_distributions.py rolls z_all out and throws it away, keeping only
the sparse code histograms (counts_K*/). Geometry work -- subspace overlap, PCA,
DSA -- needs the trajectories themselves, so this replays the *same* rollout on
the *same* cached battery and writes only z_all. Bitcodes are NOT recomputed:
counts_K*/ already has them, and per-timestep codes are one line away from z
(`bitcodes(z, N)`, see `codes_from_z` below). --verify uses the stored counts to
prove the dumped trajectories come from the same checkpoints.

Layout -- one .npz per cell inside the bitcode folder, ragged over tasks:

  <bitcode-dir>/trajectories_K{K}/P{P}_N{N}_seed{S}.npz
    z_{i}        (K, T_i, M) float32/float16  pre-activation latent states
    lengths_{i}  (K,)        int32            valid timesteps (rest is padding)
    masks_{i}    (K, T_i)    float32          response-phase LOSS mask (not padding)
    task_names, n_bits, P, seed, hidden_size, n_trials, run_dir, sweep_dir,
    battery, dtype

Phase labels live in ONE file next to the battery, since they are a property of
the stimuli and therefore shared by every (P, N, seed) cell:

  <bitcode-dir>/battery_phases_K{K}.npz
    phases_{i}      (K, T_i) int64   phase index per timestep, -1 on padding
    phase_names_{i} (n_ph,)  str     PHASE_NAMES of the task class
    informative     (n_tasks,) bool  False where the task exposes a single phase
    task_names, n_trials, seed, battery

build_battery draws trials with task(); this redraws them with
task.generate_trial_with_phases(), which consumes the RNG identically, and then
ASSERTS the regenerated inputs are byte-identical to the cached battery. That
assert is what guarantees the phase labels line up with the saved trajectories.

Example
-------
python foundation_model/notebooks/save_trajectories.py --p 8 16 --n 4 --verify --phases
"""

import argparse
import logging
import time
from pathlib import Path

import tasks as ft
from mtl.analysis.bitcodes import build_battery
from mtl.analysis.trajectories import (
    DEFAULT_BITCODES,
    DEFAULT_SWEEP,
    dump_cell,
    find_cell,
    save_phases,
    verify_against_counts,
)
from mtl.log import add_log_level_argument
from mtl.log import setup as setup_logging

logger = logging.getLogger(__name__)


def build_parser():
    """Command-line interface. Separate from main() so the parser can be
    imported and inspected (defaults, docs, tests) without running anything.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bitcode-dir", type=Path, default=DEFAULT_BITCODES,
                    help="folder holding battery_K*.npz / counts_K*; output goes here")
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP,
                    help="checkpoints; must be the sweep --bitcode-dir was built from")
    ap.add_argument("--out-name", default=None,
                    help="subfolder name (default trajectories_K{trials})")
    ap.add_argument("--p", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--n", type=int, nargs="+", default=[4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--trials", type=int, default=50, help="trials per task (K)")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--compress", action="store_true", help="np.savez_compressed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check each cell against counts_K{trials}/")
    ap.add_argument("--phases", action="store_true",
                    help="also write battery_phases_K{trials}.npz (shared by all cells)")
    ap.add_argument("--phases-only", action="store_true",
                    help="write only the phase labels; touch no trajectories")
    ap.add_argument("--allow-spectral", action="store_true")
    ap.add_argument("--max-tasks", type=int, default=None, help="debug: first T tasks")
    add_log_level_argument(ap)
    return ap


def main():
    args = build_parser().parse_args()
    setup_logging(args.log_level)

    args.battery = args.bitcode_dir / f"battery_K{args.trials}.npz"
    out_dir = args.bitcode_dir / (args.out_name or f"trajectories_K{args.trials}")
    if args.bitcode_dir.name.endswith("_varnorm") != args.sweep_dir.name.endswith("_varnorm"):
        logger.info(f"[warn] {args.sweep_dir.name} and {args.bitcode_dir.name} look like "
              f"different sweeps; the trajectories will not match counts_K*")

    all_names = list(ft.FOUNDATION_TASKS_MAP)
    battery = build_battery(all_names, args.trials, args.battery)
    task_names = all_names[: args.max_tasks] if args.max_tasks else all_names

    if args.phases or args.phases_only:
        ph_path = args.bitcode_dir / f"battery_phases_K{args.trials}.npz"
        t1 = time.time()
        logger.info(f"[run ] phase labels for {args.battery.name}")
        info = save_phases(args.battery, ph_path, args.trials)
        logger.info(
            f"    {info['n_tasks']} tasks | {info['n_informative']} with >1 phase | "
            f"{info['mb']:.1f} MB | {time.time() - t1:.1f}s -> {ph_path}",
            flush=True,
        )
        if info["trivial"]:
            logger.info(
                f"    single-phase (label 0 everywhere, {len(info['trivial'])} tasks): "
                + ", ".join(info["trivial"][:6])
                + (" ..." if len(info["trivial"]) > 6 else ""),
                flush=True,
            )
        if args.phases_only:
            return

    t0, done, missing = time.time(), [], []
    for p in args.p:
        for n in args.n:
            for s in args.seeds:
                tag = f"P{p}_N{n}_seed{s}"
                out_path = out_dir / f"{tag}.npz"
                hits = find_cell(args.sweep_dir, p, n, s)
                if len(hits) != 1:
                    missing.append(f"{tag} ({len(hits)} dirs with model.pt)")
                    logger.info(f"[skip] {tag}: {len(hits)} matching run dirs")
                    continue
                if args.skip_existing and out_path.exists():
                    logger.info(f"[have] {tag}")
                    continue

                t1 = time.time()
                logger.info(f"[run ] {tag}  <- {hits[0].name}")
                info = dump_cell(hits[0], battery, task_names, out_path, args)
                done.append(tag)
                note = ""
                if args.verify:
                    counts_path = args.bitcode_dir / f"counts_K{args.trials}" / f"{tag}.npz"
                    if counts_path.exists():
                        n_ok = verify_against_counts(
                            counts_path, info["codes"],
                            "" if args.dtype == "float32" else " (--dtype float16?)",
                        )
                        note = f" | verified vs counts ({n_ok} tasks)"
                    else:
                        note = f" | no counts file to verify ({counts_path.name})"
                logger.info(
                    f"    N={info['n_bits']} | {len(task_names)} tasks | "
                    f"{info['n_valid']} valid steps | {info['mb']:.1f} MB | "
                    f"{time.time() - t1:.1f}s -> {out_path}{note}",
                    flush=True,
                )

    logger.info(
        f"\ndone in {time.time() - t0:.1f}s | wrote {len(done)} cells to {out_dir}"
        + (f" | MISSING {len(missing)}: {', '.join(missing)}" if missing else "")
    )


if __name__ == "__main__":
    main()
