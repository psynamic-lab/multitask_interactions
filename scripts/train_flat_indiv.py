#!/usr/bin/env python
"""Independent per-task training of the flat, P-free PLRNN: one model per task."""

import argparse
import logging
from multiprocessing import Pool

import numpy as np

from mtl import tasks as ft
from mtl.indiv.runner import build_config, train_one
from mtl.log import add_log_level_argument
from mtl.log import setup as setup_logging
from mtl.shared.scoring import THRESHOLD

logger = logging.getLogger(__name__)


def build_parser():
    """Command-line interface. Separate from main() so the parser can be
    imported and inspected (defaults, docs, tests) without running anything.
    """
    all_names = list(ft.FOUNDATION_TASKS_MAP)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=all_names,
                   help=f"tasks, each in its OWN model (default: all {len(all_names)})")
    p.add_argument("--nonlinear-units", "-P", nargs="+", type=int,
                   default=[0, 1, 2, 4, 8, 16],
                   help="P: number of nonlinear (ReLU) units; the remaining M-P units stay linear. Takes a list: one run per value.")
    p.add_argument("--sample-sizes", nargs="+", type=int, default=[200])
    p.add_argument("--seeds", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--hidden-size", "-M", type=int, default=64,
                   help="M: hidden/latent state dimension")
    p.add_argument("--val-size", type=int, default=50, help="joint runs used VAL=50")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--lr-warmup-epochs", type=int, default=0,
                   help="LR ramp over this many epochs. 0 = OFF (the default here); the "
                        "joint runs used 100. Set it only if you want that ramp back.")
    p.add_argument("--patience", type=int, default=500, help="on validation loss (0 = off)")
    p.add_argument("--es-start", type=int, default=300)
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--threads", type=int, default=1,
                   help="torch threads PER worker (num_processes*threads <= cores)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip combos whose metadata.json exists (idempotent resubmits)")
    add_log_level_argument(p)
    return p


def main():
    a = build_parser().parse_args()
    setup_logging(a.log_level)

    unknown = [t for t in a.tasks if t not in ft.FOUNDATION_TASKS_MAP]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}")

    cfg = build_config(a)
    jobs = [(t, L, s, seed, cfg)
            for t in a.tasks for L in a.nonlinear_units
            for s in a.sample_sizes for seed in a.seeds]

    logger.info(f"{len(jobs)} runs | {len(a.tasks)} tasks x L={a.nonlinear_units} "
                f"x s={a.sample_sizes} x seeds={a.seeds}")
    logger.info(f"M={a.hidden_size} batch={a.batch_size} lr={a.learning_rate} "
                f"wd={a.weight_decay} clip={a.grad_clip} warmup={a.lr_warmup_epochs} "
                f"eval={a.eval_interval} patience={a.patience}@{a.es_start} "
                f"val={a.val_size} | P=0, native init, "
                f"threshold={THRESHOLD * 180 / np.pi:.0f}deg + fixation gate")
    logger.info(f"-> {a.results_dir}")

    if a.num_processes > 1:
        with Pool(processes=a.num_processes) as pool:
            for msg in pool.imap_unordered(train_one, jobs):
                logger.info(msg)
    else:
        for job in jobs:
            logger.info(train_one(job))

if __name__ == "__main__":
    main()
