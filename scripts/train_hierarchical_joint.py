#!/usr/bin/env python
# Migrated copy: imports go through the installed `mtl` package, so there is no
# sys.path manipulation. The original is still in place and unmodified.
"""Joint training of the hierarchical AL-RNN on the full foundation battery.

Trains one model over all FOUNDATION_TASKS_MAP tasks, with balanced 
per-task sampling. Saves the model, full loss/accuracy curves, and
per-task validation accuracies.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from mtl import tasks as ft
from mtl.hierarchical.bptt import BPTT
from mtl.hierarchical.joint import apply_init, build_model_config, save_run
from mtl.hierarchical.model import HierarchicalPLRNN
from mtl.hierarchical.projection import parse_hierarchisation
from mtl.log import add_log_level_argument
from mtl.log import setup as setup_logging
from mtl.tasks.dataset import HierarchicalTasksDataset, balanced_loaders

logger = logging.getLogger(__name__)


def build_parser():
    """Command-line interface. Separate from main() so the parser can be
    imported and inspected (defaults, docs, tests) without running anything."""
    p = argparse.ArgumentParser(
        description="Joint-train the AL-RNN on the foundation battery."
    )
    p.add_argument("--sample-size", type=int, default=50)
    p.add_argument("--val-size", type=int, default=50)
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--early-stopping-patience", type=int, default=50)
    p.add_argument("--early-stopping-start", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--a-clamp", type=float, default=0.0)
    p.add_argument("--lr-warmup-epochs", type=int, default=0)
    p.add_argument("--w-init-gain", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument(
        "--init-mode",
        choices=["default", "gaussian", "orthogonal"],
        default="default",
        help="p2W init: default=xavier-uniform (model default), "
        "gaussian=xavier-normal, orthogonal=orthonormal slices",
    )
    p.add_argument(
        "--init-gain",
        type=float,
        default=0.9,
        help="gain for gaussian/orthogonal init (default 0.9)",
    )
    p.add_argument("--hidden-size", "-M", type=int, default=64,
                   help="M: hidden/latent state dimension")
    p.add_argument("--nonlinear-units", "-P", type=int, default=2,
                   help="P: number of nonlinear (ReLU) units; the remaining M-P units stay linear")
    p.add_argument("--num-individual-params", "-N", type=int, default=4,
                   help="N: task feature vector size (the per-task p_vector the hypernetwork reads)")
    p.add_argument(
        "--hierarchisation",
        default="all",
        help="which of A, W, h, C, D are task-specific (projected from the p_vector); "
        "the rest are shared. 'all' (default), a subset like 'hCD', or 'all-AW' / 'all-C'.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "results" / "foundation_joint"
        ),
    )
    p.add_argument(
        "--no-varnorm",
        dest="varnorm",
        action="store_false",
        default=True,
        help="disable the variance-normalized loss. ON by default: every reference "
        "run uses it, so turning it off is the thing that needs stating.",
    )
    p.add_argument("--resume", type=str, default="")
    p.add_argument(
        "--leave-out",
        type=str,
        default="",
        help="task NAME to hold out; joint-train on the other tasks. "
        "'' = train on the full battery (default).",
    )
    add_log_level_argument(p)
    return p


def main():
    a = build_parser().parse_args()
    setup_logging(a.log_level)
    a.hierarchisation = parse_hierarchisation(a.hierarchisation)  # fail fast on a typo

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    all_names = list(ft.FOUNDATION_TASKS_MAP)
    left_out = None
    if a.leave_out:
        if a.leave_out not in ft.FOUNDATION_TASKS_MAP:
            raise SystemExit(f"--leave-out '{a.leave_out}' is not in the battery")
        left_out = a.leave_out

    names = [n for n in all_names if n != left_out]  # left_out=None -> full battery
    tasks = [ft.FOUNDATION_TASKS_MAP[n] for n in names]
    n_tasks = len(tasks)
    
    if left_out is not None:
        logger.info(
            f"LEAVE-ONE-OUT: holding out '{left_out}' "
            f"(index {all_names.index(left_out)} of {len(all_names)})"
        )
    logger.info(
        f"Foundation battery: {n_tasks} tasks "
        f"| M={a.hidden_size} P={a.nonlinear_units} N={a.num_individual_params} "
        f"hier={a.hierarchisation} s={a.sample_size} seed={a.seed}"
    )

    train_loader, val_loader = balanced_loaders(
        tasks, a.sample_size, a.val_size, a.batch_size, a.seed
    )

    args = build_model_config(tasks, a)
    logger.info(f"device: {args.device}")

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.resume:
        logger.info(f"resuming from checkpoint: {a.resume}")
        model = HierarchicalPLRNN(args, HierarchicalTasksDataset)
        model.load_state_dict(torch.load(a.resume, map_location=args.device))
        alg = BPTT(model, args)
    else:
        alg = BPTT(None, args)

    # <<< optional init-mode (OFF by default) >>>
    if a.init_mode != "default" and not a.resume:
        apply_init(alg.model, a.init_mode, a.init_gain)
        logger.info(f"init-mode: {a.init_mode} (gain={a.init_gain}) on p2W")

    curves = alg.train(
        train_loader,
        val_loader,
        verbose=True,
        task_names=names,
        eval_interval=a.eval_interval,
        checkpoint_path=str(out / "model.pt"),
        checkpoint_acc_path=str(out / "model_bestacc.pt"),
    )

    best = save_run(out, a, alg, names, all_names, left_out, curves)

    logger.info(
        f"\nmean best-epoch val acc {np.mean(list(best.values())):.1%}  ->  {out}"
    )

if __name__ == "__main__":
    main()
