#!/usr/bin/env python
"""Joint training of ONE flat, P-free PLRNN on the 100-task battery.

One shared A/W/h/C/D for every task. There is no p_vector, so task identity enters
through the INPUT as a one-hot (7 -> 7+n_tasks channels); `mtl.flat.joint` explains
what that costs and what is matched to the hierarchical runs.

    parameterization   individual                  joint
    hierarchical       (n/a, needs >1 task)        train_hierarchical_joint.py
    flat (P-free)      train_flat_indiv.py         >>> THIS SCRIPT <<<

COST: per epoch this is n_tasks x the individual runs -- at --sample-size 200 the
train set is 20,000 trials, not 200. Budget it like the hierarchical joint runs.
--skip-train-acc drops the most expensive pass.

Writes <output-dir>/{model.pt, model_bestacc.pt, metadata.json, results.npz}.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from mtl.flat.joint import (
    build_loss_channels,
    build_model,
    compute_task_ref,
    run_training,
    save_run,
)
from mtl.flat.mar import mar_units
from mtl.log import add_log_level_argument
from mtl.log import setup as setup_logging
from mtl.shared.battery import resolve_battery, single_output_dim
from mtl.shared.scoring import compute_scalar_vector_tol
from mtl.tasks.dataset import MultiTaskDataset, balanced_loaders

logger = logging.getLogger(__name__)


def build_parser():
    """Command-line interface. Separate from main() so the parser can be
    imported and inspected (defaults, docs, tests) without running anything.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--sample-size", type=int, default=200,
                   help="train trials PER TASK (train_hierarchical_joint.py semantics)")
    p.add_argument("--val-size", type=int, default=50, help="validation trials PER TASK")
    p.add_argument("--nonlinear-units", "-P", type=int, default=4,
                   help="P: number of nonlinear (ReLU) units; the remaining M-P units stay linear. NOTE: PLRNN's own third ctor arg is the "
                        "OUTPUT dim, not this.")
    p.add_argument("--hidden-size", "-M", type=int, default=64,
                   help="M: hidden/latent state dimension")
    p.add_argument("--batch-size", type=int, default=64, help="joint runs used 64")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--lr-warmup-epochs", type=int, default=100,
                   help="joint runs ramped the LR over 100 epochs")
    p.add_argument("--patience", type=int, default=300, help="on validation loss (0 = off)")
    p.add_argument("--es-start", type=int, default=300)
    p.add_argument("--eval-interval", type=int, default=10,
                   help="epochs between validation/train metric passes. best_epoch is quantised "
                        "to this grid (as in train_hierarchical_joint.py)")
    p.add_argument("--tau", type=float, default=0.01, help="MAR strength")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tasks", type=str, default="",
                   help="comma-separated task names: train on just this SUBSET of the "
                        "battery (battery order is kept; overrides --leave-out). For the "
                        "nonlinearity-as-routing battery-size sweep.")
    p.add_argument("--leave-out", type=str, default="",
                   help="hold this task out of the battery (LOO), by name")
    p.add_argument("--no-varnorm", dest="varnorm", action="store_false", default=True,
                   help="disable the variance-normalized loss. ON by default")
    p.add_argument("--skip-train-acc", action="store_true",
                   help="skip the train-accuracy pass (the most expensive one)")
    add_log_level_argument(p)
    return p


def main():
    a = build_parser().parse_args()
    setup_logging(a.log_level)

    all_names, names, tasks, left_out = resolve_battery(a)
    n_tasks = len(tasks)
    n_out = single_output_dim(tasks)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M_reg = mar_units(a.hidden_size)

    if left_out is not None:
        logger.info(f"LEAVE-ONE-OUT: holding out '{left_out}' "
                    f"(index {all_names.index(left_out)} of {len(all_names)})")
    logger.info(f"FLAT JOINT (P=0, one-hot task input): {n_tasks} tasks "
                f"| M={a.hidden_size} N={a.nonlinear_units} "
                f"s={a.sample_size}/task val={a.val_size}/task seed={a.seed}")

    # fixed=True on BOTH: trials are drawn once and reused every epoch, so --sample-size
    # really is the training-set size. task_indices is passed explicitly -- left as None,
    # MultiTaskDataset draws np.random.randint, i.e. an UNBALANCED multinomial.
    train_loader, val_loader = balanced_loaders(
        tasks, a.sample_size, a.val_size, a.batch_size, a.seed,
        dataset_cls=MultiTaskDataset,
    )

    input_dim = next(iter(train_loader))[0].shape[-1]
    logger.info(
        f"train {len(train_loader.dataset)} trials / {len(train_loader)} batches | "
        f"val {len(val_loader.dataset)} trials | input_dim={input_dim} "
    )

    model, optimizer = build_model(tasks, n_out, input_dim, a, device)
    task_loss_channels = build_loss_channels(tasks)

    
    scalar_vector_tol = compute_scalar_vector_tol(tasks) or None

    task_ref = None
    if a.varnorm:
        task_ref = compute_task_ref(
            model, train_loader, device, n_tasks, task_loss_channels)
        vals = list(task_ref.values())
        logger.info(f"[LOSS_VARNORM] fixed variance-normalized loss ON | per-task reference "
                    f"L_i(0) for {len(task_ref)} tasks: min={min(vals):.4f} "
                    f"max={max(vals):.4f} (max/min={max(vals) / max(min(vals), 1e-9):.0f}x)")

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    h = run_training(model, optimizer, train_loader, val_loader, device, a, tasks,
                     n_tasks, task_loss_channels, task_ref, M_reg, scalar_vector_tol)

    best = save_run(out, a, model, names, all_names, left_out, h, input_dim, n_out)

    _loss_msg = "n/a" if not best else f"{np.mean(list(best.values())):.2%}"
    logger.info(f"\nmean best-epoch val acc {_loss_msg}  ->  {out}")

if __name__ == "__main__":
    main()
