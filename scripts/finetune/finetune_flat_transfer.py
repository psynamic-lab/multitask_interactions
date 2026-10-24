#!/usr/bin/env python
"""Task-to-task fine-tuning of a trained FLAT (P-free) PLRNN.

Reads ONE individual model's weights (a `model.pt` from the per-task runs produced by
multi_task_training/scripts/train_indiv_models.py) and fine-tunes it on a SECOND task.

This is `multi_task_training/scripts/transfer_learning.py` with its first stage replaced:
that script trains task 1 from scratch in-process and has no way to start from saved
weights, and it is pinned to a different setup anyway (M=16, N=3, an 11-task registry with
its own durations, MultiTaskDataset's one-hot task channel, and the old pre-fixation-gate
scorer). Here the source model is LOADED, the target task comes from the 100-task
foundation battery, and scoring goes through the joint pipeline's own code.

WHY THE WEIGHTS TRANSFER WITHOUT SURGERY:
    every task in FOUNDATION_TASKS_MAP is output_dim=5 over BASE_INPUT_DIM=7 channels, so
    A (L,), W (M, M), h (M), C (M, 7), D (5, M) are shape-compatible across tasks. M and L
    are read off the checkpoint itself (W.shape[0], A.shape[0]) rather than passed in.

MATCHED TO THE INDIVIDUAL RUNS (train_indiv_models.py) so the fine-tuned numbers are
directly comparable to the from-scratch ones: the same training loop (non-finite guard with
adaptive lr halving, LR warmup, early stopping on VALIDATION LOSS), the same defaults (batch 32,
lr 5e-4, weight_decay 1e-4, grad_clip 10, tau 0.01, M_reg=M//2), fixed train AND validation sets,
and the joint scorer -- 36 deg angular threshold, Yang-2019 fixation gate, and the tau*sigma
metric for the 7 continuous-readout tasks (--legacy-accuracy restores the fixed 0.1).

--sample-size and --seed default to the SOURCE run's own values, so a fine-tune inherits the
individual run it starts from rather than a hardcoded default.

Examples:
    # full fine-tune, with the from-scratch control that makes "transfer" measurable
    python foundation_model/finetune_flat_transfer.py \\
        --source-checkpoint data/20260801_individual_runs_v2/N4_s50/DelayPro/seed_0/model.pt \\
        --target-task DelayAnti --scratch-control

    # hold the recurrent dynamics fixed; adapt only the input/readout maps
    python foundation_model/finetune_flat_transfer.py \\
        --source-checkpoint .../model.pt --target-task DelayAnti --freeze A,W,h
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mtl import tasks as ft
from mtl.flat.plrnn import FlatPLRNN
from mtl.flat.transfer import (
    PARAM_NAMES,
    epochs_ratio,
    epochs_to,
    max_acc_ratio,
    ratio_at_baseline_best,
    read_baseline,
    read_source,
    save_arm,
    train_arm,
)
from mtl.log import add_log_level_argument
from mtl.log import setup as setup_logging
from mtl.tasks.dataset import HierarchicalTasksDataset, collate_fn

logger = logging.getLogger(__name__)


def build_parser():
    """Command-line interface. Separate from main() so the parser can be
    imported and inspected (defaults, docs, tests) without running anything.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source-checkpoint", type=str, required=True,
                   help="model.pt to start from (a per-task individual run).")
    p.add_argument("--target-task", type=str, required=True,
                   help="Task to fine-tune on (a FOUNDATION_TASKS_MAP name).")
    p.add_argument("--source-task", type=str, default=None,
                   help="Label only; read from the checkpoint's metadata.json if omitted.")
    p.add_argument("--freeze", type=str, default="",
                   help="Comma list of parameters to hold fixed, from A,W,h,C,D "
                        "(default: none = full fine-tune).")
    p.add_argument("--scratch-control", action="store_true",
                   help="Also train a fresh model on the target task, for the "
                        "transfer-vs-scratch comparison.")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Training trials. Default: the SOURCE run's sample_size, so the "
                        "fine-tune matches the individual run it starts from (50 if the "
                        "checkpoint has no metadata.json).")
    p.add_argument("--val-size", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--lr-warmup-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=150)
    p.add_argument("--es-start", type=int, default=0)
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.8, 0.9],
                   help="Accuracy levels for the epochs-to-threshold comparison. Reported "
                        "per arm as {level: epoch}. Crossing is >= (see mtl.flat.transfer.epochs_to).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1,
                   help="torch threads. Default 1, matching train_indiv_models.py: BLAS "
                        "reduction order depends on it, so a different value makes the "
                        "scratch control drift away from the stored individual runs.")
    p.add_argument("--baseline-root", type=str, default=None,
                   help="Root of the from-scratch individual runs (e.g. "
                        "data/20260801_flat_individual). When given, the TARGET task's own "
                        "run at the matching N/sample/seed supplies the denominator for "
                        "epochs_ratio_vs_scratch = finetuned / from-scratch (<1 = faster).")
    p.add_argument("--output-dir", type=str, default=None)
    add_log_level_argument(p)
    return p


def main():
    a = build_parser().parse_args()
    setup_logging(a.log_level)

    if a.target_task not in ft.FOUNDATION_TASKS_MAP:
        raise SystemExit(
            f"--target-task '{a.target_task}' is not in the battery. "
            f"Known: {list(ft.FOUNDATION_TASKS_MAP)[:6]}..."
        )
    frozen = {s.strip() for s in a.freeze.split(",") if s.strip()}
    unknown = frozen - set(PARAM_NAMES)
    if unknown:
        raise SystemExit(f"--freeze: unknown parameter(s) {sorted(unknown)}. "
                         f"Choose from {list(PARAM_NAMES)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = ft.FOUNDATION_TASKS_MAP[a.target_task]
    n_out = task.output_dim

    # Read the checkpoint BEFORE seeding: torch.load does not touch the RNG, and we need
    # M / L to build the scratch control at the right point in the RNG stream (below).
    state, info = read_source(a.source_checkpoint)
    M, L = info["hidden_size"], info["nonlinear_units"]
    source_task = a.source_task or info.get("source_task") or "unknown"

    # --sample-size / --seed default to the SOURCE run's values, so a fine-tune inherits
    # the individual run it starts from and lands on the same footing as the target task's
    # own individual run at that seed. Explicit flags still win.
    if a.sample_size is None:
        a.sample_size = info.get("source_sample_size") or 50
    if a.seed is None:
        a.seed = info["source_seed"] if info.get("source_seed") is not None else 0

    cfg = {
        "sample_size": a.sample_size, "val_size": a.val_size,
        "batch_size": a.batch_size, "epochs": a.epochs,
        "learning_rate": a.learning_rate, "weight_decay": a.weight_decay,
        "grad_clip": a.grad_clip, "tau": a.tau,
        "lr_warmup_epochs": a.lr_warmup_epochs, "patience": a.patience,
        "es_start": a.es_start, "eval_interval": a.eval_interval,
        "seed": a.seed,
        "thresholds": a.thresholds,
    }

    # ---- data: the target task only, fixed train AND validation sets ----------------------
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    tasks = [task]
    train_ds = HierarchicalTasksDataset(
        tasks, n_trials=a.sample_size, task_indices=[0] * a.sample_size, fixed=True)
    val_ds = HierarchicalTasksDataset(
        tasks, n_trials=a.val_size, task_indices=[0] * a.val_size, fixed=True)

    def loaders():
        """Fresh loaders per arm, so both arms see the same trials in the same order.

        The peek below is not just for input_dim: train_indiv_models.py reads it the same
        way, and building that iterator draws one permutation from `g`, so its training
        starts from permutation #2. Replicating the peek keeps the batch order identical
        to the stored individual runs -- without it the scratch control trains on a
        differently-ordered stream and drifts away from them.
        """
        g = torch.Generator().manual_seed(a.seed)
        trl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                         collate_fn=collate_fn, generator=g)
        tel = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                         collate_fn=collate_fn)
        return trl, tel, next(iter(trl))[0].shape[-1]

    train_loader, val_loader, input_dim = loaders()

    # The scratch control is constructed HERE -- same RNG point as train_indiv_models.py
    # (seed -> datasets -> loaders -> input_dim -> model), so with matching --seed,
    # --sample-size and --val-size it is bit-identical to a real individual run. It must
    # come before the transfer model, whose own native init would otherwise consume the
    # draw (that init is discarded by load_state_dict, but the RNG advance is not).
    scratch_model = (
        FlatPLRNN(M, L, n_out, input_dim).to(device) if a.scratch_control else None
    )

    # ---- source model ---------------------------------------------------------------
    model = FlatPLRNN(M, L, n_out, input_dim)
    model.load_state_dict(state)  # strict: shapes must match exactly
    model = model.to(device)
    print(f"source: {source_task} (M={M} N={L}) <- {a.source_checkpoint}")
    print(f"target: {a.target_task} | freeze: {sorted(frozen) or 'none'} "
          f"| sample {a.sample_size} | device {device}")

    out = Path(a.output_dir) if a.output_dir else (
        Path(a.source_checkpoint).parent / f"finetune_to_{a.target_task}")
    model_dims = (M, L, n_out, input_dim)
    shared_extra = {
        "source_checkpoint": str(Path(a.source_checkpoint).resolve()),
        "source_task": source_task,
        "target_task": a.target_task,
    }

    # ---- arm 1: transfer ------------------------------------------------------------
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    transfer = train_arm(model, task, train_loader, val_loader, cfg, device, frozen,
                         "transfer")
    t_meta = save_arm(out / "transfer", transfer, a.target_task, cfg, model_dims,
                      {**shared_extra, "init": "finetuned_from_checkpoint",
                       "frozen_params": sorted(frozen),
                       "frozen_max_drift": transfer["moved_frozen"]})

    summary = {
        "source_task": source_task,
        "target_task": a.target_task,
        "source_checkpoint": shared_extra["source_checkpoint"],
        "frozen_params": sorted(frozen),
        "thresholds": a.thresholds,
        "transfer": {
            "best_val_acc": t_meta["best_val_acc"],
            "best_val_loss_epoch": t_meta["best_val_loss_epoch"],
            "epochs_run": t_meta["epochs_run"],
            "epochs_to_threshold": epochs_to(
                transfer["val_acc_hist"], transfer["eval_epochs"], a.thresholds),
        },
        "tau_sigma_metric": True,
        "seed": a.seed,
        "sample_size": a.sample_size,
        "timestamp": datetime.now().astimezone().isoformat(),
    }

    # ---- speed-up vs the target task's own from-scratch run --------------------------
    # The denominator is the TARGET's individual run, not the source's: the question is
    # how much the source initialisation changed the time to learn THIS task.
    if a.baseline_root:
        baseline = read_baseline(a.baseline_root, a.target_task, L, a.sample_size,
                                 a.seed, a.thresholds)
        summary["baseline_from_scratch"] = baseline
        summary["epochs_ratio_vs_scratch"] = epochs_ratio(
            summary["transfer"]["epochs_to_threshold"],
            baseline["epochs_to_threshold"] if baseline else None,
        )
        # Fallback, filled in only for the levels the baseline never reached.
        summary["ratio_at_baseline_best"] = ratio_at_baseline_best(
            transfer["val_acc_hist"], transfer["eval_epochs"], baseline, a.thresholds)
        # Accuracy (not epochs) ratio, always computed. >1 = fine-tuning reached higher.
        summary["max_acc_ratio_vs_scratch"] = max_acc_ratio(
            transfer["val_acc_hist"], baseline)
        if baseline is None:
            print(f"  WARNING: no from-scratch baseline for {a.target_task} under "
                  f"{a.baseline_root} (N{L}_s{a.sample_size}/seed_{a.seed}) -- "
                  f"epochs_ratio_vs_scratch is null")

    # ---- arm 2: from-scratch control ------------------------------------------------
    if a.scratch_control:
        train_loader, val_loader, _ = loaders()  # identical trials and order
        scratch = train_arm(scratch_model, task, train_loader, val_loader, cfg, device,
                            set(), "scratch")
        s_meta = save_arm(out / "scratch", scratch, a.target_task, cfg, model_dims,
                          {**shared_extra, "init": "plrnn_native",
                           "frozen_params": [], "frozen_max_drift": 0.0})
        summary["scratch"] = {
            "best_val_acc": s_meta["best_val_acc"],
            "best_val_loss_epoch": s_meta["best_val_loss_epoch"],
            "epochs_run": s_meta["epochs_run"],
            "epochs_to_threshold": epochs_to(
                scratch["val_acc_hist"], scratch["eval_epochs"], a.thresholds),
        }

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{source_task} -> {a.target_task}")
    print(f"  transfer: best acc {t_meta['best_val_acc']:.1%} @ epoch "
          f"{t_meta['best_val_loss_epoch']} | epochs to "
          f"{summary['transfer']['epochs_to_threshold']}")
    if a.scratch_control:
        print(f"  scratch : best acc {s_meta['best_val_acc']:.1%} @ epoch "
              f"{s_meta['best_val_loss_epoch']} | epochs to "
              f"{summary['scratch']['epochs_to_threshold']}")
    if a.baseline_root and summary.get("baseline_from_scratch"):
        print(f"  baseline: {summary['baseline_from_scratch']['epochs_to_threshold']} "
              f"(from-scratch {a.target_task})")
        print(f"  ratio   : {summary['epochs_ratio_vs_scratch']}  (<1 = faster than scratch)")
        mar = summary["max_acc_ratio_vs_scratch"]
        print(f"  max acc : {mar['finetuned_max_acc']} vs {mar['baseline_max_acc']} "
              f"-> ratio {mar['ratio']}  (>1 = higher than scratch)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
