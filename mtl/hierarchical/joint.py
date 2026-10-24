"""Joint training of the hierarchical AL-RNN over the whole battery.

Task identity lives in the per-task `p_vector`, which is projected into A, W, h,
C and D; the inputs stay the raw 7 channels. The epoch loop itself is `BPTT`, so
this module is the configuration and bookkeeping around it. The CLI lives in
`scripts/train_hierarchical_joint.py`.
"""

import json
import logging
from datetime import datetime

import numpy as np
import torch

from mtl import tasks as ft
from mtl.hierarchical.config import HierarchicalModelConfig as HiearchicalModelConfig
from mtl.hierarchical.projection import parse_hierarchisation
from mtl.shared.heldout import score_heldout
from mtl.shared.scoring import compute_scalar_vector_tol

logger = logging.getLogger(__name__)


def apply_init(model, mode, gain):
    """Re-initialize the recurrent projection p2W (in place).

    mode='gaussian'   -> xavier-normal (Gaussian) on the whole (dp, M, M) tensor.
    mode='orthogonal' -> each (M, M) slice orthonormal, scaled by `gain`
                         (no dead directions per slice; the effective per-subject
                         W = p_vector @ p2W is a weighted sum, so not itself
                         orthogonal, but better-conditioned).
    mode='default'    -> leave the model's own xavier-uniform init untouched.
    """
    if not hasattr(model, "p2W"):
        raise ValueError("--init-mode re-initialises p2W, but W is shared in this hierarchisation")
    with torch.no_grad():
        p2W = model.p2W  # (dp, M, M)
        if mode == "gaussian":
            torch.nn.init.xavier_normal_(p2W, gain=gain)
        elif mode == "orthogonal":
            for d in range(p2W.shape[0]):
                torch.nn.init.orthogonal_(p2W[d], gain=gain)


def build_model_config(tasks, a):
    """Build the BPTT config from the parsed CLI arguments.

    Every field below is a constructor field of HiearchicalModelConfig with a
    default, so a flag that is never copied here silently falls back to that
    default rather than failing -- add new flags to this function, not to main().

    Two steps are deliberate and not idempotent, so call this exactly once:
    `learning_rate` is replaced by the (shared, individual) pair that BPTT expects,
    and tf_alpha_end falls back to tf_alpha_start when it was left unset.
    """
    args = HiearchicalModelConfig(tasks=tasks)
    args.nonlinear_units = a.nonlinear_units
    args.hidden_size = a.hidden_size
    args.num_individual_params = a.num_individual_params
    args.num_epochs = a.num_epochs
    args.early_stopping_patience = a.early_stopping_patience
    args.early_stopping_start = a.early_stopping_start
    args.grad_clip = a.grad_clip
    args.weight_decay = a.weight_decay
    args.a_clamp = a.a_clamp
    args.hierarchisation = parse_hierarchisation(a.hierarchisation)
    args.lr_warmup_epochs = a.lr_warmup_epochs
    args.w_init_gain = a.w_init_gain
    if a.learning_rate is not None:
        args.learning_rate = a.learning_rate
        args.individual_learning_rate = a.learning_rate
    args.varnorm = a.varnorm  # bptt.py reads this; unset would fall back to LOSS_VARNORM
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.learning_rate = (args.learning_rate, args.individual_learning_rate)
    if args.tf_alpha_end is None:
        args.tf_alpha_end = args.tf_alpha_start
    return args


def select_best(names, val_acc, val_tl, val_loss_hist):
    """Pick the reported epoch and read the per-task numbers there.

    The reported run is the MIN-LOSS epoch, the same criterion early stopping uses,
    and model.pt holds those weights. The best mean accuracy over the trajectory is
    returned alongside for comparison; model_bestacc.pt holds those weights.

    Both are selected on the validation set, which also drives early stopping, so the
    reported numbers are a maximum over the trajectory rather than a clean held-out
    estimate. There is no separate validation split.
    """
    n = len(names)
    best_idx = int(np.argmin(val_loss_hist)) if val_loss_hist else -1
    best_val = {names[i]: float(val_acc[i][best_idx]) for i in range(n)}
    best_val_loss = {names[i]: float(val_tl[i][best_idx]) for i in range(n)}
    if val_acc and n:
        mean_acc_traj = np.mean([val_acc[i] for i in range(n)], axis=0)
        mean_best_acc = float(np.max(mean_acc_traj))
    else:
        mean_best_acc = None
    return best_val, best_val_loss, mean_best_acc


def save_curves(out, n_tasks, loss_hist, val_loss_hist, train_acc, val_acc,
                train_tl, val_tl):
    """Write results.npz: the full per-epoch curves, per task where applicable."""
    def stack(hist):
        return np.array([hist[i] for i in range(n_tasks)])

    np.savez(
        out / "results.npz",
        loss_history=np.array(loss_hist),
        val_loss_history=np.array(val_loss_hist),
        train_task_accuracies=stack(train_acc),
        val_task_accuracies=stack(val_acc),
        train_task_losses=stack(train_tl),
        val_task_losses=stack(val_tl),
    )


def save_run(out, a, alg, names, all_names, left_out, curves):
    """Write the three artefacts of one run and report the selection.

    `curves` is alg.train()'s return value, kept whole so this takes an argument
    list a human can read. Writes model.pt (final weights; model_bestacc.pt was
    already written during training), results.npz, and metadata.json -- the record
    the sweep scripts and every analysis notebook read. Returns the per-task
    accuracy at the selected epoch, for the caller's closing log line.

    BPTT restores the best-by-validation-loss weights into `alg.model` before
    returning (bptt.py:490), so the held-out split below scores the SELECTED
    model, which is the point of having it.
    """
    loss_hist, train_acc, val_loss_hist, val_acc, train_tl, val_tl, best_val_loss_epoch, _ = curves
    n_tasks = len(names)

    torch.save(alg.model.state_dict(), out / "model.pt")

    # HELD-OUT: a third split, never seen by training, early stopping or selection.
    tasks = [ft.FOUNDATION_TASKS_MAP[n] for n in names]
    heldout = score_heldout(
        alg.model, tasks, names, a.val_size, a.batch_size, a.seed,
        next(alg.model.parameters()).device,
        scalar_vector_tol=compute_scalar_vector_tol(tasks) or None)
    logger.info(f"[HELD-OUT] mean acc {heldout['mean_test_accuracy']:.2%} "
                f"on {a.val_size} fresh trials/task (seed {heldout['test_seed']})")

    best_val, best_val_loss, mean_best_acc_selected = select_best(
        names, val_acc, val_tl, val_loss_hist
    )
    # BPTT sets `best_acc_epoch` on itself (bptt.py:487); that attribute lives in
    # hierachical_model_task, outside this package, so the name stays as it is.
    best_val_acc_epoch = getattr(alg, "best_acc_epoch", None)
    _acc_msg = (
        "n/a" if mean_best_acc_selected is None else f"{mean_best_acc_selected:.2%}"
    )
    logger.info(
        f"[SELECT] min-loss epoch {best_val_loss_epoch} -> mean acc "
        f"{np.mean(list(best_val.values())):.2%}   |   max-acc epoch {best_val_acc_epoch} "
        f"-> mean acc {_acc_msg}",
    )
    save_curves(out, n_tasks, loss_hist, val_loss_hist, train_acc, val_acc,
                train_tl, val_tl)
    with open(out / "metadata.json", "w") as f:
        json.dump(
            {
                "n_tasks": n_tasks,
                "task_names": names,
                "seed": a.seed,
                "left_out_task": left_out,
                "left_out_index": (
                    all_names.index(left_out) if left_out is not None else None
                ),
                "resumed_from": a.resume or None,
                "num_epochs": a.num_epochs,
                "best_val_loss_epoch": best_val_loss_epoch,
                "nonlinear_units": a.nonlinear_units,
                "hidden_size": a.hidden_size,
                "num_individual_params": a.num_individual_params,
                "hierarchisation": parse_hierarchisation(a.hierarchisation),
                "loss_varnorm": bool(a.varnorm),
                "batch_size": a.batch_size,
                "learning_rate": a.learning_rate,
                "weight_decay": a.weight_decay,
                "a_clamp": a.a_clamp,
                "init_mode": a.init_mode,
                "init_gain": a.init_gain,
                "sample_size": a.sample_size,
                "best_val_accuracy_per_task": best_val,
                "mean_best_val_accuracy": float(np.mean(list(best_val.values()))),
                "best_val_loss_per_task": best_val_loss,
                "mean_best_val_loss": float(np.mean(list(best_val_loss.values()))),
                "best_val_acc_epoch": best_val_acc_epoch,
                "mean_best_acc_selected_val_accuracy": mean_best_acc_selected,
                **heldout,
                "timestamp": datetime.now().astimezone().isoformat(),
            },
            f,
            indent=2,
        )
    return best_val
