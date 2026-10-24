"""Joint training of the flat, P-free PLRNN over the whole battery.

One shared A/W/h/C/D for every task, with task identity arriving as a one-hot
appended to the INPUT by `MultiTaskDataset` -- never through a p_vector. The CLI
that drives this lives in `scripts/train_flat_joint.py`; everything here is the
experiment itself.

Matched to the hierarchical joint runs where it can be: balanced fixed datasets,
the variance-normalized loss, and the joint pipeline's own scoring functions.

The one-hot is NOT a learned task embedding -- that would be halfway back to a
p_vector and would muddy the comparison. It is the Yang-2019 choice, and what
`MultiTaskDataset` already produces. Its cost is recorded honestly: at M=64 it adds
64*100 = 6400 parameters to C, written to metadata as `n_task_identity_params`,
against the ~65k the hierarchical arm spends on p2W alone at 16 task features.

This runs its own epoch loop rather than calling
`multi_task_training.rnn_model.train_multitask`, which adds a `fix_loss` term that
BPTT does not have and would make the arms incomparable.
"""

import json
import logging
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam

from mtl import tasks as ft
from mtl.flat.mar import mar_units, regularization_loss
from mtl.flat.plrnn import FlatPLRNN
from mtl.shared.heldout import score_heldout
from mtl.shared.scoring import (
    TAU_SIGMA,
    THRESHOLD,
    compute_accuracies,
    compute_loss,
    compute_per_task_loss,
    compute_scalar_vector_tol,
    mask_pair,
    masked_eval_batches,
    normalized_mse_loss,
    per_task_mse_loss,
    variance_normalized_mse_loss,
)
from mtl.tasks.dataset import MultiTaskDataset

logger = logging.getLogger(__name__)


def build_model(tasks, n_out, input_dim, a, device):
    """The flat network and its optimizer.

    Called after the loaders because `input_dim` is read off a real batch (7 base
    channels + the one-hot), and because the weight init has to consume the RNG at
    the same point it always has.
    """
    model = FlatPLRNN(a.hidden_size, a.nonlinear_units, n_out, input_dim).to(device)
    optimizer = Adam(model.parameters(), lr=a.learning_rate,
                     weight_decay=a.weight_decay)
    return model, optimizer


def build_loss_channels(tasks):
    """Map task_id -> the output channels that task's loss is computed on.

    This dict is the ONE place task_ids changes the objective. A missing key silently
    falls back to "all channels except 0", which would change the loss for every
    channel-restricted task, so it is built from the task objects, not hardcoded.
    """
    return {
        i: t.loss_channels for i, t in enumerate(tasks)
        if getattr(t, "loss_channels", None) is not None
    }


def compute_task_ref(model, loader, device, n_tasks, task_loss_channels):
    """Frozen per-task reference scale L_i(0), mirroring bptt.py:160-188.

    One no-grad pass over the TRAIN loader at initialisation; each task's reference is the
    mean of its per-sample layer-1 losses, which is ~= its target variance. Tasks that
    somehow contribute nothing fall back to 1.0, as in bptt.
    """
    model.eval()
    acc = {i: [] for i in range(n_tasks)}
    for mo, mt, masks, task_ids in masked_eval_batches(model, loader, device):
        for tid, v in per_task_mse_loss(
            mo, mt, masks, task_ids, n_tasks, task_loss_channels
        ).items():
            if v > 0:
                acc[tid].append(v)
    model.train()
    return {i: (float(np.mean(acc[i])) if acc[i] else 1.0) for i in range(n_tasks)}


def task_accuracies(model, loader, device, n_tasks, tasks, scalar_vector_tol):
    """Per-task accuracy as a plain list in battery order, via the joint scorer."""
    accs = compute_accuracies(
        model, loader, device, n_tasks, task_thresholds=None, tasks=tasks,
        scalar_vector_tol=scalar_vector_tol,
    )
    return [float(accs[i]) for i in range(n_tasks)]


def train_one_epoch(model, loader, optimizer, device, n_tasks, task_loss_channels,
                    task_ref, tau, M_reg, grad_clip):
    """One pass over the training set.

    Returns (mean_batch_loss, per_task_mean_loss, n_nonfinite). A non-finite loss or
    gradient norm skips the batch so NaN cannot reach the weights; the caller cools
    the LR once per epoch that exploded. Per-task train loss is read off the outputs
    already computed, so it costs no extra forward pass.
    """
    model.train()
    epoch_loss, n_batches, n_nonfinite = 0.0, 0, 0
    ep_task_loss = {i: [] for i in range(n_tasks)}

    for inputs, targets, masks, task_ids in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        masks, task_ids = masks.to(device), task_ids.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, task_ids)
        masked_outputs, masked_targets = mask_pair(outputs, targets, masks)

        if task_ref is not None:
            task_loss = variance_normalized_mse_loss(
                masked_outputs, masked_targets, masks, task_ids,
                task_loss_channels, task_ref)
        else:
            task_loss = normalized_mse_loss(
                masked_outputs, masked_targets, masks, task_ids, task_loss_channels)
        loss = task_loss + regularization_loss(model, tau, M_reg)

        if not torch.isfinite(loss):
            n_nonfinite += 1
            optimizer.zero_grad()
            continue

        loss.backward()
        if grad_clip and grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(gn):
                n_nonfinite += 1
                optimizer.zero_grad()
                continue
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            for tid, v in per_task_mse_loss(
                masked_outputs.detach(), masked_targets.detach(), masks, task_ids,
                n_tasks, task_loss_channels,
            ).items():
                if v > 0:
                    ep_task_loss[tid].append(v)

    per_task = [float(np.mean(ep_task_loss[i])) if ep_task_loss[i] else 0.0
                for i in range(n_tasks)]
    return epoch_loss / max(n_batches, 1), per_task, n_nonfinite


def evaluate(model, loader, device, tasks, n_tasks, task_loss_channels, task_ref,
             tau, M_reg, scalar_vector_tol):
    """One metric pass: (total loss, per-task loss, per-task accuracy)."""
    tl = compute_loss(model, loader, device, tau, M_reg, task_loss_channels, task_ref)
    per_task_tl = compute_per_task_loss(
        model, loader, device, n_tasks, task_loss_channels)
    accs = task_accuracies(model, loader, device, n_tasks, tasks, scalar_vector_tol)
    return float(tl), [float(per_task_tl[i]) for i in range(n_tasks)], accs


def run_training(model, optimizer, train_loader, val_loader, device, a, tasks,
                 n_tasks, task_loss_channels, task_ref, M_reg, scalar_vector_tol):
    """The epoch loop. Returns the history dict that save_run() consumes.

    Test metrics run on the --eval-interval cadence, NOT per epoch: the validation set is
    n_tasks * --val-size trials (5000 at defaults), so a per-epoch pass is not
    affordable. best_val_loss_epoch is therefore quantised to that grid, as in
    mtl.hierarchical.joint. Early stopping watches VALIDATION LOSS and only arms after
    --es-start (BPTT semantics).

    Both checkpoints are held in memory and written once by save_run(), rather than
    streamed to disk as BPTT does -- the flat model is small enough that two state
    dicts cost less than the repeated writes.
    """
    base_lr, warm = a.learning_rate, a.lr_warmup_epochs
    lr_scale, n_nonfinite = 1.0, 0

    h = {
        "loss_history": [],
        "train_task_loss_hist": [],      # per EPOCH, so longer than the eval-grid lists
        "eval_epochs": [],
        "val_loss_history": [],
        "val_acc_hist": [],             # each entry: n_tasks floats
        "val_tl_hist": [],
        "train_eval_epochs": [],
        "train_acc_hist": [],
        "best_state": None,
        "best_acc_state": None,
        "best_val_acc_epoch": None,
        "stopped_early": False,
    }
    best_val_loss, best_acc_mean = float("inf"), -float("inf")

    for epoch in range(a.epochs):
        factor = min(1.0, (epoch + 1) / warm) if warm and warm > 0 else 1.0
        for grp in optimizer.param_groups:
            grp["lr"] = base_lr * factor * lr_scale

        mean_loss, per_task_loss, epoch_nonfinite = train_one_epoch(
            model, train_loader, optimizer, device, n_tasks, task_loss_channels,
            task_ref, a.tau, M_reg, a.grad_clip)

        # halve the LR once per epoch that exploded, floor 1e-3 (bptt.py:349-353)
        if epoch_nonfinite > 0:
            n_nonfinite += epoch_nonfinite
            lr_scale = max(lr_scale * 0.5, 1e-3)
            logger.info(f"[stability] epoch {epoch + 1}: {epoch_nonfinite} non-finite "
                        f"batch(es) skipped; lr_scale -> {lr_scale:.4g}")

        h["loss_history"].append(mean_loss)
        h["train_task_loss_hist"].append(per_task_loss)

        if (epoch + 1) % a.eval_interval and (epoch + 1) != a.epochs:
            continue

        tl, per_task_tl, acc_vec = evaluate(
            model, val_loader, device, tasks, n_tasks, task_loss_channels,
            task_ref, a.tau, M_reg, scalar_vector_tol)
        h["eval_epochs"].append(epoch + 1)
        h["val_loss_history"].append(tl)
        h["val_tl_hist"].append(per_task_tl)
        h["val_acc_hist"].append(acc_vec)
        mean_acc = float(np.mean(acc_vec))

        if tl < best_val_loss:
            best_val_loss = tl
            h["best_state"] = {k: v.detach().clone()
                               for k, v in model.state_dict().items()}
        if mean_acc > best_acc_mean:
            best_acc_mean = mean_acc
            h["best_val_acc_epoch"] = epoch + 1
            h["best_acc_state"] = {k: v.detach().clone()
                                   for k, v in model.state_dict().items()}

        msg = (f"epoch {epoch + 1}/{a.epochs} | train {mean_loss:.4f} | "
               f"val {tl:.4f} | mean acc {mean_acc:.2%}")
        if not a.skip_train_acc:
            h["train_eval_epochs"].append(epoch + 1)
            h["train_acc_hist"].append(task_accuracies(
                model, train_loader, device, n_tasks, tasks, scalar_vector_tol))
            msg += f" | train acc {np.mean(h['train_acc_hist'][-1]):.2%}"
        logger.info(msg)

        if epoch + 1 > a.es_start:
            since_best = (epoch + 1) - h["eval_epochs"][
                int(np.argmin(h["val_loss_history"]))]
            if a.patience and since_best >= a.patience:
                h["stopped_early"] = True
                logger.info(f"early stop: {since_best} epochs since best validation loss")
                break

    h["n_nonfinite"] = n_nonfinite
    h["lr_scale"] = lr_scale
    return h


def select_best(names, eval_epochs, val_loss_history, val_acc_hist, val_tl_hist):
    """Pick the reported epoch and read the per-task numbers there.

    The reported run is the MIN-LOSS evaluation, the same criterion early stopping uses,
    and model.pt holds those weights. The best mean accuracy over the trajectory is
    returned alongside for comparison; model_bestacc.pt holds those weights.

    Both are selected on the validation set, which also drives early stopping, so the
    reported numbers are a maximum over the trajectory rather than a clean held-out
    estimate. There is no separate validation split.

    Evaluations sit on the --eval-interval grid, so the returned best_val_loss_epoch is the real
    epoch number out of `eval_epochs`, not an index into the history lists.
    """
    n = len(names)
    val_lh = np.asarray(val_loss_history, dtype=float)
    best_idx = int(np.argmin(val_lh)) if val_lh.size else -1

    if best_idx >= 0:
        best_val = {names[i]: float(val_acc_hist[best_idx][i]) for i in range(n)}
        best_val_loss = {names[i]: float(val_tl_hist[best_idx][i]) for i in range(n)}
        best_val_loss_epoch = int(eval_epochs[best_idx])
    else:
        best_val, best_val_loss, best_val_loss_epoch = {}, {}, None

    if val_acc_hist:
        acc_traj = np.asarray(val_acc_hist, dtype=float)   # (n_evals, n_tasks)
        mean_best_acc = float(np.max(acc_traj.mean(axis=1)))
    else:
        mean_best_acc = None
    return best_val, best_val_loss, best_val_loss_epoch, mean_best_acc


def save_curves(out, n_tasks, h):
    """Write results.npz: the full per-epoch curves, per task where applicable.

    Per-task arrays are (n_tasks, n_evals) to match mtl.hierarchical.joint's
    orientation.
    """
    def stack(hist):
        return (np.asarray(hist, dtype=float).T if hist
                else np.zeros((n_tasks, 0), dtype=float))

    np.savez(
        out / "results.npz",
        loss_history=np.asarray(h["loss_history"], dtype=float),
        val_loss_history=np.asarray(h["val_loss_history"], dtype=float),
        eval_epochs=np.asarray(h["eval_epochs"], dtype=int),
        val_task_accuracies=stack(h["val_acc_hist"]),
        val_task_losses=stack(h["val_tl_hist"]),
        # train accuracy is on the eval cadence and EMPTY under --skip-train-acc; index it
        # with train_eval_epochs. train_task_losses is per EPOCH, so it is longer.
        train_task_accuracies=stack(h["train_acc_hist"]),
        train_eval_epochs=np.asarray(h["train_eval_epochs"], dtype=int),
        train_task_losses=stack(h["train_task_loss_hist"]),
    )


def save_run(out, a, model, names, all_names, left_out, h, input_dim, n_out):
    """Write the three artefacts of one run and report the selection.

    `h` is run_training()'s history dict, kept whole so this takes an argument list a
    human can read. Writes model_bestacc.pt (max mean validation accuracy), model.pt (min
    validation loss -- the weights the analysis notebooks load), results.npz and
    metadata.json, the record rank_results.py and every notebook read. Returns the
    per-task accuracy at the selected epoch, for the caller's closing log line.

    Loading best_state into `model` before saving is why this takes the live model:
    the object is left holding the reported weights, not the last-epoch ones.
    """
    n_tasks = len(names)

    if h["best_acc_state"] is not None:
        torch.save(h["best_acc_state"], out / "model_bestacc.pt")
    if h["best_state"] is not None:
        model.load_state_dict(h["best_state"])
        selection = "best_val_loss"
    else:
        selection = "final"
    torch.save(model.state_dict(), out / "model.pt")

    # HELD-OUT: a third split, never seen by training, early stopping or selection.
    # `model` already holds the reported weights (restored just above).
    tasks = [ft.FOUNDATION_TASKS_MAP[n] for n in names]
    heldout = score_heldout(
        model, tasks, names, a.val_size, a.batch_size, a.seed,
        next(model.parameters()).device, dataset_cls=MultiTaskDataset,
        scalar_vector_tol=compute_scalar_vector_tol(tasks) or None)
    logger.info(f"[HELD-OUT] mean acc {heldout['mean_test_accuracy']:.2%} "
                f"on {a.val_size} fresh trials/task (seed {heldout['test_seed']})")

    best_val, best_val_loss_pt, best_val_loss_epoch, mean_best_acc_selected = select_best(
        names, h["eval_epochs"], h["val_loss_history"], h["val_acc_hist"],
        h["val_tl_hist"])

    _acc_msg = ("n/a" if mean_best_acc_selected is None
                else f"{mean_best_acc_selected:.2%}")
    _loss_msg = "n/a" if not best_val else f"{np.mean(list(best_val.values())):.2%}"
    logger.info(f"[SELECT] min-loss epoch {best_val_loss_epoch} -> mean acc {_loss_msg}   |   "
                f"max-acc epoch {h['best_val_acc_epoch']} -> mean acc {_acc_msg}")

    save_curves(out, n_tasks, h)

    meta = {
        "n_tasks": n_tasks,
        "task_names": names,
        "seed": a.seed,
        "left_out_task": left_out,
        "left_out_index": all_names.index(left_out) if left_out is not None else None,
        # what makes this run the flat arm
        "parameterization": "flat_shared_joint",
        "num_individual_params": 0,
        "task_onehot": True,
        "init": "plrnn_native",
        "input_dim": input_dim,
        "n_task_identity_params": int(a.hidden_size * n_tasks),
        "n_shared_params": int(sum(q.numel() for q in model.parameters())),
        # config
        "nonlinear_units": a.nonlinear_units,
        "hidden_size": a.hidden_size,
        "output_size": n_out,
        "sample_size": a.sample_size,
        "val_size": a.val_size,
        "batch_size": a.batch_size,
        "num_epochs": a.epochs,
        "epochs_run": len(h["loss_history"]),
        "stopped_early": h["stopped_early"],
        "learning_rate": a.learning_rate,
        "lr_warmup_epochs": a.lr_warmup_epochs,
        "weight_decay": a.weight_decay,
        "grad_clip": a.grad_clip,
        "tau": a.tau,
        "M_reg": mar_units(a.hidden_size),
        "eval_interval": a.eval_interval,
        "early_stopping_patience": a.patience,
        "early_stopping_start": a.es_start,
        "loss_varnorm": bool(a.varnorm),
        "n_nonfinite_batches": h["n_nonfinite"],
        "final_lr_scale": h["lr_scale"],
        # scoring
        "accuracy_threshold_deg": float(THRESHOLD * 180.0 / np.pi),
        "scoring": "hierachical_model_task.rnn_model.compute_accuracies "
                   "(fixation gate on)",
        # constants since --legacy-accuracy was dropped; kept so a run still records
        # which metric produced its numbers (analysis_NPsweep.ipynb reads this field,
        # and runs from before the tau*sigma switch have it False or absent)
        "tau_sigma_metric": True,
        "tau_sigma": TAU_SIGMA,
        # results, both selections
        "checkpoint_selection": selection,
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_val_accuracy_per_task": best_val,
        "mean_best_val_accuracy": (float(np.mean(list(best_val.values())))
                                    if best_val else None),
        "best_val_loss_per_task": best_val_loss_pt,
        "mean_best_val_loss": (float(np.mean(list(best_val_loss_pt.values())))
                                if best_val_loss_pt else None),
        "best_val_acc_epoch": h["best_val_acc_epoch"],
        "mean_best_acc_selected_val_accuracy": mean_best_acc_selected,
        **heldout,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return best_val
