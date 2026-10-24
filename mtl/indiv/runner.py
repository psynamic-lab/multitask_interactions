"""Independent per-task training of the flat PLRNN.

One model per task: no hierarchisation, no p_vector, no individual parameters,
and no task one-hot in the input either, since a single-task model needs no task
cue. The network is `mtl.flat`; only the training procedure is specific here.
The CLI, and the job grid it fans out over, live in
`scripts/train_flat_indiv.py`.

NO VARIANCE-NORMALIZED LOSS, unlike the two joint arms. Varnorm divides each
task's loss by its own frozen L_i(0) so large-target tasks do not dominate a
shared objective; with one task per model there is nothing to balance against, so
the division would be by a single constant. Left out rather than made a no-op.

THE WEIGHT INITIALISATION CANNOT BE MATCHED to the joint runs, in either direction.
The joint model has no W to initialise -- its W is einsum(p_vector, p2W), with p2W
xavier-uniform at gain 0.05, so the effective std grows as ~sqrt(dp/3) (measured:
0.0005 at 1 task feature, 0.0029 at 32) and the effective A starts near 0. The flat
PLRNN uses W ~ randn*0.1/sqrt(M) (std 0.0125) and A ~ 0.9, a near-integrator. There
is no feature-dimension-independent flat equivalent, so the native init stands.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

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
    normalized_mse_loss,
)
from mtl.tasks.dataset import HierarchicalTasksDataset, collate_fn

logger = logging.getLogger(__name__)

# Every helper below scores a single task, so the n_tasks argument the joint
# scoring functions take is always 1 and the task is always id 0.
N_TASKS = 1
TASK_ID = 0


def build_config(a):
    """Flatten the parsed arguments into the plain dict each worker receives.

    A dict rather than the Namespace because `jobs` is pickled out to a
    multiprocessing Pool; keeping it to builtins means no import of this module's
    classes is needed on the other side. Anything a worker reads must be listed
    here -- a flag left out is simply invisible to train_one().
    """
    cfg = {k: getattr(a, k) for k in (
        "hidden_size", "val_size", "batch_size", "epochs", "learning_rate",
        "weight_decay", "grad_clip", "lr_warmup_epochs", "patience", "eval_interval",
        "tau", "threads", "skip_existing")}
    cfg["results_dir"] = a.results_dir
    cfg["es_start"] = a.es_start
    return cfg


def build_loaders(tasks, sample_size, cfg, seed):
    """Fixed train/validation loaders for the one task this run trains on.

    fixed=True on BOTH: the trials are drawn once and reused every epoch, so
    --sample-size really is the training-set size. task_indices is all zeros because
    there is a single task. Only the train loader shuffles, and it gets its own
    generator so the shuffle order does not depend on how much RNG the model init
    happened to consume.
    """
    train_ds = HierarchicalTasksDataset(
        tasks, n_trials=sample_size, task_indices=[TASK_ID] * sample_size, fixed=True)
    val_ds = HierarchicalTasksDataset(
        tasks, n_trials=cfg["val_size"], task_indices=[TASK_ID] * cfg["val_size"],
        fixed=True)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collate_fn, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             collate_fn=collate_fn)
    return train_loader, val_loader


def build_model(n_out, input_dim, cfg, device):
    """The flat single-task network and its optimizer."""
    model = FlatPLRNN(cfg["hidden_size"], cfg["L"], n_out, input_dim).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"],
                     weight_decay=cfg["weight_decay"])
    return model, optimizer


def build_loss_channels(task):
    """Map task_id -> the output channels this task's loss is computed on.

    Single-task, so the only key is 0. An empty dict means "all channels except 0",
    which is the right fallback for a task that declares no loss_channels.
    """
    return ({TASK_ID: task.loss_channels}
            if getattr(task, "loss_channels", None) is not None else {})


def task_accuracy(model, loader, device, tasks, scalar_vector_tol):
    """Accuracy of the single task, via the joint scorer."""
    accs = compute_accuracies(
        model, loader, device, N_TASKS, task_thresholds=None, tasks=tasks,
        scalar_vector_tol=scalar_vector_tol)
    return float(accs[TASK_ID])


def train_one_epoch(model, loader, optimizer, device, task_loss_channels, tau, M_reg,
                    grad_clip, lr_scale):
    """One pass over the training set.

    Returns (mean_batch_loss, n_nonfinite, lr_scale). A non-finite loss or gradient
    norm skips the batch and halves lr_scale ON THE SPOT, floor 1e-4 -- so an epoch
    with several bad batches cools several times. (mtl.flat.joint cools once per
    bad epoch with floor 1e-3; the two scripts differ here and always have.) The new
    lr_scale is returned rather than mutated, and only takes effect next epoch, when
    the caller rewrites the param-group LRs.
    """
    model.train()
    epoch_loss, n_batches, n_nonfinite = 0.0, 0, 0

    for inputs, targets, masks, task_ids in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        masks, task_ids = masks.to(device), task_ids.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, task_ids)
        masked_outputs, masked_targets = mask_pair(outputs, targets, masks)

        task_loss = normalized_mse_loss(
            masked_outputs, masked_targets, masks, task_ids, task_loss_channels)
        loss = task_loss + regularization_loss(model, tau, M_reg)

        if not torch.isfinite(loss):
            n_nonfinite += 1
            lr_scale = max(lr_scale * 0.5, 1e-4)
            optimizer.zero_grad()
            continue

        loss.backward()
        if grad_clip and grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(gn):
                n_nonfinite += 1
                lr_scale = max(lr_scale * 0.5, 1e-4)
                optimizer.zero_grad()
                continue
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / max(n_batches, 1), n_nonfinite, lr_scale


def evaluate(model, loader, device, tasks, task_loss_channels, tau, M_reg,
             scalar_vector_tol):
    """One validation pass: (total loss, task loss, accuracy)."""
    tl = compute_loss(model, loader, device, tau, M_reg, task_loss_channels)
    per_task_tl = compute_per_task_loss(
        model, loader, device, N_TASKS, task_loss_channels)
    acc = task_accuracy(model, loader, device, tasks, scalar_vector_tol)
    return float(tl), float(per_task_tl[TASK_ID]), acc


def run_training(model, optimizer, train_loader, val_loader, device, cfg, tasks,
                 task_loss_channels, M_reg, scalar_vector_tol):
    """The epoch loop. Returns the history dict that save_run() consumes.

    Test metrics run EVERY epoch, unlike the joint scripts: the validation set is one task's
    --val-size trials, so the pass is cheap, and computing the validation loss every epoch
    is what makes best_val_loss_epoch exact -- it is argmin(val_loss), so an --eval-interval
    grid would quantise the selected epoch to that grid. Train accuracy is the
    expensive pass (sample_size trials, 4x the validation set at s=200) and nothing selects
    on it, so that one stays on the --eval-interval cadence.

    Early stopping watches VALIDATION LOSS and only arms after --es-start (BPTT semantics).
    """
    base_lr, warm = cfg["learning_rate"], cfg["lr_warmup_epochs"]
    lr_scale, n_nonfinite = 1.0, 0

    h = {
        "loss_history": [],
        "eval_epochs": [],            # every epoch, so this is 1..epochs_run
        "val_loss_history": [],
        "val_acc_hist": [],
        "val_tl_hist": [],
        "train_eval_epochs": [],      # the --eval-interval grid, so SHORTER
        "train_acc_hist": [],
        "best_state": None,
        "stopped_early": False,
    }
    best_val_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        # effective LR = base * warmup ramp * adaptive scale (BPTT._apply_lr)
        factor = min(1.0, (epoch + 1) / warm) if warm and warm > 0 else 1.0
        for grp in optimizer.param_groups:
            grp["lr"] = base_lr * factor * lr_scale

        mean_loss, epoch_nonfinite, lr_scale = train_one_epoch(
            model, train_loader, optimizer, device, task_loss_channels, cfg["tau"],
            M_reg, cfg["grad_clip"], lr_scale)
        n_nonfinite += epoch_nonfinite
        h["loss_history"].append(mean_loss)

        tl, task_tl, val_acc = evaluate(
            model, val_loader, device, tasks, task_loss_channels, cfg["tau"], M_reg,
            scalar_vector_tol)
        h["eval_epochs"].append(epoch + 1)
        h["val_loss_history"].append(tl)
        h["val_tl_hist"].append(task_tl)
        h["val_acc_hist"].append(val_acc)

        if tl < best_val_loss:
            best_val_loss = tl
            h["best_state"] = {k: v.detach().clone()
                               for k, v in model.state_dict().items()}

        if (epoch + 1) % cfg["eval_interval"] == 0 or (epoch + 1) == cfg["epochs"]:
            h["train_eval_epochs"].append(epoch + 1)
            h["train_acc_hist"].append(
                task_accuracy(model, train_loader, device, tasks, scalar_vector_tol))

        if h["val_loss_history"] and epoch + 1 > cfg["es_start"]:
            since_best = (epoch + 1) - h["eval_epochs"][
                int(np.argmin(h["val_loss_history"]))]
            if cfg["patience"] and since_best >= cfg["patience"]:
                h["stopped_early"] = True
                break

    h["n_nonfinite"] = n_nonfinite
    h["lr_scale"] = lr_scale
    return h


def select_best(eval_epochs, val_loss_history, val_acc_hist, val_tl_hist):
    """Pick the reported epoch and read this task's numbers there.

    The reported run is the MIN-LOSS epoch, the same criterion early stopping uses,
    and model.pt holds those weights.

    It is selected on the validation set, which also drives early stopping, so the reported
    numbers are a maximum over the trajectory rather than a clean held-out estimate.
    There is no separate validation split.

    Returns (best_val_loss_epoch, best_val_acc, best_task_loss, best_mean_loss), all None when
    the run produced no evaluations at all.
    """
    val_lh = np.asarray(val_loss_history, dtype=float)
    if not val_lh.size:
        return None, None, None, None
    i = int(np.argmin(val_lh))
    return (int(eval_epochs[i]), float(val_acc_hist[i]), float(val_tl_hist[i]),
            float(val_lh[i]))


def save_curves(out, h):
    """Write results.npz: the full per-epoch curves for this task."""
    np.savez_compressed(
        out / "results.npz",
        loss_history=np.asarray(h["loss_history"], dtype=float),
        # validation-side traces: ONE ENTRY PER EPOCH, so eval_epochs == 1..epochs_run
        eval_epochs=np.asarray(h["eval_epochs"], dtype=int),
        val_loss_history=np.asarray(h["val_loss_history"], dtype=float),
        val_task_accuracies=np.asarray(h["val_acc_hist"], dtype=float),
        val_task_losses=np.asarray(h["val_tl_hist"], dtype=float),
        # train accuracy is on the --eval-interval cadence, so it is SHORTER than the
        # arrays above; index it with train_eval_epochs, not with eval_epochs.
        train_task_accuracies=np.asarray(h["train_acc_hist"], dtype=float),
        train_eval_epochs=np.asarray(h["train_eval_epochs"], dtype=int),
    )


def save_run(out, model, cfg, h, run_info):
    """Write the three artefacts of one run and return its metadata.

    `h` is run_training()'s history dict and `run_info` the few facts metadata records
    that the history does not carry (task name, seed, L, sample_size, n_out, input_dim,
    scalar_vector_tol). Writes model.pt (the min-val-loss weights), results.npz and
    metadata.json, the record the analysis notebooks read.

    Loading best_state into `model` before saving is why this takes the live model:
    the object is left holding the reported weights, not the last-epoch ones.
    """
    if h["best_state"] is not None:
        model.load_state_dict(h["best_state"])
        selection = "best_val_loss"
    else:
        selection = "final"
    torch.save(model.state_dict(), out / "model.pt")

    # HELD-OUT: a third split, never seen by training, early stopping or selection.
    # One task per run, so the per-task dicts below carry a single entry.
    task = ft.FOUNDATION_TASKS_MAP[run_info["task_name"]]
    heldout = score_heldout(
        model, [task], [run_info["task_name"]], cfg["val_size"], cfg["batch_size"],
        run_info["seed"], next(model.parameters()).device,
        scalar_vector_tol=run_info["scalar_vector_tol"])

    best_val_loss_epoch, best_acc, best_task_loss, best_mean_loss = select_best(
        h["eval_epochs"], h["val_loss_history"], h["val_acc_hist"], h["val_tl_hist"])
    svt = run_info["scalar_vector_tol"]

    meta = {
        "task": run_info["task_name"],
        "seed": run_info["seed"],
        "parameterization": "flat_shared",
        "num_individual_params": 0,
        "task_onehot": False,
        "init": "plrnn_native",
        "sample_size": run_info["sample_size"],
        "val_size": cfg["val_size"],
        "batch_size": cfg["batch_size"],
        "nonlinear_units": cfg["L"],
        "hidden_size": cfg["hidden_size"],
        "output_size": run_info["n_out"],
        "input_dim": run_info["input_dim"],
        "num_epochs": cfg["epochs"],
        "epochs_run": len(h["loss_history"]),
        "stopped_early": h["stopped_early"],
        "checkpoint_selection": selection,
        # first epoch at which validation accuracy reached 1.0 (None if never). Recorded only;
        # it does NOT stop the run.
        "epochs_to_first_perfect": next(
            (i + 1 for i, v in enumerate(h["val_acc_hist"]) if v >= 1.0), None),
        "early_stopping_patience": cfg["patience"],
        "early_stopping_start": cfg["es_start"],
        "eval_interval": cfg["eval_interval"],
        "learning_rate": cfg["learning_rate"],
        "lr_warmup_epochs": cfg["lr_warmup_epochs"],
        "weight_decay": cfg["weight_decay"],
        "grad_clip": cfg["grad_clip"],
        "tau": cfg["tau"],
        "M_reg": mar_units(cfg["hidden_size"]),
        "accuracy_threshold_deg": float(THRESHOLD * 180.0 / np.pi),
        "scoring": "hierachical_model_task.rnn_model.compute_accuracies (fixation gate on)",
        # constants since --legacy-accuracy was dropped; kept so a run still records
        # which metric produced its numbers (older runs have them False or absent)
        "tau_sigma_metric": True,
        "tau_sigma": TAU_SIGMA,
        # the actual tolerance used for this task (None => fell back to
        # NON_ANGULAR_THRESHOLD, i.e. this is not one of the 7 continuous tasks)
        "scalar_vector_tol": (float(svt[TASK_ID])
                              if svt and TASK_ID in svt else None),
        "n_nonfinite_batches": h["n_nonfinite"],
        "final_lr_scale": h["lr_scale"],
        "n_shared_params": int(sum(q.numel() for q in model.parameters())),
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_val_acc": best_acc,
        "best_val_loss": best_task_loss,
        "best_mean_val_loss": best_mean_loss,
        "final_train_acc": (float(h["train_acc_hist"][-1])
                            if h["train_acc_hist"] else None),
        "final_val_acc": (float(h["val_acc_hist"][-1])
                           if h["val_acc_hist"] else None),
        **heldout,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    save_curves(out, h)
    return meta


def train_one(job):
    """Train a single (task, L, sample_size, seed) combination. Pool-friendly.

    Returns the one-line status string the parent logs; raising here would take the
    whole Pool down, so everything that can be validated is validated in main().
    """
    task_name, L, sample_size, seed, cfg = job
    cfg = {**cfg, "L": L}   # L varies per job; the rest of cfg is shared

    out = Path(cfg["results_dir"]) / f"N{L}_s{sample_size}" / task_name / f"seed_{seed}"
    if cfg["skip_existing"] and (out / "metadata.json").exists():
        return f"skip  {task_name} L={L} s={sample_size} seed={seed} (already done)"

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(cfg["threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = ft.FOUNDATION_TASKS_MAP[task_name]
    tasks = [task]
    M_reg = mar_units(cfg["hidden_size"])

    train_loader, val_loader = build_loaders(tasks, sample_size, cfg, seed)
    input_dim = next(iter(train_loader))[0].shape[-1]

    model, optimizer = build_model(task.output_dim, input_dim, cfg, device)
    task_loss_channels = build_loss_channels(task)

 
    scalar_vector_tol = compute_scalar_vector_tol(tasks) or None

    # created before training so an unwritable --results-dir fails now, not in hour three
    out.mkdir(parents=True, exist_ok=True)

    h = run_training(model, optimizer, train_loader, val_loader, device, cfg, tasks,
                     task_loss_channels, M_reg, scalar_vector_tol)

    meta = save_run(out, model, cfg, h, {
        "task_name": task_name,
        "seed": seed,
        "sample_size": sample_size,
        "n_out": task.output_dim,
        "input_dim": input_dim,
        "scalar_vector_tol": scalar_vector_tol,
    })

    return (f"done  {task_name} L={L} s={sample_size} seed={seed} | "
            f"epochs={len(h['loss_history'])}/{cfg['epochs']}"
            f"{' (early stop)' if h['stopped_early'] else ''} "
            f"best_val_loss_epoch={meta['best_val_loss_epoch']} best_val_acc={meta['best_val_acc']}")
