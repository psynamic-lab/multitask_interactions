"""Fine-tuning ONE new p_vector row on a leave-one-out held-out task.

The hierarchical counterpart to `mtl.hierarchical.joint`: instead of training a
battery from scratch, this loads a LOO checkpoint, appends a single p_vector row
for the task that was held out, freezes everything else, and learns only that
row. Because no shared weight and no pretrained row can move, forgetting is zero
by construction rather than by measurement -- `check_only_new_row_moved` asserts
it afterwards.

Selection differs from the from-scratch trainers: the reported checkpoint is the
MAX VALIDATION ACCURACY epoch, not min validation loss. See `run_finetune`.

The CLI, and the per-checkpoint init-source policy it applies, live in
`scripts/finetune_hierarchical_loo.py`.
"""

import copy
import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mtl import tasks as ft
from mtl.hierarchical.config import HierarchicalModelConfig as HiearchicalModelConfig
from mtl.hierarchical.model import HierarchicalPLRNN
from mtl.paths import ROOT
from mtl.shared.compat import both, pick
from mtl.shared.scoring import (
    compute_accuracies,
    mask_pair,
    masked_eval_batches,
    normalized_mse_loss,
)
from mtl.tasks.dataset import HierarchicalTasksDataset, collate_fn

logger = logging.getLogger(__name__)

DEFAULT_LOO_DIR = ROOT / "data" / "LeaveOneOutData84"

# Per-epoch history columns (shared by each run's history.csv and the combined
# --all history so the latter is just every run's rows stacked).
HISTORY_FIELDS = (
    "loo_index",
    "new_task",
    "output_type",
    "init_source",
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
)

# One row per held-out task in finetune_loo_summary.csv.
SUMMARY_FIELDS = (
    "loo_index",
    "new_task",
    "output_type",
    "init_source",
    "best_val_acc",
    "best_val_acc_epoch",
    "final_val_acc",
)


def load_checkpoint(ckpt_dir):
    """Read one LOO checkpoint and work out which task it held out.

    Returns (state, meta, trained_names, held_out_name, all_tasks, new_idx). The
    held-out task is APPENDED, so its p_vector row is the last one and `new_idx`
    indexes it in every tasks list below.

    The three checks are about row<->task alignment: a p_vector row only means
    anything if row i still corresponds to trained_names[i] in today's battery.
    They raise RuntimeError (a bug or a stale checkpoint), unlike the init-source
    failure in resolve_init_source(), which is a ValueError the --all sweep skips.
    """
    ckpt_dir = Path(ckpt_dir)
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    meta = json.loads((ckpt_dir / "metadata.json").read_text())

    trained_names = meta["task_names"]
    held_out_name = meta["left_out_task"]

    missing = [n for n in trained_names if n not in ft.FOUNDATION_TASKS_MAP]
    if missing:
        raise RuntimeError(
            f"{len(missing)} checkpoint tasks are not in the current battery "
            f"(e.g. {missing[:3]}); row<->task alignment would be wrong."
        )
    if held_out_name not in ft.FOUNDATION_TASKS_MAP:
        raise RuntimeError(f"held-out task '{held_out_name}' not in current battery.")
    if held_out_name in trained_names:
        raise RuntimeError(f"held-out task '{held_out_name}' is in the trained set?!")

    all_tasks = [ft.FOUNDATION_TASKS_MAP[n] for n in trained_names] + [
        ft.FOUNDATION_TASKS_MAP[held_out_name]
    ]
    return state, meta, trained_names, held_out_name, all_tasks, len(all_tasks) - 1


def resolve_init_source(trained_names, held_out_name, init_source, rng=None):
    """Return the trained-task name whose p_vector row seeds the new row.

    Raises ValueError when no usable source exists -- the --all sweep catches that
    and skips the checkpoint, so it must stay a ValueError, not a RuntimeError.
    """
    held_type = ft.TAGS[held_out_name]["accuracy_type"]
    same_type = [n for n in trained_names if ft.TAGS[n]["accuracy_type"] == held_type]
    if init_source == "auto":
        if not same_type:
            raise ValueError(
                f"no trained task shares output type '{held_type}' with "
                f"{held_out_name}; pass --init-source explicitly."
            )
        resolved = same_type[0]
    elif init_source == "random-sametype":
        # pick a RANDOM trained task that shares the held-out task's output type
        if not same_type:
            raise ValueError(
                f"no trained task shares output type '{held_type}' with "
                f"{held_out_name}; pass --init-source explicitly."
            )
        r = rng if rng is not None else np.random
        resolved = same_type[int(r.randint(len(same_type)))]
    elif init_source == "random":
        resolved = "random"
    elif init_source == "mean":
        resolved = "mean"
    else:
        if init_source not in trained_names:
            raise ValueError(
                f"--init-source '{init_source}' is not 'auto'/'random' or a trained task."
            )
        resolved = init_source
    return resolved, held_type, same_type


def make_new_row(old_p, resolved, trained_names):
    """The initial p_vector row for the held-out task.

    'random' draws U(-1, 1), matching the model's own p_vector init; 'mean' takes
    the centroid of the trained rows; anything else copies that task's row. The
    caller seeds the RNG immediately before, so 'random' is reproducible.
    """
    if resolved == "random":
        return torch.empty(1, old_p.shape[1]).uniform_(-1, 1)
    if resolved == "mean":
        return old_p.mean(dim=0, keepdim=True).clone()
    src = trained_names.index(resolved)
    return old_p[src : src + 1].clone()


def build_model(state, meta, all_tasks, new_row, use_gpu):
    """Rebuild the checkpoint's model with one extra p_vector row.

    The config is read back out of metadata.json rather than re-specified, so the
    architecture always matches the weights being loaded. `noise_cov` is grown by
    one zero row alongside p_vector, or load_state_dict would reject the shape.
    """
    args = HiearchicalModelConfig(tasks=all_tasks)
    args.num_individual_params = meta["num_individual_params"]
    args.nonlinear_units = meta["nonlinear_units"]
    args.hidden_size = meta["hidden_size"]
    args.a_clamp = meta.get("a_clamp", 0.0)
    args.hierarchisation = meta.get("hierarchisation", "all")  # key absent in pre-2026-10-27 runs
    args.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    state["p_vector"] = torch.cat([state["p_vector"], new_row], dim=0)
    if "noise_cov" in state:
        old_noise = state["noise_cov"]
        state["noise_cov"] = torch.cat(
            [old_noise, torch.zeros(1, *old_noise.shape[1:])], dim=0
        )

    model = HierarchicalPLRNN(args, HierarchicalTasksDataset)
    model.load_state_dict(state)
    model.to(args.device)
    return model, args.device


def freeze_all_but_new_row(model, old_n, lr):
    """Freeze the network, leave only the LAST p_vector row trainable.

    requires_grad=False on everything else stops those gradients being computed at
    all; the backward hook then zeroes the pretrained rows of the one tensor that
    is still live, so a single Adam state covers just the new row. This is what
    makes forgetting zero by construction rather than by measurement -- verified
    after training by check_only_new_row_moved().
    """
    for p in model.parameters():
        p.requires_grad = False
    model.p_vector.requires_grad = True

    def mask_old(grad, n_old=old_n):
        grad[:n_old] = 0  # only the new (last) row keeps its gradient
        return grad

    model.p_vector.register_hook(mask_old)
    return torch.optim.Adam([model.p_vector], lr=lr)


def build_loaders(all_tasks, new_idx, sample_size, n_val_trials, batch_size):
    """Fixed train/validation loaders drawing ONLY the held-out task.

    task_indices is all new_idx: the pretrained tasks are never sampled, since
    nothing about them can change. fixed=True on both so the trials are drawn once
    and reused every epoch.
    """
    train_ds = HierarchicalTasksDataset(
        all_tasks,
        n_trials=sample_size,
        task_indices=[new_idx] * sample_size,
        fixed=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_ds = HierarchicalTasksDataset(
        all_tasks,
        n_trials=n_val_trials,
        task_indices=[new_idx] * n_val_trials,
        fixed=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


def build_loss_channels(all_tasks):
    """Map task_id -> the output channels that task's loss is computed on.

    Built for the whole battery even though only new_idx is ever sampled, because
    normalized_mse_loss looks the key up by task_id.
    """
    return {
        i: t.loss_channels
        for i, t in enumerate(all_tasks)
        if getattr(t, "loss_channels", None) is not None
    }


def eval_loss(model, loader, device, task_loss_channels):
    """Mean normalized_mse_loss over a loader (no grad)."""
    model.eval()
    tot, nb = 0.0, 0
    for mo, mt, masks, task_ids in masked_eval_batches(model, loader, device):
        tot += normalized_mse_loss(mo, mt, masks, task_ids, task_loss_channels).item()
        nb += 1
    return tot / max(nb, 1)


def task_accuracy(model, loader, device, all_tasks, new_idx):
    """Accuracy of the held-out task alone.

    compute_accuracies scores every task id it sees; only new_idx appears in these
    loaders, so the rest are absent from the dict and .get defaults to 0.0.
    """
    accs = compute_accuracies(
        model, loader, device, len(all_tasks), tasks=all_tasks
    )
    return accs.get(new_idx, 0.0)


def train_one_epoch(model, loader, optimizer, device, task_loss_channels):
    """One pass over the training set; returns the mean batch loss.

    No grad clipping and no non-finite guard, unlike the from-scratch trainers:
    a single p_vector row under a frozen network is a small, well-conditioned
    problem and the originals never needed them.
    """
    model.train()
    ep_loss = 0.0
    for inputs, targets, masks, task_ids in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        masks, task_ids = masks.to(device), task_ids.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, task_ids)
        mo, mt = mask_pair(outputs, targets, masks)
        loss = normalized_mse_loss(mo, mt, masks, task_ids, task_loss_channels)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
    return ep_loss / len(loader)


def evaluate(model, train_loader, val_loader, device, all_tasks, new_idx,
             task_loss_channels):
    """One metric pass: (train accuracy, validation accuracy, validation loss)."""
    tr_acc = task_accuracy(model, train_loader, device, all_tasks, new_idx)
    te_acc = task_accuracy(model, val_loader, device, all_tasks, new_idx)
    te_loss = eval_loss(model, val_loader, device, task_loss_channels)
    return tr_acc, te_acc, te_loss


def run_finetune(model, optimizer, train_loader, val_loader, device, all_tasks,
                 new_idx, task_loss_channels, num_epochs, eval_every, patience,
                 verbose):
    """The epoch loop. Returns the history dict that save_run() consumes.

    The reported checkpoint is the MAX VALIDATION ACCURACY epoch -- note this differs
    from the from-scratch trainers, which select on min validation loss. Selection and
    both early-stopping rules all read the same validation set, so best_val_acc is a
    maximum over the trajectory, not a clean held-out estimate; there is no
    validation split. Fine for "how fast does a new row fit", not a generalization
    number.

    Two stopping rules: te_acc >= 1.0 (nothing left to gain) and `patience` epochs
    with no improvement. Both are checked only on eval epochs, so with
    --eval-every > 1 patience is counted in epochs but tested on that grid.
    """
    h = {
        "hist": {k: [] for k in
                 ("epoch", "train_loss", "val_loss", "train_acc", "val_acc")},
        "best_acc": -1.0,
        "best_val_acc_epoch": None,
        "best_state": None,
        "stopped_epoch": num_epochs,   # may be < num_epochs if a rule fires
    }
    hist = h["hist"]

    # zero-shot: the held-out task scored with the INITIAL row, before any update
    h["zero_shot_acc"] = task_accuracy(model, val_loader, device, all_tasks, new_idx)
    h["zero_shot_loss"] = eval_loss(model, val_loader, device, task_loss_channels)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, task_loss_channels)

        if not (epoch == 0 or (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1):
            continue

        tr_acc, te_acc, te_loss = evaluate(
            model, train_loader, val_loader, device, all_tasks, new_idx,
            task_loss_channels)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(te_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(te_acc)

        if te_acc > h["best_acc"]:
            h["best_acc"], h["best_val_acc_epoch"] = te_acc, epoch + 1
            h["best_state"] = copy.deepcopy(model.state_dict())
        if verbose and ((epoch + 1) % 25 == 0 or epoch == 0):
            logger.info(
                f"  epoch {epoch + 1:4d} | train_loss {train_loss:.4f} "
                f"| val_loss {te_loss:.4f} | train_acc {tr_acc:.3f} "
                f"| val_acc {te_acc:.3f} | best {h['best_acc']:.3f}"
            )

        if te_acc >= 1.0:
            h["stopped_epoch"] = epoch + 1
            if verbose:
                logger.info(
                    f"  early stop at epoch {h['stopped_epoch']} (100% validation accuracy)")
            break
        if (patience and h["best_val_acc_epoch"] is not None
                and (epoch + 1) - h["best_val_acc_epoch"] >= patience):
            h["stopped_epoch"] = epoch + 1
            if verbose:
                logger.info(
                    f"  early stop at epoch {h['stopped_epoch']} "
                    f"(no improvement for {patience} epochs; best @ {h['best_val_acc_epoch']})"
                )
            break

    return h


def check_only_new_row_moved(p_before, p_after, old_n, verbose):
    """Assert the gradient mask held: the pretrained rows must be untouched.

    This is the claim the whole design rests on -- "forgetting is zero by
    construction" -- so it is asserted, not logged and shrugged at. Returns
    (max |dp| over the old rows, max |dp| over the new row) for the metadata.
    """
    moved_old = (p_after[:old_n] - p_before[:old_n]).abs().max().item()
    moved_new = (p_after[old_n:] - p_before[old_n:]).abs().max().item()
    if verbose:
        logger.info(
            f"  max |dp| old rows = {moved_old:.2e} (should be ~0), "
            f"new row = {moved_new:.2e}"
        )
    assert moved_old < 1e-6, "old p_vector rows changed -- gradient mask failed!"
    return moved_old, moved_new


def build_history_rows(hist, loo_index, new_task, output_type, init_source):
    """Per-epoch CSV rows, each carrying the task-identifying columns.

    Those columns are redundant in a single run's history.csv, and the point of
    them: the --all sweep stacks every run's rows into one file.
    """
    return [
        {
            "loo_index": loo_index,
            "new_task": new_task,
            "output_type": output_type,
            "init_source": init_source,
            "epoch": e,
            "train_loss": trl,
            "val_loss": tel,
            "train_acc": tra,
            "val_acc": tea,
        }
        for e, trl, tel, tra, tea in zip(
            hist["epoch"], hist["train_loss"], hist["val_loss"],
            hist["train_acc"], hist["val_acc"],
        )
    ]


def save_curves(out, hist, hist_rows):
    """Write the per-epoch history as BOTH npz and csv (easy to plot either way)."""
    np.savez(out / "history.npz", **{k: np.array(v) for k, v in hist.items()})
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(HISTORY_FIELDS))
        w.writeheader()
        w.writerows(hist_rows)


def save_run(out, h, p_after, new_row, old_n, run_info, verbose):
    """Write the four artefacts of one run and return the result dict.

    model.pt holds the BEST-val-accuracy weights; p_vector.pt holds the initial,
    final and best rows separately, because the final row is the one the movement
    check was computed on while the best row is the one model.pt carries.
    `history_rows` is returned but deliberately NOT written into results.json --
    the per-epoch data lives in history.csv, and the key exists only so the --all
    sweep can stack the curves without re-reading them.
    """
    out.mkdir(parents=True, exist_ok=True)
    torch.save(h["best_state"], out / "model.pt")

    hist_rows = build_history_rows(
        h["hist"], run_info["loo_index"], run_info["new_task"],
        run_info["output_type"], run_info["init_source"])
    save_curves(out, h["hist"], hist_rows)

    best_state = h["best_state"]
    torch.save(
        {
            "p_init_row": new_row.cpu(),
            "init_source": run_info["init_source"],
            "new_task_idx": run_info["new_task_idx"],
            "final_p_row": p_after[old_n:].cpu(),
            "best_p_row": (
                best_state["p_vector"][old_n:].cpu()
                if best_state is not None
                else p_after[old_n:].cpu()
            ),
        },
        out / "p_vector.pt",
    )

    hist = h["hist"]
    result = {
        **run_info,
        "stopped_epoch": h["stopped_epoch"],
        "best_val_acc": h["best_acc"],
        "best_val_acc_epoch": h["best_val_acc_epoch"],
        "final_val_acc": hist["val_acc"][-1] if hist["val_acc"] else 0.0,
        "final_train_acc": hist["train_acc"][-1] if hist["train_acc"] else 0.0,
        "zero_shot_val_acc": float(h["zero_shot_acc"]),
        "zero_shot_val_loss": float(h["zero_shot_loss"]),
        "moved_old_rows_max": run_info["moved_old_rows_max"],
        "moved_new_row_max": run_info["moved_new_row_max"],
        "timestamp": datetime.now().astimezone().isoformat(),
        "history_rows": hist_rows,  # per-epoch rows, for the combined --all csv
    }
    result.pop("new_task_idx", None)  # internal index, not part of the record
    with open(out / "results.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history_rows"}, f, indent=2)
    if verbose:
        logger.info(
            f"{run_info['new_task']} <- {run_info['init_source']}: best val acc "
            f"{h['best_acc']:.1%} @ epoch {h['best_val_acc_epoch']}  ->  {out}\n"
        )
    return result


def finetune(
    ckpt_dir,
    sample_size,
    n_val_trials,
    num_epochs,
    lr,
    batch_size,
    finetune_seed,
    eval_every,
    init_source,
    use_gpu,
    output_dir=None,
    patience=200,
    verbose=True,
):
    """Fine-tune one checkpoint's new p_vector row end to end.

    The seed is re-applied three times -- before the row init, before the data, and
    before training -- so each stage draws the same numbers whatever the previous
    one consumed. That is load-bearing, not redundant: the model constructor runs a
    full parameter init between the first two, and how much RNG it burns depends on
    the architecture read out of the checkpoint.
    """
    ckpt_dir = Path(ckpt_dir)
    state, meta, trained_names, held_out_name, all_tasks, new_idx = load_checkpoint(
        ckpt_dir)

    resolved, held_type, same_type = resolve_init_source(
        trained_names,
        held_out_name,
        init_source,
        rng=np.random.RandomState(finetune_seed),
    )
    if verbose:
        logger.info(
            f"held-out task: {held_out_name} (output type '{held_type}') | "
            f"trained on {len(trained_names)} tasks"
        )
        logger.info(
            f"new p_vector row initialized from: {resolved} "
            f"({len(same_type)} same-type candidates)"
        )

    old_n, _dp = state["p_vector"].shape
    assert old_n == len(
        trained_names
    ), f"checkpoint p_vector rows ({old_n}) != trained tasks ({len(trained_names)})"

    torch.manual_seed(finetune_seed)
    np.random.seed(finetune_seed)
    new_row = make_new_row(state["p_vector"], resolved, trained_names)

    model, device = build_model(state, meta, all_tasks, new_row, use_gpu)
    optimizer = freeze_all_but_new_row(model, old_n, lr)

    torch.manual_seed(finetune_seed)
    np.random.seed(finetune_seed)
    train_loader, val_loader = build_loaders(
        all_tasks, new_idx, sample_size, n_val_trials, batch_size)
    task_loss_channels = build_loss_channels(all_tasks)

    p_before = model.p_vector.detach().clone()

    torch.manual_seed(finetune_seed)
    np.random.seed(finetune_seed)
    h = run_finetune(
        model, optimizer, train_loader, val_loader, device, all_tasks, new_idx,
        task_loss_channels, num_epochs, eval_every, patience, verbose)

    # p_after is read BEFORE best_state is restored: the movement check is about
    # where training actually ended up, not about the checkpoint being kept.
    p_after = model.p_vector.detach()
    moved_old, moved_new = check_only_new_row_moved(p_before, p_after, old_n, verbose)
    if h["best_state"] is not None:
        model.load_state_dict(h["best_state"])

    out = (
        Path(output_dir)
        if output_dir
        else (ckpt_dir / f"finetune_{held_out_name}_init{resolved}")
    )
    return save_run(out, h, p_after, new_row, old_n, {
        "new_task": held_out_name,
        "init_source": resolved,
        "output_type": held_type,
        "checkpoint": str(ckpt_dir),
        "loo_index": meta.get("left_out_index"),
        "new_task_idx": new_idx,
        "sample_size": sample_size,
        "n_val_trials": n_val_trials,
        "num_epochs": num_epochs,
        "patience": patience,
        "lr": lr,
        "batch_size": batch_size,
        "finetune_seed": finetune_seed,
        "moved_old_rows_max": moved_old,
        "moved_new_row_max": moved_new,
    }, verbose)


def load_init_map(path):
    """Read the optional task -> init_source CSV."""
    with open(path, newline="") as f:
        return {row["task"]: row["init_source"] for row in csv.DictReader(f)}


def find_checkpoints(loo_dir):
    """Every LOO checkpoint under loo_dir, in (loo index, seed, name) order.

    Accepts both the old naming ("loo_5_M64_...") and the new joint-run naming
    ("foundation_joint_M64_..._loo5_seed0"); a checkpoint is any subdir carrying
    both a model.pt and a metadata.json.
    """
    def key(d):
        m = re.search(r"loo_?(\d+)", d.name)
        s = re.search(r"seed(\d+)", d.name)
        return (int(m.group(1)) if m else 0, int(s.group(1)) if s else 0, d.name)

    return sorted(
        (d for d in loo_dir.iterdir()
         if d.is_dir() and (d / "model.pt").exists()
         and (d / "metadata.json").exists()),
        key=key,
    )


def existing_result_dir(ckpt_dir, odir):
    """Return an already-finetuned output dir for this checkpoint, or None."""
    if odir is not None:  # output-root mode: result lands directly in odir
        return odir if (odir / "results.json").exists() else None
    hits = sorted(ckpt_dir.glob("finetune_*_init*/results.json"))
    return hits[0].parent if hits else None


def load_existing(rdir):
    """Reload a finished run so the combined CSVs stay complete on resume."""
    res = json.loads((rdir / "results.json").read_text())
    hist_rows = []
    hp = rdir / "history.csv"
    if hp.exists():
        with open(hp, newline="") as f:
            hist_rows = list(csv.DictReader(f))
    res["history_rows"] = hist_rows
    return res


def write_sweep_outputs(out_base, summary):
    """One summary row per held-out task, plus every task's curve stacked."""
    with open(out_base / "finetune_loo_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDS))
        w.writeheader()
        for r in summary:
            # .get: a results.json from before the val/test rename lacks the
            # new keys, and a resumed sweep must not die on one old cell.
            w.writerow({k: r.get(k, pick(r, *both(k))) for k in SUMMARY_FIELDS})
    with open(out_base / "finetune_loo_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(HISTORY_FIELDS))
        w.writeheader()
        for r in summary:
            w.writerows(r["history_rows"])


def run_sweep(loo_dir, init_for, run_one, output_root, skip_existing=True):
    """Fine-tune every checkpoint under `loo_dir`, resuming by default.

    A checkpoint whose held-out task has no valid init source raises ValueError;
    that is collected and reported at the end rather than aborting the sweep, so
    one unusable task does not cost the other 83.

    `init_for` and `run_one` are supplied by the caller rather than built here, so
    this does not need to know the CLI: the script closes them over its parsed
    arguments. `skip_existing` is the positive form of the script's
    --no-skip-existing.
    """
    folders = find_checkpoints(loo_dir)
    summary, skipped = [], []

    for i, d in enumerate(folders):
        odir = (output_root / d.name) if output_root else None
        done = existing_result_dir(d, odir) if skip_existing else None
        if done is not None:
            res = load_existing(done)
            summary.append(res)
            logger.info(
                f"[{i + 1}/{len(folders)}] {d.name} -- already done "
                f"({res['new_task']} <- {res['init_source']}, "
                f"best {res['best_val_acc']:.1%}); skipping"
            )
            continue
        logger.info(f"[{i + 1}/{len(folders)}] {d.name} <- init {init_for(d)}")
        try:
            summary.append(run_one(d, output_dir=odir))
        except ValueError as e:
            # e.g. held-out task is the only one of its output type -> no init source
            skipped.append((d.name, str(e)))
            logger.info(f"    SKIP: {e}")

    out_base = output_root or loo_dir
    write_sweep_outputs(out_base, summary)

    mean_acc = np.mean([float(r["best_val_acc"]) for r in summary]) if summary else 0.0
    logger.info(f"\nmean best_val_acc over {len(summary)} held-out tasks: {mean_acc:.1%}")
    logger.info(f"summary -> {out_base / 'finetune_loo_summary.csv'}")
    logger.info(f"stacked per-epoch history -> {out_base / 'finetune_loo_history.csv'}")
    if skipped:
        logger.info(f"\nskipped {len(skipped)} checkpoint(s) with no valid init source:")
        for name, msg in skipped:
            logger.info(f"  {name}: {msg}")


def resolve_single_checkpoint(loo_dir, ckpt_name, loo_index):
    """The one checkpoint directory --ckpt-name / --loo-index selects."""
    if ckpt_name is not None:
        return loo_dir / ckpt_name
    matches = sorted(loo_dir.glob(f"loo_{loo_index}_*"))
    if not matches:
        raise SystemExit(f"no loo_{loo_index}_* under {loo_dir}")
    return matches[0]
