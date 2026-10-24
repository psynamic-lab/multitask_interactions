"""Task-to-task transfer for the flat, P-free PLRNN.

Takes one task's individually-trained model and fine-tunes it onto another task,
then reports how much faster that was than training the target from scratch. The
flat counterpart to `mtl.hierarchical.finetune`: there, transfer means adding a
p_vector row to a joint model; here it means re-training a whole single-task
network from a different task's weights.

The from-scratch denominator is the TARGET task's own individual run, never the
source's -- the question is how much the source initialisation changed the time to
learn THIS task. `read_baseline` reads that run off disk rather than recomputing
it; `--scratch-control` retrains it only when you want the check.

The CLI lives in `scripts/finetune_flat_transfer.py`.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from mtl.flat.mar import mar_units, regularization_loss
from mtl.shared.compat import both, pick, pick_array
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
from mtl.shared.thresholds import crossing_epoch, field_names

logger = logging.getLogger(__name__)

# The five parameter tensors --freeze can pin.
PARAM_NAMES = ("A", "W", "h", "C", "D")


def read_source(ckpt_path):
    """Read a source checkpoint and infer M and L from the weights themselves.

    Returns (state, info) where info records what the checkpoint says about itself, from
    the sibling metadata.json when one exists. Deliberately does NOT build the model:
    constructing a PLRNN draws from the global RNG, and the scratch control has to be
    constructed at exactly the point train_indiv_models.py constructs its model (right
    after the datasets) for the two to be bit-identical.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise SystemExit(
            f"source checkpoint not found: {ckpt_path}\n"
            "The individual runs' model.pt files live on the cluster at "
            "/gs/home/garcias/Projects/multitask_learning/data/20260801_flat_individual "
            "-- data/20260801_individual_runs_v2 locally holds metadata + results only."
        )
    state = torch.load(ckpt_path, map_location="cpu")
    missing = [k for k in PARAM_NAMES if k not in state]
    if missing:
        raise SystemExit(
            f"{ckpt_path} is not a flat PLRNN checkpoint (missing {missing}). "
            "Hierarchical checkpoints (p_vector/p2W/...) are not loadable here -- "
            "use others/finetune_new_task.py for those."
        )

    M, L = int(state["W"].shape[0]), int(state["A"].shape[0])
    info = {"hidden_size": M, "nonlinear_units": L, "source_task": None}

    meta_path = ckpt_path.parent / "metadata.json"
    if meta_path.exists():
        src_meta = json.loads(meta_path.read_text())
        info["source_task"] = src_meta.get("task")
        info["source_seed"] = src_meta.get("seed")
        info["source_sample_size"] = src_meta.get("sample_size")
        info["source_best_val_acc"] = src_meta.get("best_val_acc")
        for key in ("hidden_size", "nonlinear_units"):
            if src_meta.get(key) is not None and src_meta[key] != info[key]:
                raise SystemExit(
                    f"checkpoint/metadata disagree on {key}: weights say {info[key]}, "
                    f"{meta_path} says {src_meta[key]}"
                )

    return state, info


def apply_freeze(model, frozen):
    """Freeze the named parameters. Returns a snapshot of them for the after-check."""
    for name, p in model.named_parameters():
        p.requires_grad = name not in frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit(f"--freeze {','.join(sorted(frozen))} leaves nothing to train.")
    return {n: p.detach().clone() for n, p in model.named_parameters() if n in frozen}


def check_freeze(model, snapshot):
    """Assert the frozen parameters are bit-for-bit unchanged; return the max drift."""
    if not snapshot:
        return 0.0
    moved = max(
        (p.detach() - snapshot[n]).abs().max().item()
        for n, p in model.named_parameters()
        if n in snapshot
    )
    assert moved == 0.0, f"frozen parameters changed (max |delta| = {moved:.3e})"
    return moved


def train_arm(model, task, train_loader, val_loader, cfg, device, frozen, label):
    """Train one arm (transfer or scratch). Mirrors train_indiv_models.train_one's loop."""
    tasks = [task]
    M = model.M
    M_reg, tau = mar_units(M), cfg["tau"]

    task_loss_channels = (
        {0: task.loss_channels} if getattr(task, "loss_channels", None) is not None else {}
    )
    # tau*sigma tolerance for the continuous-readout tasks; {} for the others, which then
    # fall back to the fixed NON_ANGULAR_THRESHOLD. Snapshots/restores the RNG itself.
    scalar_vector_tol = (
        compute_scalar_vector_tol(tasks) or None
    )

    snapshot = apply_freeze(model, frozen)
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    base_lr, warm = cfg["learning_rate"], cfg["lr_warmup_epochs"]
    grad_clip, eval_interval = cfg["grad_clip"], cfg["eval_interval"]
    lr_scale = 1.0

    loss_history = []
    eval_epochs, val_loss_history, val_acc_hist, val_tl_hist = [], [], [], []
    train_acc_hist, train_eval_epochs = [], []
    best_val_loss, best_state = float("inf"), None
    n_nonfinite, stopped_early = 0, False

    for epoch in range(cfg["epochs"]):
        # effective LR = base * warmup ramp * adaptive scale (BPTT._apply_lr)
        factor = min(1.0, (epoch + 1) / warm) if warm and warm > 0 else 1.0
        for grp in optimizer.param_groups:
            grp["lr"] = base_lr * factor * lr_scale

        model.train()
        epoch_loss, n_batches = 0.0, 0
        for inputs, targets, masks, task_ids in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            masks, task_ids = masks.to(device), task_ids.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, task_ids)

            masked_outputs, masked_targets = mask_pair(outputs, targets, masks)

            task_loss = normalized_mse_loss(
                masked_outputs, masked_targets, masks, task_ids, task_loss_channels
            )
            loss = task_loss + regularization_loss(model, tau, M_reg)

            # non-finite guard: skip the batch, cool the LR down (BPTT behaviour)
            if not torch.isfinite(loss):
                n_nonfinite += 1
                lr_scale = max(lr_scale * 0.5, 1e-4)
                optimizer.zero_grad()
                continue

            loss.backward()
            if grad_clip and grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                if not torch.isfinite(gn):
                    n_nonfinite += 1
                    lr_scale = max(lr_scale * 0.5, 1e-4)
                    optimizer.zero_grad()
                    continue
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        loss_history.append(epoch_loss / max(n_batches, 1))

        # ---- validation-side metrics EVERY epoch (cheap, and makes best_val_loss_epoch exact) --------
        tl = compute_loss(model, val_loader, device, tau, M_reg, task_loss_channels)
        per_task_tl = compute_per_task_loss(model, val_loader, device, 1, task_loss_channels)
        val_acc = float(
            compute_accuracies(
                model, val_loader, device, 1, task_thresholds=None, tasks=tasks,
                scalar_vector_tol=scalar_vector_tol,
            )[0]
        )

        eval_epochs.append(epoch + 1)
        val_loss_history.append(float(tl))
        val_tl_hist.append(float(per_task_tl[0]))
        val_acc_hist.append(val_acc)

        if tl < best_val_loss:
            best_val_loss = float(tl)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # ---- train accuracy only on the interval (the expensive pass) -----------------
        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == cfg["epochs"]:
            tr_accs = compute_accuracies(
                model, train_loader, device, 1, task_thresholds=None, tasks=tasks,
                scalar_vector_tol=scalar_vector_tol,
            )
            train_eval_epochs.append(epoch + 1)
            train_acc_hist.append(float(tr_accs[0]))

        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info(
                f"  [{label}] epoch {epoch + 1:4d} | train_loss {loss_history[-1]:.4f} "
                f"| val_loss {tl:.4f} | val_acc {val_acc:.3f}"
            )

        # early stopping in EPOCHS, only after early_stopping_start (BPTT semantics)
        if val_loss_history and epoch + 1 > cfg["es_start"]:
            since_best = (epoch + 1) - eval_epochs[int(np.argmin(val_loss_history))]
            if cfg["patience"] and since_best >= cfg["patience"]:
                stopped_early = True
                break

    moved_frozen = check_freeze(model, snapshot)

    # Saved weights = the best-by-val-loss state (train_foundation.py's convention).
    if best_state is not None:
        model.load_state_dict(best_state)
        selection = "best_val_loss"
    else:
        selection = "final"

    return {
        "model": model,
        "selection": selection,
        "loss_history": loss_history,
        "eval_epochs": eval_epochs,
        "val_loss_history": val_loss_history,
        "val_acc_hist": val_acc_hist,
        "val_tl_hist": val_tl_hist,
        "train_acc_hist": train_acc_hist,
        "train_eval_epochs": train_eval_epochs,
        "n_nonfinite": n_nonfinite,
        "stopped_early": stopped_early,
        "lr_scale": lr_scale,
        "scalar_vector_tol": scalar_vector_tol,
        "moved_frozen": moved_frozen,
    }


def save_arm(out, arm, task_name, cfg, model_dims, extra):
    """Write model.pt + results.npz + metadata.json, mirroring the individual-run schema."""
    out.mkdir(parents=True, exist_ok=True)
    model = arm["model"]
    torch.save(model.state_dict(), out / "model.pt")

    val_lh = np.asarray(arm["val_loss_history"], dtype=float)
    best_idx = int(np.argmin(val_lh)) if val_lh.size else -1
    M, L, n_out, input_dim = model_dims
    svt = arm["scalar_vector_tol"]

    # epochs_to_80 / reached_80 / epochs_to_90 / reached_90 -- same names, same position
    # and the same crossing_epoch used by the script that backfills these into the
    # individual runs, so a fine-tuned cell and its from-scratch counterpart share a schema.
    threshold_fields = {}
    for t in cfg["thresholds"]:
        key_epochs, key_reached = field_names(t)
        ep = crossing_epoch(arm["val_acc_hist"], arm["eval_epochs"], t)
        threshold_fields[key_epochs] = ep
        threshold_fields[key_reached] = ep is not None

    meta = {
        "task": task_name,
        "seed": cfg["seed"],
        "parameterization": "flat_shared",
        "num_individual_params": 0,
        "task_onehot": False,
        "sample_size": cfg["sample_size"],
        "val_size": cfg["val_size"],
        "batch_size": cfg["batch_size"],
        "nonlinear_units": L,
        "hidden_size": M,
        "output_size": n_out,
        "input_dim": input_dim,
        "num_epochs": cfg["epochs"],
        "epochs_run": len(arm["loss_history"]),
        "stopped_early": arm["stopped_early"],
        "checkpoint_selection": arm["selection"],
        "epochs_to_first_perfect": next(
            (i + 1 for i, v in enumerate(arm["val_acc_hist"]) if v >= 1.0), None
        ),
        **threshold_fields,
        "early_stopping_patience": cfg["patience"],
        "early_stopping_start": cfg["es_start"],
        "eval_interval": cfg["eval_interval"],
        "learning_rate": cfg["learning_rate"],
        "lr_warmup_epochs": cfg["lr_warmup_epochs"],
        "weight_decay": cfg["weight_decay"],
        "grad_clip": cfg["grad_clip"],
        "tau": cfg["tau"],
        "M_reg": mar_units(M),
        "accuracy_threshold_deg": float(THRESHOLD * 180.0 / np.pi),
        "scoring": "hierachical_model_task.rnn_model.compute_accuracies (fixation gate on)",
        "tau_sigma_metric": True,
        "tau_sigma": TAU_SIGMA,
        "scalar_vector_tol": (float(svt[0]) if svt and 0 in svt else None),
        "n_nonfinite_batches": arm["n_nonfinite"],
        "final_lr_scale": arm["lr_scale"],
        "n_shared_params": int(sum(q.numel() for q in model.parameters())),
        "best_val_loss_epoch": int(arm["eval_epochs"][best_idx]) if best_idx >= 0 else None,
        "best_val_acc": float(arm["val_acc_hist"][best_idx]) if best_idx >= 0 else None,
        "best_val_loss": float(arm["val_tl_hist"][best_idx]) if best_idx >= 0 else None,
        "best_mean_val_loss": float(val_lh[best_idx]) if best_idx >= 0 else None,
        "final_train_acc": float(arm["train_acc_hist"][-1]) if arm["train_acc_hist"] else None,
        "final_val_acc": float(arm["val_acc_hist"][-1]) if arm["val_acc_hist"] else None,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    meta.update(extra)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    np.savez_compressed(
        out / "results.npz",
        loss_history=np.asarray(arm["loss_history"], dtype=float),
        # validation-side traces: ONE ENTRY PER EPOCH, so eval_epochs == 1..epochs_run
        eval_epochs=np.asarray(arm["eval_epochs"], dtype=int),
        val_loss_history=val_lh,
        val_task_accuracies=np.asarray(arm["val_acc_hist"], dtype=float),
        val_task_losses=np.asarray(arm["val_tl_hist"], dtype=float),
        train_eval_epochs=np.asarray(arm["train_eval_epochs"], dtype=int),
        train_task_accuracies=np.asarray(arm["train_acc_hist"], dtype=float),
    )
    return meta


def read_baseline(baseline_root, target_task, L, sample_size, seed, levels):
    """The TARGET task's own from-scratch run, for the speed-up denominator.

    The comparison a transfer cell wants is "how long did THIS task take when trained from
    scratch?", so the baseline is always the target's individual run -- never the source's
    -- at the matching N / sample_size / seed:

        <baseline_root>/N{L}_s{sample_size}/<target_task>/seed_{seed}/

    Prefers the stored epochs_to_* fields (written by backfill_epochs_to_threshold.py) but
    falls back to recomputing them from that run's results.npz, so this works whether or not
    the backfill has been run yet. Returns None if the baseline run cannot be found, and
    never raises: a missing baseline must not kill a 9,900-run sweep.
    """
    run_dir = Path(baseline_root) / f"N{L}_s{sample_size}" / target_task / f"seed_{seed}"
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    # The trace is loaded unconditionally rather than trusting the stored epochs_to_* fields
    # alone, because the fallback level below needs the PEAK accuracy, which no metadata
    # field records. (best_val_acc is the accuracy at the min-val-loss epoch, and across
    # these runs 56% peak higher than it.) Reading one small npz per cell is cheap and it
    # keeps every number here computed the same way.
    npz_path = run_dir / "results.npz"
    acc = ev = None
    if npz_path.exists():
        try:
            with np.load(npz_path) as d:
                # both spellings: runs written before the val/test rename say test_*
                a = pick_array(d, *both("val_task_accuracies"))
                if a is not None and a.ndim == 1:
                    acc = a
                    ev = d["eval_epochs"] if "eval_epochs" in d.files else None
        except (OSError, KeyError, ValueError):
            acc = None

    if acc is not None:
        source = "results.npz"
        epochs = {f"{t:g}": crossing_epoch(acc, ev, t) for t in levels}
        i_peak = int(np.argmax(acc))
        peak_acc = float(acc[i_peak])
        peak_epoch = int(ev[i_peak]) if ev is not None else i_peak + 1
    else:
        # No usable trace: fall back to the stored fields, and no peak-based salvage.
        if any(field_names(t)[0] not in meta for t in levels):
            return None
        source = "metadata"
        epochs = {f"{t:g}": meta[field_names(t)[0]] for t in levels}
        peak_acc = peak_epoch = None

    return {
        "run": str(run_dir),
        "read_from": source,
        "epochs_to_threshold": epochs,
        "peak_val_acc": peak_acc,
        "peak_epoch": peak_epoch,
        "best_val_acc": pick(meta, *both("best_val_acc")),
        "epochs_run": meta.get("epochs_run"),
    }


def epochs_ratio(finetuned, baseline):
    """finetuned / baseline per threshold. None where either side never reached the level.

    A null is NOT the same as "no effect": it means one of the two arms never got there.
    Both raw epoch counts are kept in the summary next to this, so a null can always be
    read as transfer-failed vs baseline-failed rather than guessed at.
    """
    out = {}
    for k, ft_ep in finetuned.items():
        bl_ep = baseline.get(k) if baseline else None
        out[k] = round(ft_ep / bl_ep, 4) if (ft_ep and bl_ep) else None
    return out


def ratio_at_baseline_best(ft_acc, ft_eval, baseline, levels):
    """Fallback ratio for ONLY the levels the from-scratch baseline never reached.

    Where the baseline did reach the level, epochs_ratio has a real denominator and this is
    left null -- the two fields never describe the same cell. Where it did not (~1/3 of the
    individual runs never reach 80%), the same ratio is computed at the best accuracy the
    baseline actually managed, so the cell still gets a number.

    `level` records the bar used, and it differs per cell: these are NOT interchangeable with
    the strict ratios, and a baseline that peaked near chance gives a meaningless number, so
    filter on `level` before trusting a cell. A null `ratio` with a non-null `level` means
    the fine-tuned run never reached even that bar -- a real result, not a missing value.

    The bar is max(val_task_accuracies), NOT the baseline's best_val_acc field: that field
    is the accuracy at the min-val-loss epoch and 56% of these runs peak above it, which
    would understate what the fine-tuned model had to clear.
    """
    out = {}
    for t in levels:
        key = f"{t:g}"
        entry = {"ratio": None, "level": None, "finetuned_epochs": None,
                 "baseline_epochs": None}
        if (baseline is not None
                and baseline["epochs_to_threshold"].get(key) is None
                and baseline.get("peak_val_acc") is not None):
            level, bl_ep = baseline["peak_val_acc"], baseline["peak_epoch"]
            ft_ep = crossing_epoch(ft_acc, ft_eval, level)
            entry.update(level=round(float(level), 6), baseline_epochs=bl_ep,
                         finetuned_epochs=ft_ep,
                         ratio=round(ft_ep / bl_ep, 4) if (ft_ep and bl_ep) else None)
        out[key] = entry
    return out


def max_acc_ratio(ft_acc, baseline):
    """Best accuracy reached fine-tuning / best accuracy reached from scratch. Always set.

    NOTE the direction is INVERTED relative to the epoch ratios: here >1 means the
    initialisation reached a HIGHER accuracy, whereas <1 means faster for the epoch ratios.

    Unlike the epoch metrics this needs no threshold and no crossing, so it is defined for
    every cell with a baseline -- including the ~1/3 whose runs never reach 80%. Both maxima
    are peaks over the trajectory (max of validation accuracy), not the best_val_acc field, which
    is the accuracy at the min-val-loss epoch.
    """
    ft_max = float(np.max(ft_acc)) if len(ft_acc) else None
    bl_max = baseline.get("peak_val_acc") if baseline else None
    return {
        "ratio": (round(ft_max / bl_max, 4) if (ft_max is not None and bl_max) else None),
        "finetuned_max_acc": ft_max,
        "baseline_max_acc": bl_max,
    }


def epochs_to(acc_hist, eval_epochs, levels):
    """{level: first epoch reaching it} for each level; None where never reached.

    Note this deliberately does NOT use multi_task_training.utils.epochs_to_threshold: that
    helper tests `acc > threshold` strictly, and with --val-size 50 the accuracies are
    multiples of 0.02, so 0.80 and 0.90 are exactly attainable -- a run that plateaus at
    exactly 0.90 would be reported as never reaching 90%. `crossing_epoch` uses `>=`, which
    is also what the existing epochs_to_first_perfect field does (`v >= 1.0`).
    """
    return {f"{t:g}": crossing_epoch(acc_hist, eval_epochs, t) for t in levels}
