"""Reading runs written before a field was renamed.

The validation/test rename (2026-10-21) changed what new runs WRITE. Anything that
reads a stored run has to accept both spellings, or it silently sees nothing: a
`.get("best_val_acc")` against a 2026-08 metadata.json returns None rather than
raising, and a missing npz key sends the caller down a fallback path. Both failures
are quiet, which is what makes them worth a helper instead of ad-hoc `or` chains.

New code writes the `val_*` / `test_*` names; these readers accept the old ones too.
"""


def pick(mapping, *names, default=None):
    """First of `names` present in `mapping`, else `default`.

    Pass the current name first and the historical one after, so a run written
    today wins over a legacy fallback that happens to also be present.
    """
    for n in names:
        if n in mapping:
            return mapping[n]
    return default


def pick_array(npz, *names):
    """First of `names` present in an open npz, else None."""
    for n in names:
        if n in npz.files:
            return npz[n]
    return None


# Current name -> the spelling runs used before the rename.
LEGACY = {
    "val_task_accuracies": "test_task_accuracies",
    "val_task_losses": "test_task_losses",
    "val_loss_history": "test_loss_history",
    "best_val_acc": "best_test_acc",
    "best_val_loss": "best_test_loss",
    "best_mean_val_loss": "best_mean_test_loss",
    "final_val_acc": "final_test_acc",
    "mean_best_val_accuracy": "mean_best_test_accuracy",
    "best_val_accuracy_per_task": "best_test_accuracy_per_task",
    "best_val_loss_per_task": "best_test_loss_per_task",
    "val_size": "test_size",
    "best_val_loss_epoch": "best_epoch",
    "best_val_acc_epoch": "best_acc_epoch",
}


def both(name):
    """(current, legacy) spellings for `name`, ready to splat into pick()."""
    return (name, LEGACY[name]) if name in LEGACY else (name,)


__all__ = ["LEGACY", "both", "pick", "pick_array"]
