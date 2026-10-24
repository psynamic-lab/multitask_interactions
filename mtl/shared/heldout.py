"""The held-out test split: drawn once, scored once, after model selection.

The validation split in every arm does double duty -- it drives early stopping AND
picks the reported checkpoint -- so the number read off it is a maximum over the
trajectory, not a clean estimate. This module adds a third split that nothing
selects on: it is built after training, scored once against the already-chosen
weights, and written straight to metadata.json.

TRIALS ARE DRAWN FROM A DERIVED SEED, with the global numpy/torch RNG snapshotted
and restored around the draw. That is the whole trick: `balanced_loaders` builds
train and then validation off the global stream, so slipping a third dataset into
that stream would shift every trial after it and change the model. Taking the
held-out trials from `seed + SEED_OFFSET` under a restored RNG means train and
validation see exactly the numbers they always did, while the test split stays
reproducible from the run's own seed.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from mtl.shared.scoring import compute_accuracies, compute_per_task_loss
from mtl.tasks.dataset import HierarchicalTasksDataset, collate_fn

# Far enough from any seed in use that no run's test split collides with another
# run's training trials.
SEED_OFFSET = 10_000


def heldout_loader(tasks, per_task, batch_size, seed, dataset_cls=None):
    """Balanced fixed loader over trials disjoint from train and validation.

    `dataset_cls` picks the trial format, as in `balanced_loaders`:
    `HierarchicalTasksDataset` for the raw 7 channels, `MultiTaskDataset` when the
    model needs the one-hot task identity appended.
    """
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        s = seed + SEED_OFFSET
        np.random.seed(s)
        torch.manual_seed(s)
        idx = [t for t in range(len(tasks)) for _ in range(per_task)]
        cls = dataset_cls or HierarchicalTasksDataset
        ds = cls(tasks, n_trials=len(idx), task_indices=idx, fixed=True)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn)
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def evaluate_heldout(model, loader, device, tasks, scalar_vector_tol=None):
    """(per-task accuracy, per-task loss) on the held-out split.

    Scored with the joint pipeline's own functions, so the numbers sit on the same
    scale as the validation ones. Nothing here feeds selection -- the caller has
    already fixed the weights.
    """
    n = len(tasks)
    task_loss_channels = {
        i: t.loss_channels for i, t in enumerate(tasks)
        if getattr(t, "loss_channels", None) is not None
    }
    accs = compute_accuracies(model, loader, device, n, task_thresholds=None,
                              tasks=tasks, scalar_vector_tol=scalar_vector_tol)
    losses = compute_per_task_loss(model, loader, device, n, task_loss_channels)
    return ([float(accs[i]) for i in range(n)],
            [float(losses[i]) for i in range(n)])


def heldout_metadata(names, acc, loss, per_task, seed):
    """The metadata block, so every arm records the held-out split identically."""
    return {
        "test_size": per_task,
        "test_seed": seed + SEED_OFFSET,
        "test_accuracy_per_task": dict(zip(names, acc)),
        "mean_test_accuracy": float(np.mean(acc)) if acc else None,
        "test_loss_per_task": dict(zip(names, loss)),
        "mean_test_loss": float(np.mean(loss)) if loss else None,
    }


def score_heldout(model, tasks, names, per_task, batch_size, seed, device,
                  dataset_cls=None, scalar_vector_tol=None):
    """Build the split, score the selected model on it, return the metadata block."""
    loader = heldout_loader(tasks, per_task, batch_size, seed, dataset_cls)
    acc, loss = evaluate_heldout(model, loader, device, tasks, scalar_vector_tol)
    return heldout_metadata(names, acc, loss, per_task, seed)


__all__ = ["SEED_OFFSET", "evaluate_heldout", "heldout_loader", "heldout_metadata",
           "score_heldout"]
