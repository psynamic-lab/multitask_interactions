"""Trial generators and the batch collate function."""

from tasks.dataset import (
    CognitiveTask,
    HierarchicalTasksDataset,
    MultiTaskDataset,
    collate_fn,
)

__all__ = ["CognitiveTask", "HierarchicalTasksDataset", "MultiTaskDataset", "collate_fn"]


def balanced_indices(n_tasks, per_task):
    """Exactly `per_task` trials for every task, in task order."""
    return [t for t in range(n_tasks) for _ in range(per_task)]


def balanced_loaders(
    tasks, sample_size, val_size, batch_size, seed, dataset_cls=HierarchicalTasksDataset
):
    """Fixed, class-balanced train and validation loaders over `tasks`.

    `fixed=True` makes both datasets draw their trials once at construction, so a
    run sees the same data every epoch. Construction order matters: the train set
    is built before the validation set, and both consume the global RNG, so changing the
    order changes the trials for a given seed.

    The DataLoader gets its own generator, seeded separately, so the batch ORDER is
    independent of the global RNG. Without it an orthogonal re-init -- which draws
    extra numbers -- would silently reshuffle the batches and spoil an init A/B,
    where only p2W should differ.

    `dataset_cls` picks the trial format: `HierarchicalTasksDataset` yields the raw
    7 input channels, `MultiTaskDataset` appends the one-hot task identity a P-free
    joint model needs.
    """
    import torch
    from torch.utils.data import DataLoader

    n_tasks = len(tasks)
    train_idx = balanced_indices(n_tasks, sample_size)
    val_idx = balanced_indices(n_tasks, val_size)
    train_ds = dataset_cls(
        tasks, n_trials=len(train_idx), task_indices=train_idx, fixed=True
    )
    val_ds = dataset_cls(
        tasks, n_trials=len(val_idx), task_indices=val_idx, fixed=True
    )
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


__all__ += ["balanced_indices", "balanced_loaders"]
