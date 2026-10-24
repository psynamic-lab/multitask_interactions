"""Choosing which tasks a run trains on.

Arm-independent: both the flat and the hierarchical joint runs take the same
--tasks / --leave-out semantics, and both need every run to agree on task order
so a task index means the same thing across checkpoints.
"""

import logging

from mtl import tasks as ft

logger = logging.getLogger(__name__)


def resolve_battery(a):
    """Choose the tasks this run trains on.

    Returns (all_names, names, tasks, left_out). Both the --tasks and the --leave-out
    path keep battery order, so a task's index means the same thing across runs; the
    two flags are mutually exclusive. left_out=None means the full battery.
    """
    all_names = list(ft.FOUNDATION_TASKS_MAP)

    left_out = None
    if a.leave_out:
        if a.leave_out not in ft.FOUNDATION_TASKS_MAP:
            raise SystemExit(f"--leave-out '{a.leave_out}' is not in the battery")
        left_out = a.leave_out

    if a.tasks:
        if left_out is not None:
            raise SystemExit("--tasks and --leave-out are mutually exclusive")
        subset = [t.strip() for t in a.tasks.split(",") if t.strip()]
        bad = [t for t in subset if t not in ft.FOUNDATION_TASKS_MAP]
        if bad:
            raise SystemExit(f"--tasks names not in the battery: {bad}")
        if len(set(subset)) != len(subset):
            raise SystemExit("--tasks contains duplicates")
        names = [n for n in all_names if n in set(subset)]
        logger.info(f"SUBSET: {len(names)} tasks: {names}")
    else:
        names = [n for n in all_names if n != left_out]

    tasks = [ft.FOUNDATION_TASKS_MAP[n] for n in names]
    return all_names, names, tasks, left_out


def single_output_dim(tasks):
    """The battery's common readout width.

    The flat arm has ONE D for every task, so a disagreement here is a battery bug
    rather than something to pad around. Checked before the loaders are built, so a
    bad --tasks list fails in a second instead of after the trial generation.
    """
    out_dims = {t.output_dim for t in tasks}
    if len(out_dims) != 1:
        raise SystemExit(f"tasks disagree on output_dim: {sorted(out_dims)}")
    return out_dims.pop()


__all__ = ["resolve_battery", "single_output_dim"]
