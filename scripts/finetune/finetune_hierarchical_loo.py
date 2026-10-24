#!/usr/bin/env python
# Migrated copy: imports go through the installed `mtl` package, so there is no
# sys.path manipulation. The original is still in place and unmodified.
"""Fine-tune ONLY a new p_vector row for a leave-one-out (LOO) held-out task.

Each LOO checkpoint in data/LeaveOneOutData84/ was trained on the foundation
battery with ONE real task left out (metadata['left_out_task']). This script:

  1. loads the checkpoint (shared p2A/p2W/p2h/p2C/p2D + the trained p_vector rows),
  2. appends ONE new p_vector row for the held-out task, initialized by copying
     a trained task that shares the same OUTPUT TYPE (accuracy_type in ft.TAGS),
  3. FREEZES every weight, unfreezes only p_vector, and masks the gradient so
     ONLY the new (last) row is learned,
  4. trains on --sample-size trials, validates on --n-val-trials trials,
  5. saves the loss + accuracy across ALL epochs to history.npz and history.csv.

Because only the new row moves, the pretrained tasks are bit-for-bit unchanged
(forgetting is zero by construction).

Examples:
    # single checkpoint (loo_0 -> held out NoiseCleaner), auto init source
    python scripts/finetune_hierarchical_loo.py --loo-index 0

    # sweep every loo_* checkpoint, write one summary row per held-out task
    python scripts/finetune_hierarchical_loo.py --all
"""

import argparse
import json
import logging
from pathlib import Path

from mtl.hierarchical.finetune import (
    DEFAULT_LOO_DIR,
    finetune,
    load_init_map,
    resolve_single_checkpoint,
    run_sweep,
)
from mtl.log import add_log_level_argument
from mtl.log import setup as setup_logging

logger = logging.getLogger(__name__)


def build_parser():
    """Command-line interface. Separate from main() so the parser can be
    imported and inspected (defaults, docs, tests) without running anything.
    """
    p = argparse.ArgumentParser(
        description="Fine-tune only a new p_vector row on a LOO held-out task."
    )
    p.add_argument(
        "--loo-dir",
        type=str,
        default=str(DEFAULT_LOO_DIR),
        help="Directory containing the loo_* checkpoints.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--loo-index",
        type=int,
        default=0,
        help="Fine-tune the single loo_<index>_* checkpoint.",
    )
    g.add_argument(
        "--ckpt-name",
        type=str,
        default=None,
        help="Fine-tune this exact checkpoint folder name.",
    )
    g.add_argument(
        "--all", action="store_true", help="Sweep every loo_* checkpoint in --loo-dir."
    )
    p.add_argument(
        "--init-source",
        type=str,
        default="auto",
        help="'auto' (same output type), 'random', 'mean' (centroid of the trained rows), "
        "or a trained task name.",
    )
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--n-val-trials", type=int, default=256)
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Early-stop after this many epochs without a val-acc "
        "improvement (0 = disabled).",
    )
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--finetune-seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="With --all: re-run checkpoints that already have a "
        "finetune_*/ result (default is to skip/resume).",
    )
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output dir (single-checkpoint mode only).",
    )
    p.add_argument(
        "--init-map",
        type=str,
        default=None,
        help="CSV with columns 'task,init_source' giving a per-held-out-task "
        "init source (looked up by each checkpoint's left_out_task; overrides "
        "--init-source per checkpoint, falls back to it if a task is missing).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="With --all: write each checkpoint's result to a NEW folder "
        "<output-root>/<checkpoint_name>/ instead of inside the checkpoint, "
        "so the (downloaded) models stay read-only.",
    )
    add_log_level_argument(p)
    return p


def main():
    a = build_parser().parse_args()
    setup_logging(a.log_level)

    loo_dir = Path(a.loo_dir)

    init_map = {}
    if a.init_map:
        init_map = load_init_map(a.init_map)
        logger.info(f"loaded init map for {len(init_map)} tasks from {a.init_map}")
    output_root = Path(a.output_root) if a.output_root else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)

    def init_for(ckpt_dir):
        """Per-checkpoint init source: --init-map by left_out_task, else --init-source."""
        if not init_map:
            return a.init_source
        held = json.loads((ckpt_dir / "metadata.json").read_text())["left_out_task"]
        return init_map.get(held, a.init_source)

    def run_one(ckpt_dir, output_dir=None):
        return finetune(
            ckpt_dir=ckpt_dir,
            sample_size=a.sample_size,
            n_val_trials=a.n_val_trials,
            num_epochs=a.num_epochs,
            lr=a.lr,
            batch_size=a.batch_size,
            finetune_seed=a.finetune_seed,
            eval_every=a.eval_every,
            init_source=init_for(ckpt_dir),
            use_gpu=a.use_gpu,
            output_dir=output_dir,
            patience=a.patience,
        )

    if a.all:
        run_sweep(loo_dir, init_for, run_one, output_root,
                  skip_existing=not a.no_skip_existing)
    else:
        ckpt_dir = resolve_single_checkpoint(loo_dir, a.ckpt_name, a.loo_index)
        logger.info(f"checkpoint: {ckpt_dir}")
        run_one(ckpt_dir, output_dir=a.output_dir)

if __name__ == "__main__":
    main()
