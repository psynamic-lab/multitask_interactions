#!/usr/bin/env python
"""Per-task bitcode (linear-region) distributions across the N sweep at fixed P.

The models in the N x P sweep are almost-linear PLRNNs: only the last
N = `nonlinear_units` coordinates of the M-dimensional latent state pass through
a ReLU (see hierachical_model_task/model.py:58-71). The network is therefore
piecewise-linear with at most 2**N regions, and the region occupied at time t is
exactly the sign pattern

    bitcode(t) = z[t, -N:] > 0

read MSB-first, which matches the string convention in
multi_task_training/analysis_functions.py:41 so results are cross-checkable
against the existing `bitcode_distribution` helper (--selfcheck does that).

For every (P, N, seed) cell and every one of the 100 battery tasks this writes:

  A. counts_K{K}/P{P}_N{N}_seed{S}.npz  -- the full sparse distribution
        (ragged CSR-style: `codes`, `counts`, `task_offsets`)
  B. bitcode_metrics_K{K}.csv           -- n_distinct / entropy / eff_regions /
        concentration / per-trial / dwell, joined with tasks.TAGS and
        the checkpoint's per-task test accuracy
  C. js_K{K}/P{P}_N{N}_seed{S}.npy      -- 100x100 task x task Jensen-Shannon
        similarity, same `1 - jensenshannon(...)` convention as
        multi_task_training/utils.py:505-521

All cells see a byte-identical stimulus battery (cached to battery_K{K}.npz), so
distributions are comparable across N, P and seed. Bitcode *labels* are not
comparable across seeds -- unit indices are arbitrary per run -- but the metrics
are.

Example
-------
python foundation_model/notebooks/compute_bitcode_distributions.py --p 8 16 --selfcheck
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import tasks as ft
from mtl.analysis.bitcodes import (
    DEFAULT_OUT,
    DEFAULT_SWEEP,
    build_battery,
    process_cell,
)
from mtl.log import add_log_level_argument
from mtl.log import setup as setup_logging
from mtl.paths import find_runs

logger = logging.getLogger(__name__)


def build_parser():
    """Command-line interface. Separate from main() so the parser can be
    imported and inspected (defaults, docs, tests) without running anything.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--p", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--n", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--trials", type=int, default=50, help="trials per task (K)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None, help="single cell, e.g. P8_N2_seed2")
    ap.add_argument("--skip-existing", action="store_true", help="skip cells whose counts npz exists")
    ap.add_argument("--no-js", action="store_true", help="skip the task x task JS matrices")
    ap.add_argument("--selfcheck", action="store_true", help="cross-check bit order per cell")
    ap.add_argument("--allow-spectral", action="store_true")
    add_log_level_argument(ap)
    return ap


def main():
    args = build_parser().parse_args()
    setup_logging(args.log_level)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    task_names = list(ft.FOUNDATION_TASKS_MAP)
    battery = build_battery(
        task_names, args.trials, out / f"battery_K{args.trials}.npz"
    )

    cells = [
        (p, n, s) for p in args.p for n in args.n for s in args.seeds
    ]
    if args.only:
        cells = [c for c in cells if f"P{c[0]}_N{c[1]}_seed{c[2]}" == args.only]
        if not cells:
            raise SystemExit(
                f"--only {args.only} matches no cell in the requested grid")

    all_rows, skipped, missing, t0 = [], [], [], time.time()
    for p, n, s in cells:
        tag = f"P{p}_N{n}_seed{s}"
        # find_runs, not a glob: it matches both the legacy single-letter names
        # and the spelled-out nonlin/feat ones.
        hits = find_runs(args.sweep_dir, nonlin=n, feat=p, seed=s)
        hits = [h for h in hits if (h / "model.pt").exists()]
        if len(hits) != 1:
            missing.append(f"{tag} ({len(hits)} dirs with model.pt)")
            logger.info(f"[skip] {tag}: {len(hits)} matching run dirs with model.pt")
            continue
        if args.skip_existing and (out / f"counts_K{args.trials}" / f"{tag}.npz").exists():
            skipped.append(tag)
            logger.info(f"[have] {tag}")
            continue

        t1 = time.time()
        logger.info(f"[run ] {tag}  <- {hits[0].name}")
        rows, n_tasks = process_cell(hits[0], battery, task_names, out, args, tag)
        all_rows.extend(rows)
        med = float(np.median([r["n_distinct"] for r in rows]))
        logger.info(
            f"    {n_tasks} tasks | median n_distinct={med:.0f} "
            f"| {time.time() - t1:.1f}s",
            flush=True,
        )

    if all_rows:
        csv_path = out / f"bitcode_metrics_K{args.trials}.csv"
        df = pd.DataFrame(all_rows)
        if csv_path.exists() and (args.skip_existing or args.only):
            # merge with previously computed cells instead of clobbering them
            old = pd.read_csv(csv_path)
            key = ["P", "N", "seed", "task_idx"]
            keep = old.merge(df[key].drop_duplicates(), on=key, how="left", indicator=True)
            old = keep[keep["_merge"] == "left_only"].drop(columns="_merge")
            df = pd.concat([old, df], ignore_index=True)
        df = df.sort_values(["P", "N", "seed", "task_idx"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"\nmetrics -> {csv_path}  ({len(df)} rows)")

    logger.info(
        f"done in {time.time() - t0:.1f}s | "
        f"computed {len(all_rows) // max(len(task_names), 1)} cells"
        + (f" | reused {len(skipped)}" if skipped else "")
        + (f" | MISSING {len(missing)}: {', '.join(missing)}" if missing else "")
    )


if __name__ == "__main__":
    main()
