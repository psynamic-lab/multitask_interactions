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


from __future__ import annotations

import json
import logging

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

import tasks as ft
from mtl.hierarchical.config import HiearchicalModelConfig
from mtl.hierarchical.model import HierarchicalPLRNN
from mtl.paths import ROOT
from mtl.tasks.dataset import CognitiveTask, HierarchicalTasksDataset

logger = logging.getLogger(__name__)

DEFAULT_SWEEP = ROOT / "data" / "M64_lr5e4_NPsweep_varnorm"
DEFAULT_OUT = ROOT / "data" / "bitcodes_NPsweep_varnorm"
TAG_FIELDS = (
    "accuracy_type",
    "sequence_output",
    "has_delay",
    "requires_memory",
    "requires_rotation",
    "wm_load",
    "n_fixed_points",
)


# --------------------------------------------------------------------------- #
# stimulus battery                                                            #
# --------------------------------------------------------------------------- #
def build_battery(task_names, n_trials, cache_path, seed=0):
    """Draw `n_trials` trials per task once and cache them.

    Returns a list (task-major, aligned with `task_names`) of dicts with
    `inputs` (K, T_max, 7) float32, `lengths` (K,) int32, `masks` (K, T_max)
    float32. Trials of one task are right-padded to that task's own T_max;
    `lengths` is the only thing that defines validity -- `masks` is the
    response-phase LOSS mask (e.g. tasks/flanker.py:49), not a
    padding mask. It is stored so a response-only window can be added later
    without regenerating the battery.
    """
    if cache_path.exists():
        blob = np.load(cache_path, allow_pickle=False)
        cached = [str(x) for x in blob["task_names"]]
        if cached != list(task_names):
            raise RuntimeError(
                f"cached battery {cache_path} has a different task order; delete it"
            )
        if int(blob["n_trials"]) != n_trials:
            raise RuntimeError(
                f"cached battery {cache_path} has K={int(blob['n_trials'])}, "
                f"requested K={n_trials}; delete it or change --trials"
            )
        return [
            {
                "inputs": blob[f"inputs_{i}"],
                "lengths": blob[f"lengths_{i}"],
                "masks": blob[f"masks_{i}"],
            }
            for i in range(len(cached))
        ]

    logger.info(f"building battery: {len(task_names)} tasks x {n_trials} trials ...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    battery, payload = [], {}
    for i, name in enumerate(task_names):
        task = ft.FOUNDATION_TASKS_MAP[name]
        trials = [task() for _ in range(n_trials)]
        lengths = np.array([t[0].shape[0] for t in trials], dtype=np.int32)
        t_max = int(lengths.max())
        inputs = np.zeros((n_trials, t_max, CognitiveTask.BASE_INPUT_DIM), np.float32)
        masks = np.zeros((n_trials, t_max), np.float32)
        for b, (inp, _tgt, msk) in enumerate(trials):
            t = inp.shape[0]
            inputs[b, :t] = inp.numpy()
            masks[b, :t] = msk.numpy()
        battery.append({"inputs": inputs, "lengths": lengths, "masks": masks})
        payload[f"inputs_{i}"] = inputs
        payload[f"lengths_{i}"] = lengths
        payload[f"masks_{i}"] = masks

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        task_names=np.array(list(task_names)),
        n_trials=np.int32(n_trials),
        seed=np.int32(seed),
        **payload,
    )
    logger.info(f"  cached -> {cache_path}")
    return battery


# --------------------------------------------------------------------------- #
# model loading / rollout                                                     #
# --------------------------------------------------------------------------- #
def load_model(folder, device="cpu", allow_spectral=False):
    """Rebuild a checkpoint from its own metadata (works for every grid cell)."""
    meta = json.loads((folder / "metadata.json").read_text())
    names = meta["task_names"]

    rho = meta.get("spectral_rho") or 0.0
    if rho > 0 and not allow_spectral:
        raise RuntimeError(
            f"{folder.name} was trained with spectral_rho={rho}: training monkey-patched "
            "get_parameters to spectrally cap W (foundation_model/train_foundation.py:72), "
            "so a plain rollout does not reproduce the trained dynamics. "
            "Pass --allow-spectral only if you know the cap is irrelevant here."
        )

    tasks = [ft.FOUNDATION_TASKS_MAP[n] for n in names]  # metadata order matters:
    a = HiearchicalModelConfig(tasks=tasks)              # it fixes p_vector row order
    a.num_individual_params = meta["num_individual_params"]
    a.nonlinear_units = meta["nonlinear_units"]
    a.hidden_size = meta["hidden_size"]
    a.a_clamp = meta.get("a_clamp", 0.0) or 0.0
    a.hierarchisation = meta.get("hierarchisation", "all")  # key absent in pre-2026-10-27 runs
    a.device = device

    model = HierarchicalPLRNN(a, HierarchicalTasksDataset)
    state = torch.load(folder / "model.pt", map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, names, meta


@torch.no_grad()
def rollout_states(model, inputs, subject):
    """HierarchicalPLRNN.forward (model.py:43-84) but returning z_all (B, T, M).

    forward() discards the latent trajectory, so the recurrence is replayed here.
    z_all holds the PRE-activation state -- the bitcode is taken from it directly.
    """
    B, T, _ = inputs.shape
    A, W, h, C, D = model.hierarchisation_scheme.get_parameters(subject)

    L = model.L
    z = torch.zeros(B, model.M, device=model.device)
    z_all = torch.empty(B, T, model.M, device=model.device)
    for t in range(T):
        if L > 0:
            z_non_latent = z[:, :-L]
            z_latent = z[:, -L:]
            z_combined = torch.cat([z_non_latent, torch.relu(z_latent)], dim=1)
            z_update = torch.cat(
                [torch.zeros_like(z_non_latent), A * z_latent], dim=1
            )
        else:  # N = 0: exactly linear, no ReLU units at all
            z_combined = z
            z_update = torch.zeros_like(z)
        C_output = torch.einsum("bij,bj->bi", C, inputs[:, t])
        W_output = torch.einsum("bij,bj->bi", W, z_combined)
        z = z_update + W_output + C_output + h
        z_all[:, t] = z
    return z_all


# --------------------------------------------------------------------------- #
# bitcodes + metrics                                                          #
# --------------------------------------------------------------------------- #
def bitcodes(z_all, n_bits):
    """(B, T, M) latent states -> (B, T) int64 codes, MSB-first over the last N units."""
    if n_bits == 0:
        # No ReLU units: one linear region. Guard explicitly -- z[:, :, -0:] would
        # slice the WHOLE state, not an empty one.
        return np.zeros(tuple(z_all.shape[:2]), dtype=np.int64)
    bits = (z_all[:, :, -n_bits:] > 0).to(torch.int64)
    weights = 1 << torch.arange(n_bits - 1, -1, -1, dtype=torch.int64)
    return (bits * weights).sum(-1).cpu().numpy()


def task_bitcode_stats(code_bt, lengths):
    """Sparse distribution + per-trial structure over valid timesteps only."""
    chunks, per_trial_distinct = [], []
    n_runs = 0
    for b, t_valid in enumerate(lengths):
        c = code_bt[b, : int(t_valid)]
        chunks.append(c)
        per_trial_distinct.append(np.unique(c).size)
        # a "run" = maximal stretch inside one trial with a constant code
        n_runs += 1 + int(np.count_nonzero(c[1:] != c[:-1]))

    flat = np.concatenate(chunks)
    codes, counts = np.unique(flat, return_counts=True)
    p = counts / counts.sum()
    order = np.argsort(counts)[::-1]
    entropy_bits = float(-(p * np.log2(p)).sum())

    return {
        "codes": codes.astype(np.int64),
        "counts": counts.astype(np.int64),
        "probs": p,
        "metrics": {
            "n_distinct": int(codes.size),
            "entropy_bits": entropy_bits,
            "eff_regions": float(2.0**entropy_bits),
            "top1_mass": float(p[order[:1]].sum()),
            "top5_mass": float(p[order[:5]].sum()),
            "mean_distinct_per_trial": float(np.mean(per_trial_distinct)),
            "mean_dwell_steps": float(flat.size / n_runs),
            "n_samples": int(flat.size),
            "mean_T": float(np.mean(lengths)),
        },
    }


def js_similarity(codes_list, probs_list):
    """100x100 task x task similarity, 1 - JS distance (base 2), symmetric.

    Codes absent from both tasks contribute nothing to JS, so each pair is
    evaluated on the union of just its two supports. That keeps this exact and
    cheap even at N=16, where the 100-task union can be very wide.
    """
    n = len(codes_list)
    sim = np.ones((n, n), dtype=np.float32)
    for i in range(n):
        ci, pi = codes_list[i], probs_list[i]
        for j in range(i + 1, n):
            cj, pj = codes_list[j], probs_list[j]
            union = np.union1d(ci, cj)
            pv = np.zeros(union.size)
            qv = np.zeros(union.size)
            pv[np.searchsorted(union, ci)] = pi
            qv[np.searchsorted(union, cj)] = pj
            d = jensenshannon(pv, qv, base=2)
            sim[i, j] = sim[j, i] = 1.0 - (0.0 if np.isnan(d) else float(d))
    return sim


# --------------------------------------------------------------------------- #
# per-cell driver                                                             #
# --------------------------------------------------------------------------- #
def process_cell(folder, battery, task_names, out, args, tag):
    model, names, meta = load_model(folder, args.device, args.allow_spectral)
    if names != list(task_names):
        raise RuntimeError(f"{folder.name}: task order differs from the battery")
    n_bits = int(meta["nonlinear_units"])
    n_tasks = len(names)

    codes_cat, counts_cat, offsets = [], [], [0]
    codes_list, probs_list, rows = [], [], []
    acc = meta.get("best_test_accuracy_per_task") or {}

    for t_idx, name in enumerate(names):
        item = battery[t_idx]
        inputs = torch.from_numpy(item["inputs"]).to(args.device)
        subject = torch.full(
            (inputs.shape[0],), t_idx, dtype=torch.long, device=args.device
        )
        z_all = rollout_states(model, inputs, subject)
        stats = task_bitcode_stats(bitcodes(z_all, n_bits), item["lengths"])

        codes_cat.append(stats["codes"])
        counts_cat.append(stats["counts"])
        offsets.append(offsets[-1] + stats["codes"].size)
        codes_list.append(stats["codes"])
        probs_list.append(stats["probs"])

        assert stats["metrics"]["n_distinct"] <= min(
            2**n_bits, stats["metrics"]["n_samples"]
        ), f"{tag} {name}: n_distinct exceeds min(2**N, n_samples)"

        row = {
            "P": int(meta["num_individual_params"]),
            "N": n_bits,
            "seed": int(meta["seed"]),
            "task_idx": t_idx,
            "task_name": name,
            **stats["metrics"],
            "test_accuracy": float(acc[name]) if name in acc else np.nan,
        }
        row.update({f: ft.TAGS[name][f] for f in TAG_FIELDS})
        rows.append(row)

        if args.selfcheck and t_idx == 0:
            _selfcheck(z_all, item["lengths"], n_bits, tag)

    counts_dir = out / f"counts_K{args.trials}"
    counts_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        counts_dir / f"{tag}.npz",
        codes=np.concatenate(codes_cat),
        counts=np.concatenate(counts_cat),
        task_offsets=np.array(offsets, dtype=np.int64),
        task_names=np.array(names),
        n_bits=np.int32(n_bits),
        P=np.int32(meta["num_individual_params"]),
        seed=np.int32(meta["seed"]),
        hidden_size=np.int32(meta["hidden_size"]),
        n_trials=np.int32(args.trials),
        run_dir=np.array(folder.name),
    )

    if args.no_js or n_bits == 0:
        if n_bits == 0:
            logger.info("    JS skipped (N=0 is degenerate: one region, all tasks identical)")
    else:
        js_dir = out / f"js_K{args.trials}"
        js_dir.mkdir(parents=True, exist_ok=True)
        np.save(js_dir / f"{tag}.npy", js_similarity(codes_list, probs_list))

    return rows, n_tasks


def _selfcheck(z_all, lengths, n_bits, tag):
    """Verify bit order against multi_task_training.analysis_functions."""
    from multi_task_training.analysis_functions import bitcode_distribution

    if n_bits == 0:
        logger.info(f"    selfcheck skipped for {tag} (N=0 has no bits)")
        return
    t_min = int(np.min(lengths))
    z_trunc = z_all[:, :t_min].cpu().numpy()  # helper has no notion of lengths
    ref = bitcode_distribution(z_trunc, n_bits)

    code_bt = bitcodes(z_all[:, :t_min], n_bits)
    codes, counts = np.unique(code_bt.reshape(-1), return_counts=True)
    mine = {
        format(int(c), f"0{n_bits}b"): n / counts.sum() for c, n in zip(codes, counts)
    }
    assert set(mine) == set(ref), f"selfcheck {tag}: bitcode supports differ"
    for k, v in mine.items():
        assert abs(v - ref[k]) < 1e-9, f"selfcheck {tag}: mass differs for {k}"
    logger.info(f"    selfcheck OK ({len(mine)} codes match analysis_functions)")


# --------------------------------------------------------------------------- #
