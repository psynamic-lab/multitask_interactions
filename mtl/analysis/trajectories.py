"""Dump the latent trajectories of the N x P sweep next to the existing bitcodes.

compute_bitcode_distributions.py rolls z_all out and throws it away, keeping only
the sparse code histograms (counts_K*/). Geometry work -- subspace overlap, PCA,
DSA -- needs the trajectories themselves, so this replays the *same* rollout on
the *same* cached battery and writes only z_all. Bitcodes are NOT recomputed:
counts_K*/ already has them, and per-timestep codes are one line away from z
(`bitcodes(z, N)`, see `codes_from_z` below). --verify uses the stored counts to
prove the dumped trajectories come from the same checkpoints.

Layout -- one .npz per cell inside the bitcode folder, ragged over tasks:

  <bitcode-dir>/trajectories_K{K}/P{P}_N{N}_seed{S}.npz
    z_{i}        (K, T_i, M) float32/float16  pre-activation latent states
    lengths_{i}  (K,)        int32            valid timesteps (rest is padding)
    masks_{i}    (K, T_i)    float32          response-phase LOSS mask (not padding)
    task_names, n_bits, P, seed, hidden_size, n_trials, run_dir, sweep_dir,
    battery, dtype

Phase labels live in ONE file next to the battery, since they are a property of
the stimuli and therefore shared by every (P, N, seed) cell:

  <bitcode-dir>/battery_phases_K{K}.npz
    phases_{i}      (K, T_i) int64   phase index per timestep, -1 on padding
    phase_names_{i} (n_ph,)  str     PHASE_NAMES of the task class
    informative     (n_tasks,) bool  False where the task exposes a single phase
    task_names, n_trials, seed, battery

build_battery draws trials with task(); this redraws them with
task.generate_trial_with_phases(), which consumes the RNG identically, and then
ASSERTS the regenerated inputs are byte-identical to the cached battery. That
assert is what guarantees the phase labels line up with the saved trajectories.

Example
-------
python foundation_model/notebooks/save_trajectories.py --p 8 16 --n 4 --verify --phases
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import tasks as ft
from mtl.analysis.bitcodes import bitcodes, load_model, rollout_states
from mtl.paths import ROOT, find_runs

# the bitcode folder and the sweep it was computed from -- keep these paired
DEFAULT_BITCODES = ROOT / "data" / "bitcodes_NPsweep_varnorm"
DEFAULT_SWEEP = ROOT / "data" / "M64_lr5e4_NPsweep_varnorm"


def find_cell(sweep_dir, p, n, seed):
    """Run dir for one sweep cell. `n` = nonlinear units, `p` = task features.

    Goes through `find_runs` rather than a glob so both naming schemes are found:
    a glob written for one silently returns nothing for the other.
    """
    hits = find_runs(sweep_dir, nonlin=n, feat=p, seed=seed)
    return [h for h in hits if (h / "model.pt").exists()]


def valid_mask(lengths, shape):
    m = np.zeros(shape, bool)
    for b, t in enumerate(lengths):
        m[b, : int(t)] = True
    return m


@torch.no_grad()
def dump_cell(folder, battery, task_names, out_path, args):
    model, names, meta = load_model(folder, args.device, args.allow_spectral)
    if names[: len(task_names)] != list(task_names):
        raise RuntimeError(f"{folder.name}: task order differs from the battery")
    n_bits = int(meta["nonlinear_units"])

    payload, n_valid, codes_seen = {}, 0, []
    for i, name in enumerate(task_names):
        item = battery[i]
        inputs = torch.from_numpy(item["inputs"]).to(args.device)
        subject = torch.full(
            (inputs.shape[0],), i, dtype=torch.long, device=args.device
        )
        z_all = rollout_states(model, inputs, subject)            # (K, T_i, M) float32

        payload[f"z_{i}"] = z_all.cpu().numpy().astype(args.dtype)
        payload[f"lengths_{i}"] = item["lengths"].astype(np.int32)
        payload[f"masks_{i}"] = item["masks"]
        n_valid += int(item["lengths"].sum())

        if args.verify:  # histogram only, for the cross-check against counts_K*
            keep = valid_mask(item["lengths"], z_all.shape[:2])
            codes_seen.append(bitcodes(z_all, n_bits)[keep])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if args.compress else np.savez
    saver(
        out_path,
        task_names=np.array(list(task_names)),
        n_bits=np.int32(n_bits),
        P=np.int32(meta["num_individual_params"]),
        seed=np.int32(meta["seed"]),
        hidden_size=np.int32(meta["hidden_size"]),
        n_trials=np.int32(args.trials),
        run_dir=np.array(folder.name),
        sweep_dir=np.array(str(args.sweep_dir)),
        battery=np.array(str(args.battery)),
        dtype=np.array(args.dtype),
        **payload,
    )
    return {
        "n_bits": n_bits,
        "n_valid": n_valid,
        "mb": out_path.stat().st_size / 1e6,
        "codes": codes_seen,
    }


# --------------------------------------------------------------------------- #
# phase labels for the cached battery                                         #
# --------------------------------------------------------------------------- #
def save_phases(battery_path, out_path, trials):
    """Redraw the cached battery with phase annotations and store the labels.

    The redraw must consume the RNG exactly like build_battery did, so ALL tasks
    are regenerated in the original order even if only some are of interest, and
    the resulting inputs are checked against the cache task by task. A mismatch
    means the two draw paths diverged (e.g. a task whose
    generate_trial_with_phases does extra sampling) and the labels cannot be
    trusted -- it raises instead of writing a misaligned file.
    """
    blob = np.load(battery_path, allow_pickle=False)
    names = [str(x) for x in blob["task_names"]]
    if int(blob["n_trials"]) != trials:
        raise RuntimeError(
            f"{battery_path.name} has K={int(blob['n_trials'])}, requested K={trials}"
        )
    seed = int(blob["seed"])

    torch.manual_seed(seed)
    np.random.seed(seed)

    payload, informative, trivial = {}, [], []
    for i, name in enumerate(names):
        task = ft.FOUNDATION_TASKS_MAP[name]
        drawn = [task.generate_trial_with_phases() for _ in range(trials)]

        ref = blob[f"inputs_{i}"]
        got = np.zeros_like(ref)
        phases = np.full(ref.shape[:2], -1, np.int64)
        for b, (inp, _tgt, _msk, ph) in enumerate(drawn):
            t = inp.shape[0]
            if t > ref.shape[1]:
                raise AssertionError(f"{name}: trial {b} longer than the cached battery")
            got[b, :t, : inp.shape[1]] = inp.numpy()  # task() zero-pads to 7 channels
            if ph.shape[0] != t:
                raise AssertionError(f"{name}: {ph.shape[0]} phase labels for T={t}")
            phases[b, :t] = ph
        if not np.array_equal(got, ref):
            raise AssertionError(
                f"{name}: regenerated trials differ from {battery_path.name}; "
                "generate_trial_with_phases does not follow the same RNG path"
            )

        seen = np.unique(phases[phases >= 0])
        n_seen = int(seen.max()) + 1 if seen.size else 1
        # Instance first: a modifier builds its names from the base it wraps
        # (tasks/modifiers.py:53). Name what is actually labelled,
        # never what the class merely advertises.
        declared = list(getattr(task, "PHASE_NAMES", []) or [])
        if n_seen == 1:
            declared = ["all"]
        elif len(declared) < n_seen:
            declared = [f"phase{k}" for k in range(n_seen)]
        payload[f"phases_{i}"] = phases
        payload[f"phase_names_{i}"] = np.array(declared)
        informative.append(seen.size > 1)
        if seen.size <= 1:
            trivial.append(name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        task_names=np.array(names),
        n_trials=np.int32(trials),
        seed=np.int32(seed),
        battery=np.array(str(battery_path)),
        informative=np.array(informative),
        **payload,
    )
    return {
        "n_tasks": len(names),
        "n_informative": int(np.sum(informative)),
        "trivial": trivial,
        "mb": out_path.stat().st_size / 1e6,
    }


def verify_against_counts(counts_path, codes_seen, dtype_note=""):
    """Cross-check the dumped trajectories against the stored bitcode counts.

    Reads counts_K*/P*_N*_seed*.npz (already on disk, nothing recomputed) and
    compares its per-task (code -> count) histogram with the one implied by the
    trajectories. A mismatch means the trajectories and the bitcodes come from
    different checkpoints -- almost always a sweep-dir/bitcode-dir mispairing.
    """
    blob = np.load(counts_path, allow_pickle=False)
    codes, counts, off = blob["codes"], blob["counts"], blob["task_offsets"]
    for i, mine in enumerate(codes_seen):
        c_ref, n_ref = codes[off[i] : off[i + 1]], counts[off[i] : off[i + 1]]
        c_mine, n_mine = np.unique(mine, return_counts=True)
        if not (np.array_equal(c_ref, c_mine) and np.array_equal(n_ref, n_mine)):
            raise AssertionError(
                f"{counts_path.name} task {i}: histogram differs from the dumped "
                f"trajectories ({c_ref.size} vs {c_mine.size} codes){dtype_note}"
            )
    return len(codes_seen)


# --------------------------------------------------------------------------- #
# readers (import these from a notebook)                                      #
# --------------------------------------------------------------------------- #
def codes_from_z(z, n_bits):
    """Per-timestep bitcode of a saved z array -- MSB-first over the last N units."""
    return bitcodes(torch.as_tensor(np.asarray(z, dtype=np.float32)), n_bits)


def phases_path_for(path):
    """battery_phases_K{K}.npz that goes with a trajectories_K{K}/*.npz file."""
    path = Path(path)
    blob = np.load(path, allow_pickle=False)
    return path.parent.parent / f"battery_phases_K{int(blob['n_trials'])}.npz"


def load_cell(path, tasks=None, valid_only=False, with_codes=False, with_phases=False):
    """{task_name: {"z", "lengths", "masks"[, "code"][, "phase", "phase_names"]}}
    plus a "_meta" entry.

    valid_only=True drops padding: z -> (n_valid, M), masks/code/phase -> (n_valid,).
    with_codes=True derives the bitcodes from z (nothing is read from counts_K*).
    with_phases=True pulls the labels from battery_phases_K{K}.npz; without
    valid_only they are -1 on padded timesteps, matching `phases >= 0` masking.
    """
    path = Path(path)
    blob = np.load(path, allow_pickle=False)
    names = [str(x) for x in blob["task_names"]]
    n_bits = int(blob["n_bits"])
    want = set(names if tasks is None else tasks)

    ph_blob = None
    if with_phases:
        ph_path = phases_path_for(path)
        if not ph_path.exists():
            raise FileNotFoundError(
                f"{ph_path} not found; run save_trajectories.py --phases-only"
            )
        ph_blob = np.load(ph_path, allow_pickle=False)
        if [str(x) for x in ph_blob["task_names"]] != names:
            raise RuntimeError(f"{ph_path.name}: task order differs from {path.name}")

    out = {
        "_meta": {
            k: (blob[k].item() if blob[k].ndim == 0 else blob[k])
            for k in ("n_bits", "P", "seed", "hidden_size", "n_trials",
                      "run_dir", "sweep_dir", "battery", "dtype")
        }
    }
    for i, name in enumerate(names):
        if name not in want:
            continue
        z = blob[f"z_{i}"]
        lengths, masks = blob[f"lengths_{i}"], blob[f"masks_{i}"]
        code = codes_from_z(z, n_bits) if with_codes else None
        phase = None
        if ph_blob is not None:
            phase = ph_blob[f"phases_{i}"]
            if phase.shape != z.shape[:2]:
                raise RuntimeError(
                    f"{name}: phases {phase.shape} do not match z {z.shape[:2]}"
                )
        if valid_only:
            keep = valid_mask(lengths, z.shape[:2])
            z, masks = z[keep], masks[keep]
            code = None if code is None else code[keep]
            phase = None if phase is None else phase[keep]
        rec = {"z": z, "lengths": lengths, "masks": masks}
        if with_codes:
            rec["code"] = code
        if ph_blob is not None:
            rec["phase"] = phase
            rec["phase_names"] = [str(x) for x in ph_blob[f"phase_names_{i}"]]
            rec["phase_informative"] = bool(ph_blob["informative"][i])
        out[name] = rec
    return out


# --------------------------------------------------------------------------- #
