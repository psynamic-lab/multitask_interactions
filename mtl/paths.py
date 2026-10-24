"""Repo paths, resolved once, and the naming scheme for run directories.

Replaces the per-file ``repo_root()`` definitions and ``sys.path`` blocks: the root
is the nearest ancestor of this file holding ``data/``, so it is correct in either
cluster clone and from any working directory. Set ``MTL_ROOT`` to override.
"""

import os
import re
from pathlib import Path

ROOT = (
    Path(
        os.environ.get("MTL_ROOT")
        or next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())
    )
    .expanduser()
    .resolve()
)

DATA = ROOT / "data"

# canonical joint sweep and its cached analysis products
JOINT = DATA / "total_run_final"
BITCODES = JOINT / "bitcodes_total_run_final"
TRAJECTORIES = BITCODES / "trajectories_K256"


# --------------------------------------------------------------------- naming
#
# NEW runs are named with the quantity spelled out:
#
#     foundation_joint_M64_nonlin16_feat16_s200_seed0
#
# LEGACY runs use single letters, and they are the OPPOSITE WAY ROUND from the
# CLI and from the paper: `N` in a legacy folder name is the nonlinear-unit count
# (`-P` on the command line) and `P` is the task-feature size (`-N`). That is
# exactly why the new scheme spells them out -- `nonlin`/`feat` cannot be read
# backwards, and a legacy glob cannot match a new name by accident.
#
# Read `nonlinear_units` / `num_individual_params` from a run's metadata.json
# when you need certainty; folder names are labels, metadata is the record.

LEGACY_RE = re.compile(
    r"^(?P<prefix>.+?)_M(?P<m>\d+)_N(?P<nonlin>\d+)_P(?P<feat>\d+)"
    r"_s(?P<sample>\d+).*?_seed(?P<seed>\d+)$"
)
RUN_RE = re.compile(
    r"^(?P<prefix>.+?)_M(?P<m>\d+)_nonlin(?P<nonlin>\d+)_feat(?P<feat>\d+)"
    r"_s(?P<sample>\d+).*?_seed(?P<seed>\d+)$"
)


def run_name(m, nonlin, feat, sample, seed, prefix="foundation_joint"):
    """Directory name for a NEW run: foundation_joint_M64_nonlin16_feat16_s200_seed0.

    `nonlin` is the number of nonlinear units (`-P`), `feat` the task-feature size
    (`-N`). Batch size and learning rate are deliberately absent -- they are in
    metadata.json, and keeping them out stops the name growing a field per flag.
    """
    return f"{prefix}_M{m}_nonlin{nonlin}_feat{feat}_s{sample}_seed{seed}"


def parse_run_name(name):
    """Pull (m, nonlin, feat, sample, seed) out of a run folder name, new or legacy.

    Returns None when the name matches neither scheme. The new pattern is tried
    first; a legacy name cannot match it, because `nonlin16` never looks like `N16`.
    """
    for rx in (RUN_RE, LEGACY_RE):
        m = rx.match(Path(name).name)
        if m:
            d = m.groupdict()
            return {
                "prefix": d["prefix"],
                "m": int(d["m"]),
                "nonlin": int(d["nonlin"]),
                "feat": int(d["feat"]),
                "sample": int(d["sample"]),
                "seed": int(d["seed"]),
                "scheme": "new" if rx is RUN_RE else "legacy",
            }
    return None


def find_runs(sweep_dir, nonlin=None, feat=None, seed=None, m=None, sample=None,
              prefix=None):
    """Every run under `sweep_dir` matching the given fields, in either scheme.

    Use this instead of globbing a name pattern: a glob written for one scheme
    silently finds nothing in the other, and a legacy glob reads N and P the wrong
    way round. Fields left as None are not constrained.
    """
    want = {"nonlin": nonlin, "feat": feat, "seed": seed, "m": m,
            "sample": sample, "prefix": prefix}
    out = []
    for d in sorted(Path(sweep_dir).iterdir()):
        if not d.is_dir():
            continue
        got = parse_run_name(d.name)
        if got and all(v is None or got[k] == v for k, v in want.items()):
            out.append(d)
    return out


def run_dir(n, p=16, seed=0, m=64, sample=200):
    """LEGACY checkpoint folder of one cell of the joint sweep.

    Kept because 23 notebooks call it and the stored corpus uses this spelling.
    Note the legacy argument order: `n` is the nonlinear-unit count and `p` the
    task-feature size -- the reverse of the CLI. New code should use `find_runs`.
    """
    return JOINT / f"foundation_joint_M{m}_N{n}_P{p}_s{sample}_b64_lr0.0005_seed{seed}"


__all__ = [
    "BITCODES", "DATA", "JOINT", "ROOT", "TRAJECTORIES",
    "find_runs", "parse_run_name", "run_dir", "run_name",
]
