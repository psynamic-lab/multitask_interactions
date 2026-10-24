"""Analysis products computed from trained checkpoints.

These are PRODUCERS, not plots: they read a sweep of checkpoints and write the
derived data the interpretability notebooks consume -- per-task bitcode
distributions, the trajectories behind the DSA work, and the flat-model bitcodes.

Each module keeps its own argparse CLI, so they stay runnable as
`python -m mtl.analysis.bitcodes ...` exactly as the originals were.
"""
