"""The individual arm: one flat model per task, trained independently.

The network is the flat one (`mtl.flat`): no hierarchisation, no p_vector, and no
task one-hot either, since a single-task model needs no task cue. Only the
training procedure is specific to this arm, and that is `mtl.indiv.runner`.
"""
