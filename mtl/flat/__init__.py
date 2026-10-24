"""The flat, P-free arm: one shared A/W/h/C/D, no p_vector.

Shared by the two flat experiments -- `mtl.flat.joint` (one model over the whole
battery, task identity as a one-hot input) and `mtl.indiv` (one model per task,
no task cue at all). They differ only in how they are trained, so the network and
its regulariser live here and the training loops live beside each experiment.
"""

from mtl.flat.mar import mar_units, regularization_loss
from mtl.flat.plrnn import PLRNN, FlatPLRNN

__all__ = ["PLRNN", "FlatPLRNN", "mar_units", "regularization_loss"]
