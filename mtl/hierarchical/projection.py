"""The p_vector -> (A, W, h, C, D) projection schemes.

Moved here from `hierachical_model_task/projection_plrnn.py` (2026-10-21).
`args.hierarchisation` picks which of A, W, h, C, D are task-specific (p2X) and
which are shared (X_shared): "all", a subset like "hCD", or "all-AW".
"""

import matplotlib.pyplot as plt
import torch
from torch import nn


class base_hierarchisation:
    def __init__(self, model, args):
        """Base class for hierarchisation schemes."""
        self.model = model
        self.dh = model.dh
        self.dx = model.dx
        self.dz = model.dz
        self.df = model.df
        self.obs_model = args.obs_model
        self.learn_noise_cov = args.learn_noise_cov
        self.num_subjects = args.num_subjects
        # Optional hard bound on the latent self-recurrence A (0 = off). When > 0,
        # get_parameters clamps A to [0, a_clamp] so the linear self-loop can't diverge.
        self.a_clamp = getattr(args, "a_clamp", 0.0)
        # xavier gain for the recurrent projection p2W (smaller = more contractive
        # W at init, helps big-M stability). Defaults to the original 0.1.
        self.w_init_gain = getattr(args, "w_init_gain", 0.1)
        assert self.obs_model in ["identity", "linear"]

    def _clamp_A(self, A):
        """Bound A to [0, a_clamp] if enabled; otherwise return A unchanged."""
        if self.a_clamp and self.a_clamp > 0:
            return A.clamp(min=0.0, max=self.a_clamp)
        return A

    def re_init(self):
        """Re-initializes all subject specific parameters for finetuning."""
        self.model.noise_cov = nn.Parameter(
            torch.zeros(size=(self.num_subjects, 1, self.dx))
        )

    def init_parameters(self):
        """Initializes the model parameters for the given hierarchisation scheme.
        Initializes the covariance of the gaussian noise model to all ones
        (Here: log cov -> to zeros)."""
        self.model.noise_cov = nn.Parameter(
            torch.zeros(size=(self.num_subjects, 1, self.dx)),
            requires_grad=self.learn_noise_cov,
        )
        # obs model
        if self.obs_model == "identity":
            self.model.obs_matrix = nn.Parameter(
                torch.eye(self.dz, self.dx), requires_grad=False
            )
        else:
            self.model.obs_matrix = nn.Parameter(torch.randn(self.dz, self.dx))

    def get_parameters(self, subject):
        """Constructs and returns the model parameters.
        Args:
            subject: subject index for each sample in the batch
        returns:
            model parameters for the given subject(s)"""

    def grouped_parameters(self):
        """Returns a generator object for the individual and the shared
        parameters respectively."""
        return [self.model.obs_matrix], [self.model.noise_cov]

    @torch.no_grad()
    def plot_stuff(self):
        """Function for viualizing interesting things. This is called by a plotter object
        to write it to tensorboard. Plots the noise covariance as an image"""
        fig, ax = plt.subplots(1, figsize=(self.num_subjects + 1, self.dz))
        c = ax.imshow(self.model.noise_cov.squeeze(1).cpu().T)
        fig.colorbar(c, ax=ax)
        ax.set_ylabel(r"$\sigma^2_i$")
        ax.set_xlabel("subject")
        return [(fig, "noise_covariance")]

    def loss(self):
        """Defines any regularization losses on the parameters."""
        return 0.0


PARAM_ORDER = "AWhCD"  # init order; keep it, or seeded "all" runs stop reproducing


def parse_hierarchisation(mode):
    """'all', a letter subset ('AW', 'hCD') or 'all-<letters>' -> canonical 'AWhCD' subset."""
    lookup = {k.lower(): k for k in PARAM_ORDER}
    mode = (mode or "all").strip()
    if mode.lower() == "all":
        letters = PARAM_ORDER
    elif mode.lower().startswith("all-"):
        excluded = _letters(mode[4:], lookup, mode)
        letters = "".join(k for k in PARAM_ORDER if k not in excluded)
    else:
        letters = _letters(mode, lookup, mode)
    if not letters:
        raise ValueError(f"hierarchisation {mode!r} leaves nothing task-specific")
    return "".join(k for k in PARAM_ORDER if k in letters)


def _letters(s, lookup, mode):
    bad = [c for c in s if c.lower() not in lookup]
    if bad:
        raise ValueError(f"hierarchisation {mode!r}: unknown {bad}, use letters of {PARAM_ORDER}")
    return {lookup[c.lower()] for c in s}


class plrnn_projection_hierarchisation(base_hierarchisation):
    """Hierarchisation scheme for the PLRNN variant (A diag, full W, h, C, D).

    Each parameter in `args.hierarchisation` is task-specific, projected from the
    per-task p_vector by a shared tensor p2X; every other one is a single shared
    tensor X_shared. Only p_vector (and noise_cov) is individual.
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self.dp = args.num_individual_params  # dimension of individual parameter vector
        self.M = args.hidden_size
        self.L = args.nonlinear_units if args.nonlinear_units is not None else 0
        self.N = args.output_size
        self.input_dim = args.obs_size
        self.hierarchised = parse_hierarchisation(getattr(args, "hierarchisation", "all"))
        self.shapes = {
            "A": (self.L,),
            "W": (self.M, self.M),
            "h": (self.M,),
            "C": (self.M, self.input_dim),
            "D": (self.N, self.M),
        }

    def _xavier(self, shape, gain):
        if len(shape) == 1:  # xavier needs 2D: init a (1, n) row and drop the axis
            return nn.init.xavier_uniform_(torch.empty(1, *shape), gain=gain).squeeze(0)
        return nn.init.xavier_uniform_(torch.empty(*shape), gain=gain)

    def init_parameters(self):
        """Initialize p_vector, then a p2X or X_shared for each of A, W, h, C, D."""
        super().init_parameters()  # noise_cov
        self.model.p_vector = nn.Parameter(
            torch.repeat_interleave(
                torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0
            )
        )
        for k in PARAM_ORDER:
            gain = self.w_init_gain if k == "W" else 0.1
            if k in self.hierarchised:
                setattr(self.model, f"p2{k}",
                        nn.Parameter(self._xavier((self.dp, *self.shapes[k]), gain)))
            else:
                setattr(self.model, f"{k}_shared",
                        nn.Parameter(self._xavier(self.shapes[k], gain)))

    def re_init(self):
        super().init_parameters()
        self.model.p_vector = nn.Parameter(
            torch.repeat_interleave(
                torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0
            )
        )

    def get_parameters(self, subject):
        """Map per-subject p_vector to concrete parameters.

        Args:
            subject: LongTensor of shape (batch,) with subject indices
        Returns:
            A (batch, L), W (batch, M, M), h (batch, M), C (batch, M, input_dim), D (batch, N, M)
        """
        p = self.model.p_vector[subject]  # (B, dp)
        out = {}
        for k in PARAM_ORDER:
            if k in self.hierarchised:
                proj = getattr(self.model, f"p2{k}")
                # (B, dp) @ (dp, n) for A, h; (B, dp) x (dp, i, j) -> (B, i, j) for W, C, D
                out[k] = p @ proj if proj.dim() == 2 else torch.einsum("bd,dij->bij", p, proj)
            else:
                shared = getattr(self.model, f"{k}_shared")
                out[k] = shared.unsqueeze(0).expand(p.shape[0], *shared.shape)
        return self._clamp_A(out["A"]), out["W"], out["h"], out["C"], out["D"]

    def grouped_parameters(self):
        shared, individual = super().grouped_parameters()
        shared += [
            getattr(self.model, f"p2{k}" if k in self.hierarchised else f"{k}_shared")
            for k in PARAM_ORDER
        ]
        individual += [self.model.p_vector]
        return shared, individual


__all__ = [
    "PARAM_ORDER",
    "base_hierarchisation",
    "parse_hierarchisation",
    "plrnn_projection_hierarchisation",
]
