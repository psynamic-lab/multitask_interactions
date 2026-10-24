
import numpy as np
import torch

from tasks.dataset import CognitiveTask


class DelayedCueReversal(CognitiveTask):
    """
    A rule cue may appear early (whole context period, on cue_prob = 50% of
    trials) and is withdrawn before the stimulus. The network reports the sign
    of a noiseless stimulus, inverted if the cue appeared.

    Input channels (3): [fixation, stimulus, pm_cue].
    Output: channel 3 = gated decision (binary).
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_perstep_binary_task = True

    def __init__(self, duration_params: dict, cue_prob: float = 0.5):
        super().__init__(duration_params)
        self.cue_prob = cue_prob
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"])
        Ts = np.random.randint(*dp["stimulus"])
        Tr = np.random.randint(*dp["response"])
        T_total = Tc + Ts + Tr

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0

        # PM cue: present for the WHOLE context window on PM trials (salient, easy
        # to detect), then OFF for stimulus+response -- so it must be held across
        # the delay to gate the response. That retention is the PM demand.
        pm = np.random.rand() < self.cue_prob
        if pm:
            inputs[:Tc, 2] = 1.0

        # Ongoing stimulus: signed evidence during the stimulus window.
        stim_sign = 1.0 if np.random.rand() < 0.5 else -1.0
        amp = np.random.uniform(0.5, 1.0)
        inputs[Tc:Tc + Ts, 1] = stim_sign * amp

        t_resp = Tc + Ts
        inputs[t_resp:, 0] = 0.0

        # Ongoing answer = stimulus positive; PM flips it.
        ongoing = stim_sign > 0
        answer = (not ongoing) if pm else ongoing

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if answer else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (Tc, Ts, Tr)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Tr, 2, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
