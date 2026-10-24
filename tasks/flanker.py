import numpy as np
import torch

from tasks.dataset import CognitiveTask


class Flanker(CognitiveTask):
    """
    Eriksen flanker task (Eriksen & Eriksen, 1974).

    A central target (+/-1) is shown with flanking distractors (+/-1); report
    the target's category, ignoring the flankers. The flankers leak into the
    noisy target channel (target + leak*flankers + noise), so incongruent trials
    carry weaker target evidence unless the flanker channel is subtracted out.

    Input channels (3): [fixation, target + leak*flankers + noise, flankers].
    Output: channel 0 = fixation; channel 3 = target category (binary).
    """

    PHASE_NAMES = ["context", "stimulus", "response"]
    is_binary_task = True

    def __init__(self, duration_params: dict, leak: float = 0.5,
                 noise: float = 1.0):
        super().__init__(duration_params)
        self.leak = leak
        self.noise = noise
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_stim = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_stim + T_response

        target = np.random.choice([-1.0, 1.0])
        flankers = np.random.choice([-1.0, 1.0])

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        ev = target + self.leak * flankers + np.random.normal(0, self.noise, size=T_stim)
        inputs[T_context:T_context + T_stim, 1] = torch.from_numpy(ev.astype(np.float32))
        inputs[T_context:T_context + T_stim, 2] = flankers
        t_resp = T_context + T_stim
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if target > 0 else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_stim, T_response)

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
