import torch
import torch.nn as nn

from .snn_layers import SpikeAct


class TemporalConditionEncoder(nn.Module):
    """Encode a static multi-label condition into a causal spike sequence.

    The learnable temporal embedding and recurrent LIF state make c_t depend on
    both the condition and time, instead of copying the same vector T times.
    """

    def __init__(self, condition_dim, embed_dim, n_steps, mode='lif', tau=0.25):
        super().__init__()
        self.condition_dim = condition_dim
        self.embed_dim = embed_dim
        self.n_steps = n_steps
        self.mode = mode
        self.tau = tau

        self.input_projection = nn.Linear(condition_dim, embed_dim)
        self.temporal_embedding = nn.Parameter(torch.empty(n_steps, embed_dim))
        self.recurrent = nn.Linear(embed_dim, embed_dim, bias=False)
        self.input_norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.temporal_embedding, mean=0.0, std=0.02)
        nn.init.orthogonal_(self.recurrent.weight, gain=0.25)

    def forward(self, condition):
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                f'expected condition shape (B, {self.condition_dim}), '
                f'got {tuple(condition.shape)}'
            )
        #基础电流
        base_current = self.input_norm(self.input_projection(condition))
        if self.mode == 'static':
            # Ablation baseline: projected condition, without temporal dynamics.
            return torch.sigmoid(base_current).unsqueeze(-1).expand(
                -1, -1, self.n_steps
            )
        if self.mode != 'lif':
            raise ValueError(f'unknown temporal_condition_mode: {self.mode}')
#LIF更新
        membrane = torch.zeros_like(base_current)
        previous_spike = torch.zeros_like(base_current)
        condition_spikes = []
        for t in range(self.n_steps):
            current = (
                base_current
                + self.temporal_embedding[t].unsqueeze(0)
                + self.recurrent(previous_spike)
            )
            membrane = self.tau * membrane * (1.0 - previous_spike) + current
            previous_spike = SpikeAct.apply(membrane)
            condition_spikes.append(previous_spike)

        return torch.stack(condition_spikes, dim=-1)
