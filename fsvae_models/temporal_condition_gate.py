import torch
import torch.nn as nn


class TemporalConditionGate(nn.Module):
    """Condition-dependent, time-selective membrane-current modulation.

    Given SNN features h_t and temporal condition spikes c_t, this module uses
    g_t = sigmoid(a_h * h_t + W_g c_t + b_t) and returns
    h'_t = h_t + alpha * g_t * W_c c_t.  The result is fed into the next LIF
    layer, so the gated residual changes membrane charging and spike firing.
    """

    def __init__(self, feature_dim, condition_dim, n_steps,
                 injection_type='temporal_gate'):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.n_steps = n_steps
        self.injection_type = injection_type

        self.condition_projection = nn.Linear(condition_dim, feature_dim)
        self.gate_projection = nn.Linear(condition_dim, feature_dim)
        self.feature_scale = nn.Parameter(torch.zeros(1, feature_dim, 1))
        self.temporal_bias = nn.Parameter(torch.zeros(1, 1, n_steps))
        self.injection_scale = nn.Parameter(torch.tensor(0.1))
        self.last_gate = None

    @staticmethod
    def _time_linear(layer, sequence):
        return layer(sequence.transpose(1, 2)).transpose(1, 2)

    def forward(self, features, condition_sequence):
        if condition_sequence is None:
            self.last_gate = None
            return features#无条件直接返回
        if features.ndim != 3 or condition_sequence.ndim != 3:
            raise ValueError('features and condition_sequence must be (B,C,T)')#输入形状检查
        if features.shape[0] != condition_sequence.shape[0]:
            raise ValueError('feature and condition batch sizes differ')

        steps = features.shape[-1]#对齐时间长度
        condition_sequence = condition_sequence[..., :steps]
        condition_current = self._time_linear(#计算条件电流
            self.condition_projection, condition_sequence
        )
#计算gate值
        if self.injection_type == 'additive':
            gate = torch.ones_like(condition_current)
        elif self.injection_type == 'temporal_gate':
            condition_gate = self._time_linear(
                self.gate_projection, condition_sequence
            )
            gate = torch.sigmoid(
                self.feature_scale * features
                + condition_gate
                + self.temporal_bias[..., :steps]#门控大小公式
            )
        else:
            raise ValueError(f'unknown injection_type: {self.injection_type}')

        self.last_gate = gate.detach()
        return features + self.injection_scale * gate * condition_current#残差注入
