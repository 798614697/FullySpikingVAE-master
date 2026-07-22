from argparse import ZERO_OR_MORE
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T

import global_v as glv

from .snn_layers import *
from .temporal_condition_gate import TemporalConditionGate


class PriorBernoulliSTBP(nn.Module):
    """条件自回归 Bernoulli 脉冲先验 p(z_t|z_<t,c_<=t)。

    先验不看原图 x，这是生成阶段可以只凭 y 生成图片的关键。训练时它用
    后验 sampled_z 的移位历史作为 teacher forcing 输入；生成时则只能使用
    自己之前生成的 z。输出 (B,Dz,k,T)，k 维均值近似 pi_p,t。
    """
    def __init__(self, k=20) -> None:
        """
        modeling of p(z_t|z_<t)
        """
        super().__init__()
        self.channels = glv.network_config['latent_dim']
        self.condition_dim = int(glv.network_config.get('condition_embed_dim', 0)) \
            if glv.network_config.get('conditional', False) else 0
        self.k = k
        self.n_steps = glv.network_config['n_steps']

        # 与 posterior 不同，prior 的基本输入只有 Dz 维历史 z；条件 c_t
        # 通过 TemporalConditionGate 调制，不通过 concat 增加输入通道。
        self.layers = nn.Sequential(
            tdLinear(self.channels,
                    self.channels*2,
                    bias=True,
                    bn=tdBatchNorm(self.channels*2, alpha=2), 
                    spike=LIFSpike()),
            tdLinear(self.channels*2,
                    self.channels*4,
                    bias=True,
                    bn=tdBatchNorm(self.channels*4, alpha=2),
                    spike=LIFSpike()),
            tdLinear(self.channels*4,
                    self.channels*k,
                    bias=True,
                    bn=tdBatchNorm(self.channels*k, alpha=2),
                    spike=LIFSpike())
        )
        self.register_buffer('initial_input', torch.zeros(1, self.channels, 1))# (1,C,1)
        self.condition_gate = None
        if self.condition_dim:
            self.condition_gate = TemporalConditionGate(
                self.channels,
                self.condition_dim,
                self.n_steps,
                glv.network_config.get('injection_type', 'temporal_gate'),
            )


    def _with_condition(self, z, condition_sequence):
        """统一封装有/无条件分支；自回归短序列切片在 gate 内完成。"""
        if self.condition_gate is None:
            return z
        if condition_sequence is None:
            raise ValueError('condition is required when conditional=true')
        return self.condition_gate(z, condition_sequence)

    def forward(self, z, scheduled=False, p=None, condition=None):
        """训练/测试入口；scheduled 决定是否启用混合 teacher forcing。"""
        if scheduled:
            return self._forward_scheduled_sampling(z, p, condition)
        else:
            return self._forward(z, condition)
    
    def _forward(self, z, condition=None):
        """
        input z: (B,C,T) # latent spike sampled from posterior
        output : (B,C,k,T) # indicates p(z_t|z_<t) (t=1,...,T)
        """
        # 训练 prior 时断开历史 z 的梯度：分布匹配损失更新 prior 本身，但不
        # 沿“作为 prior 输入的 sampled_z”这条旁路反向干扰 posterior。
        z_shape = z.shape # (B,C,T)
        batch_size = z_shape[0]
        z = z.detach()

        # teacher forcing 移位：t=1 输入 z0=0，t=2 输入 z1，...，t=T 输入 zT-1。
        z0 = self.initial_input.repeat(batch_size, 1, 1) # (B,C,1)
        inputs = torch.cat([z0, z[...,:-1]], dim=-1) # (B,C,T)
        outputs = self.layers(self._with_condition(inputs, condition)) # (B,C*k,T)
        
        p_z = outputs.view(batch_size, self.channels, self.k, self.n_steps) # (B,C,k,T)
        return p_z

    def _forward_scheduled_sampling(self, z, p, condition=None):
        """
        use scheduled sampling
        input 
            z: (B,C,T) # latent spike sampled from posterior
            p: float # prob of scheduled sampling
        output : (B,C,k,T) # indicates p(z_t|z_<t) (t=1,...,T)
        """
        z_shape = z.shape # (B,C,T)
        batch_size = z_shape[0]
        z = z.detach()

        # scheduled sampling：部分时间步用 prior 自己采样的 z_t 继续滚动，
        # 缓解“训练看真实后验历史、生成看自身错误历史”的 exposure bias。
        z_t_minus = self.initial_input.repeat(batch_size,1,1) # z_<t, z0=zeros:(B,C,1)
        if self.training:
            with torch.no_grad():
                for t in range(self.n_steps-1):
                    # 前五步保持 teacher forcing 以稳定序列起点；之后以概率 p
                    # 换成 prior 自己的输出。p 由 fsvae.update_p 随 epoch 增长。
                    if t>=5 and random.random() < p: # scheduled sampling                    
                        outputs = self.layers(self._with_condition(z_t_minus.detach(), condition))
                        p_z_t = outputs[...,-1] # (B, C*k, 1)
                        # sampling from p(z_t | z_<t)
                        # population mean 作为 pi_p,t，加微噪声后以 0.5 硬阈值取样。
                        prob1 = p_z_t.view(batch_size, self.channels, self.k).mean(-1) # (B,C)
                        prob1 = prob1 + 1e-3 * torch.randn_like(prob1) 
                        z_t = (prob1>0.5).float() # (B,C)
                        z_t = z_t.view(batch_size, self.channels, 1) #(B,C,1)
                        z_t_minus = torch.cat([z_t_minus, z_t], dim=-1) # (B,C,t+2)
                    else:
                        z_t_minus = torch.cat([z_t_minus, z[...,t].unsqueeze(-1)], dim=-1) # (B,C,t+2)
        else: # for test time
            z_t_minus = torch.cat([z_t_minus, z[:,:,:-1]], dim=-1) # (B,C,T)

        z_t_minus = z_t_minus.detach() # (B,C,T) z_{<=T-1} 
        p_z = self.layers(self._with_condition(z_t_minus, condition)) # (B,C*k,T)
        p_z = p_z.view(batch_size, self.channels, self.k, self.n_steps)# (B,C,k,T)
        return p_z

    def sample(self, batch_size=64, condition=None):
        """free-running 生成：从 z0=0 开始，逐步产生完整 z_1:T。"""
        # 生成时只能依赖 prior 自己的历史输出，因此这里逐时间步自回归采样。
        z_minus_t = self.initial_input.repeat(batch_size, 1, 1) # (B, C, 1)
        for t in range(self.n_steps):
            outputs = self.layers(self._with_condition(z_minus_t, condition))
            p_z_t = outputs[...,-1] # (B, C*k, 1)

            # 和 posterior 相同：每个 latent channel 从对应的 k 个成员选一个，
            # 因而 population 中 1 的比例决定实际采到脉冲 1 的概率。
            random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k) #(B*C,) pick one from k
            random_index = random_index.to(z_minus_t.device)

            z_t = p_z_t.view(batch_size*self.channels*self.k)[random_index] # (B*C,)
            z_t = z_t.view(batch_size, self.channels, 1) #(B,C,1)
            z_minus_t = torch.cat([z_minus_t, z_t], dim=-1) # (B,C,t+2)

        
        sampled_z = z_minus_t[...,1:] # (B,C,T)

        return sampled_z
