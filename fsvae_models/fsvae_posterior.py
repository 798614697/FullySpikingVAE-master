import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import global_v as glv

from .snn_layers import *
from .temporal_condition_gate import TemporalConditionGate


class PosteriorBernoulliSTBP(nn.Module):
    """自回归 Bernoulli 脉冲后验 q(z_t|x_<=t,z_<t,c_<=t)。

    注意：本文件的 x 已不是原图，而是 fsvae.py 编码后的 (B,Dz,T)
    latent feature。条件模型先用 c_1:T 门控该特征，再与历史潜变量拼接。
    输出 q_z 的形状是 (B,Dz,k,T)，sampled_z 为 (B,Dz,T)。
    """
    def __init__(self, k=20) -> None:
        """
        modeling of q(z_t | x_<=t, z_<t)
        """
        super().__init__()
        self.channels = glv.network_config['latent_dim']
        self.k = k
        self.n_steps = glv.network_config['n_steps']
        self.condition_dim = int(glv.network_config.get('condition_embed_dim', 0)) \
            if glv.network_config.get('conditional', False) else 0

        self.condition_gate = None
        if self.condition_dim:
            self.condition_gate = TemporalConditionGate(
                self.channels,
                self.condition_dim,
                self.n_steps,
                glv.network_config.get('injection_type', 'temporal_gate'),
            )

        # 输入通道为 2*Dz：前一半是图像 latent feature，后一半是 z 历史。
        # 最后一层输出 Dz*k 个 LIF 脉冲，k 个成员共同表示一个 Bernoulli 概率。
        self.layers = nn.Sequential(
            tdLinear(self.channels*2,
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

        self.is_true_scheduled_sampling = True

    def forward(self, x, condition_sequence=None):
        """
        input: 
            x:(B,C,T)
        returns: 
            sampled_z:(B,C,T)
            q_z: (B,C,k,T) # indicates q(z_t | x_<=t, z_<t) (t=1,...,T)
        """
        # 后验需要同时依赖编码特征 x_<=t 和历史潜变量 z_<t。
        # 代码先无梯度预采样历史 z，再用完整序列重算 q_z，以保留 tdBN/梯度路径。
        # q 的专属门控直接改变 posterior feature，因此条件能够改变 pi_q,t。
        if self.condition_gate is not None:
            x = self.condition_gate(x, condition_sequence)
        x_shape = x.shape # (B,C,T)
        batch_size=x_shape[0]
        random_indices = []
        # 阶段一：无梯度地逐步预采样 z_1,...,z_(T-1)，目的是为每个 t
        # 构造自回归条件 z_<t。该阶段不承担参数学习。
        with torch.no_grad():
            z_t_minus = self.initial_input.repeat(x_shape[0],1,1) # z_<t z0=zeros:(B,C,1)
            for t in range(self.n_steps-1):
                # 两条序列长度都为 t+1：x_0:t 与 [z0,z1,...,z_(t-1)]。
                inputs = torch.cat([x[...,:t+1].detach(), z_t_minus.detach()], dim=1) # (B,C+C,t+1) x_<=t and z_<t
                outputs = self.layers(inputs) #(B, C*k, t+1) 
                q_z_t = outputs[...,-1] # (B, C*k, 1) q(z_t | x_<=t, z_<t) 
                
                # 每个 latent channel 的 k 个 population 成员中随机选一个。
                # 若 k 个成员有 14 个为 1，则选到 1 的概率约为 14/k。
                random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k) #(B*C,) select 1 from every k value
                random_index = random_index.to(x.device)
                random_indices.append(random_index)

                z_t = q_z_t.view(batch_size*self.channels*self.k)[random_index] # (B*C,)
                z_t = z_t.view(batch_size, self.channels, 1) #(B,C,1)

                z_t_minus = torch.cat([z_t_minus, z_t], dim=-1) # (B,C,t+2)

        # 阶段二：用完整历史重新计算一次 q_z。这样 tdBatchNorm 能看到完整
        # T 维序列，且真正用于 loss 的 q_z 保留参数梯度。
        z_t_minus = z_t_minus.detach() # (B,C,T) z_0,...,z_{T-1}
        q_z = self.layers(torch.cat([x, z_t_minus], dim=1)) # (B,C*k,T)
        
        # 使用阶段一记录的随机索引从带梯度的 q_z 中重新取样；最后一个时间步
        # 在此处首次生成索引，因为阶段一只需要构造到 z_(T-1) 的历史。
        sampled_z = None
        for t in range(self.n_steps):
            
            if t == self.n_steps-1:
                # when t=T
                random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k)
                random_indices.append(random_index)
            else:
                # when t<=T-1
                random_index = random_indices[t]

            # sampling
            sampled_z_t = q_z[...,t].view(batch_size*self.channels*self.k)[random_index] # (B*C,)
            sampled_z_t = sampled_z_t.view(batch_size, self.channels, 1) #(B,C,1)
            if t==0:
                sampled_z = sampled_z_t
            else:
                sampled_z = torch.cat([sampled_z, sampled_z_t], dim=-1)
                
        # (B,Dz*k,T) -> (B,Dz,k,T)，显式恢复 population 维度。
        q_z = q_z.view(batch_size, self.channels, self.k, self.n_steps)# (B,C,k,T)

        return sampled_z, q_z
