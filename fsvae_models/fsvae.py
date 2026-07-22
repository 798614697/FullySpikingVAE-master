
import torch
import torch.nn as nn
from .snn_layers import *
from .fsvae_prior import *
from .fsvae_posterior import *
from .condition_encoder import TemporalConditionEncoder
from .temporal_condition_gate import TemporalConditionGate
from .temporal_loss import TemporalConditionAlignmentLoss, ConditionConsistencyClassifier
import torch.nn.functional as F

import global_v as glv


class FSVAE(nn.Module):
    """FSVAE/FS-CVAE 主模型。

    conditional=false 时完全退化为原始 FSVAE；conditional=true 时额外建立
    Condition Encoder 和四组独立门控（encoder/posterior/prior/decoder）。
    主张量约定：图像脉冲为 (B,C,H,W,T)，潜变量为 (B,Dz,T)。
    """
    def __init__(self):
        super().__init__()

        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']
        # condition_dim=0 是无条件模型的统一开关，避免维护两套 forward。
        self.condition_dim = int(glv.network_config.get('condition_dim', 0)) \
            if glv.network_config.get('conditional', False) else 0

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Encoder：空间维逐层下采样，时间维 T 始终保留在最后一维。
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike(),
                        is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        # prior 和 posterior 输出的最后第二维 k 是 Bernoulli population：
        # k 个二值神经元的均值近似某个 latent bit 的放电概率。
        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)

        # Decoder：先把 latent spike 投影回低分辨率特征图，再逐层上采样回图像大小。
        modules = []
        
        self.decoder_input = tdLinear(latent_dim,
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=glv.network_config['in_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()

        self._init_condition_modules(self.hidden_dims[-1] * 4)

    def _init_condition_modules(self, encoder_feature_dim):
        """只在 conditional=true 时创建条件分支及训练辅助头。

        四个门控具有独立参数，因此编码器、后验、先验、解码器可以学到
        不同的时间注入曲线，而不必在同一时刻以同样强度使用条件。
        """
        self.condition_embed_dim = int(glv.network_config.get(
            'condition_embed_dim', self.latent_dim
        )) if self.condition_dim else 0
        self.condition_encoder = None
        self.encoder_condition_gate = None
        self.decoder_condition_gate = None
        self.temporal_alignment = None
        self.condition_classifier = None
        self._last_condition_sequence = None
        self._last_condition = None
        self._last_prior_generated = None

        if not self.condition_dim:
            return

        mode = glv.network_config.get('temporal_condition_mode', 'lif')
        injection_type = glv.network_config.get('injection_type', 'temporal_gate')
        self.condition_encoder = TemporalConditionEncoder(
            self.condition_dim, self.condition_embed_dim, self.n_steps, mode
        )
        self.encoder_condition_gate = TemporalConditionGate(
            encoder_feature_dim, self.condition_embed_dim, self.n_steps,
            injection_type
        )
        self.decoder_condition_gate = TemporalConditionGate(
            self.latent_dim, self.condition_embed_dim, self.n_steps,
            injection_type
        )
        self.temporal_alignment = TemporalConditionAlignmentLoss(
            self.condition_embed_dim, self.latent_dim
        )
        self.condition_classifier = ConditionConsistencyClassifier(
            glv.network_config['in_channels'], self.condition_dim
        )

    def _prepare_condition(self, condition, batch_size, device, dtype):
        """检查条件形状，并兼容 CelebA 属性的 {-1,1} 或 {0,1} 表示。"""
        if self.condition_dim == 0:
            return None
        if condition is None:
            raise ValueError('CelebA attributes are required when conditional=true')
        if condition.ndim != 2 or condition.shape != (batch_size, self.condition_dim):
            raise ValueError(f'expected condition shape ({batch_size}, {self.condition_dim}), got {tuple(condition.shape)}')
        # >0 同时兼容旧 torchvision 的 {-1,1} 和当前版本返回的 {0,1}。
        return (condition.to(device=device, dtype=dtype) > 0).to(dtype)

    def forward(self, x, condition=None, scheduled=False):
        """训练/重建前向。

        返回：
            x_recon:  (B,C,H,W)，时间膜电位聚合后的静态重建图。
            q_z/p_z:  (B,Dz,k,T)，条件后验/先验的 population 输出。
            sampled_z:(B,Dz,T)，从后验 population 选出的二值潜变量。
        """
        condition = self._prepare_condition(condition, x.shape[0], x.device, x.dtype)
        # 静态 y -> 动态条件脉冲 c_1:T；无条件模型中保持 None。
        condition_sequence = self.condition_encoder(condition) if condition is not None else None
        self._last_condition = condition
        self._last_condition_sequence = condition_sequence
        sampled_z, q_z, p_z = self.encode(x, condition_sequence, scheduled)
        self._last_prior_generated = None
        # lambda_cc>0 时额外构造一张“条件先验潜变量率”生成图，供属性一致性
        # 损失监督。只在训练阶段做，避免普通测试前向增加一倍解码开销。
        use_cc_prior = (self.training and self.condition_dim
                        and float(glv.network_config.get('lambda_cc', 0.0)) > 0)
        if use_cc_prior:
            # p_z is a k-member Bernoulli population. Its mean is a
            # differentiable conditional-prior latent rate used for the
            # condition-consistency generation branch.
            prior_latent_rate = p_z.mean(dim=2)
            # 合并成一个 2B batch 只调用一次 decoder，让两条分支共享同一次
            # BatchNorm 统计；再沿 batch 维切回重建图和先验生成图。
            joint_z = torch.cat([sampled_z, prior_latent_rate], dim=0)
            joint_condition = torch.cat(
                [condition_sequence, condition_sequence], dim=0
            )
            joint_output = self.decode(joint_z, joint_condition)
            x_recon, self._last_prior_generated = joint_output.chunk(2, dim=0)
        else:
            x_recon = self.decode(sampled_z, condition_sequence)
        return x_recon, q_z, p_z, sampled_z
    
    def encode(self, x, condition_sequence=None, scheduled=False):
        """图像编码、后验采样以及条件先验拟合的完整编码路径。"""
        x = self.encoder(x) # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3) # (N,C*H*W,T)
        if self.encoder_condition_gate is not None:
            x = self.encoder_condition_gate(x, condition_sequence)
        # 空间特征先受条件门控，再由 tdLinear+LIF 压到 latent_dim。
        latent_x = self.before_latent_layer(x) # (N,latent_dim,T)
        # posterior 内部还会结合历史 z_<t，并进行第二组独立条件门控。
        sampled_z, q_z = self.posterior(latent_x, condition_sequence)

        # prior 只看已经采样的 z 历史，用于逼近后验的时间条件分布。
        p_z = self.prior(sampled_z, scheduled, self.p, condition_sequence)
        return sampled_z, q_z, p_z

    def decode(self, z, condition_sequence=None):
        """把潜变量脉冲序列解码为静态图像。

        z 在进入 decoder_input 前先受时间条件门控；随后恢复为 2x2 特征图，
        逐层反卷积上采样。MembraneOutputLayer 最后才压缩时间维。
        """
        if self.decoder_condition_gate is not None:
            z = self.decoder_condition_gate(z, condition_sequence)
        result = self.decoder_input(z) # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        # MembraneOutputLayer 把时间维上的膜电位累积成普通图像，再映射到 [-1, 1]。
        out = torch.tanh(self.membrane_output_layer(result))        
        return out

    def sample(self, batch_size=64, condition=None):
        """不使用输入图像，从 p(z|y) 自回归采样并生成图片。

        条件模型最好显式传入真实/人工构造的合法属性组合。condition=None 的
        随机 40 bit 仅为兼容旧 FID 接口，可能形成现实中罕见的属性组合。
        """
        if self.condition_dim:
            if condition is None:
                # Backward-compatible fallback for metric helpers. Supplying real
                # CelebA attributes is preferable because attributes are correlated.
                condition = torch.randint(0, 2, (batch_size, self.condition_dim),
                                          device=self.prior.initial_input.device).float()
            condition = self._prepare_condition(condition, batch_size,
                                                self.prior.initial_input.device,
                                                self.prior.initial_input.dtype)
        condition_sequence = self.condition_encoder(condition) if condition is not None else None
        sampled_z = self.prior.sample(batch_size, condition_sequence)
        sampled_x = self.decode(sampled_z, condition_sequence)
        return sampled_x, sampled_z

    def _extra_conditional_losses(self, input_img, recons_img, latent_prob):
        """计算条件一致性 CC 和条件-潜变量时间对齐 TEMP 两项附加损失。"""
        zero = recons_img.new_zeros(())
        if not self.condition_dim:
            return zero, zero

        cc_loss = zero
        if float(glv.network_config.get('lambda_cc', 0.0)) > 0:
            # The real-image term trains the auxiliary classifier; the
            # reconstruction term also sends attribute gradients to FS-CVAE.
            real_logits = self.condition_classifier(input_img.detach())
            recon_logits = self.condition_classifier(recons_img)
            # real 项训练分类器；recon/prior 项既训练分类器，也向生成主干传梯度。
            terms = [
                F.binary_cross_entropy_with_logits(real_logits, self._last_condition),
                F.binary_cross_entropy_with_logits(recon_logits, self._last_condition),
            ]
            if self._last_prior_generated is not None:
                prior_logits = self.condition_classifier(self._last_prior_generated)
                terms.append(F.binary_cross_entropy_with_logits(
                    prior_logits, self._last_condition
                ))
            cc_loss = torch.stack(terms).mean()

        temp_loss = zero
        if float(glv.network_config.get('lambda_temp', 0.0)) > 0:
            temp_loss = self.temporal_alignment(
                self._last_condition_sequence, latent_prob
            )
        return cc_loss, temp_loss
        
    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        # population mean：k 个二值成员的平均值近似 Bernoulli 参数 pi。
        q_z_ber = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2) # (N, latent_dim, T)

        # PSP 先对时间脉冲做突触后电位滤波，再匹配 q(z|x,y) 与 p(z|y)。
        mmd_loss = torch.mean((self.psp(q_z_ber)-self.psp(p_z_ber))**2)
        cc_loss, temp_loss = self._extra_conditional_losses(
            input_img, recons_img, q_z_ber
        )
        loss = (recons_loss
                + float(glv.network_config.get('lambda_cond', 1.0)) * mmd_loss
                + float(glv.network_config.get('lambda_cc', 0.0)) * cc_loss
                + float(glv.network_config.get('lambda_temp', 0.0)) * temp_loss)
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'Distance_Loss': mmd_loss, 'Condition_Loss': cc_loss,
                'Temporal_Loss': temp_loss}

    def loss_function_kld(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        prob_q = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        prob_p = torch.mean(p_z, dim=2) # (N, latent_dim, T)
        
        # Bernoulli KL；1e-2 数值平滑避免 log(0)。若配置从 MMD 改为 KLD，
        # lambda_cond 通常也应从 1.0 调小到原论文使用的约 1e-4。
        kld_loss = prob_q * torch.log((prob_q+1e-2)/(prob_p+1e-2)) + (1-prob_q)*torch.log((1-prob_q+1e-2)/(1-prob_p+1e-2))
        kld_loss = torch.mean(torch.sum(kld_loss, dim=(1,2)))

        cc_loss, temp_loss = self._extra_conditional_losses(
            input_img, recons_img, prob_q
        )
        cond_weight = float(glv.network_config.get('lambda_cond', 1e-4))
        loss = (recons_loss + cond_weight * kld_loss
                + float(glv.network_config.get('lambda_cc', 0.0)) * cc_loss
                + float(glv.network_config.get('lambda_temp', 0.0)) * temp_loss)
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'Distance_Loss': kld_loss, 'Condition_Loss': cc_loss,
                'Temporal_Loss': temp_loss}

    def condition_gate_means(self):
        """返回各门控沿 batch/通道平均后的 (T,) 曲线，仅用于可视化。"""
        gates = {}
        candidates = {
            'encoder': self.encoder_condition_gate,
            'posterior': getattr(self.posterior, 'condition_gate', None),
            'prior': getattr(self.prior, 'condition_gate', None),
            'decoder': self.decoder_condition_gate,
        }
        for name, module in candidates.items():
            if module is not None and module.last_gate is not None:
                gates[name] = module.last_gate.mean(dim=(0, 1))
        return gates
    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4,4)

    def update_p(self, epoch, max_epoch):
        # scheduled sampling 的概率日程：从 0.1 线性增加到约 0.3。
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p-init_p) * epoch / max_epoch + init_p
        

class FSVAELarge(FSVAE):
    """用于 64x64 CelebA/CIFAR 的五层大模型。

    与 FSVAE 的算法完全相同，只把 encoder hidden dims 从四层扩为
    [32,64,128,256,512]，使 64x64 输入最终降采样到 512x2x2。
    """
    def __init__(self):
        super(FSVAE, self).__init__()
        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']
        self.condition_dim = int(glv.network_config.get('condition_dim', 0)) \
            if glv.network_config.get('conditional', False) else 0

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)
        
        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim,
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=glv.network_config['in_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()

        self._init_condition_modules(self.hidden_dims[-1] * 4)
