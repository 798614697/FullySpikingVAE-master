import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConditionAlignmentLoss(nn.Module):
    """让条件时间动态与后验潜变量放电概率在时间上对齐。

    condition_sequence 为 (B,E,T)，latent_probability 为 (B,Dz,T)。
    因二者通道数不同，先学习 E -> Dz 的映射，再比较绝对状态和相邻
    时间步变化量。变化量项用于减少“两者都退化为时间常数”的可能。
    """

    def __init__(self, condition_dim, latent_dim):
        super().__init__()
        self.condition_to_latent = nn.Linear(condition_dim, latent_dim)

    def forward(self, condition_sequence, latent_probability):
        if condition_sequence is None:
            return latent_probability.new_zeros(())
        # a_t = sigmoid(W_a c_t)，得到与 Bernoulli 概率相同的 [0,1] 目标。
        target = torch.sigmoid(
            self.condition_to_latent(
                condition_sequence.transpose(1, 2)
            ).transpose(1, 2)
        )
        # 状态对齐：当前条件活动应能解释当前潜变量放电概率。
        alignment = F.mse_loss(latent_probability, target)

        # Match changes as well as absolute activity, preventing the objective
        # from being satisfied only by a time-constant representation.
        if latent_probability.shape[-1] > 1:
            # 动态对齐：若条件活动在 t 时刻变化，潜变量概率也应产生相应变化。
            latent_delta = latent_probability[..., 1:] - latent_probability[..., :-1]
            target_delta = target[..., 1:] - target[..., :-1]
            alignment = alignment + F.mse_loss(latent_delta, target_delta)
        return alignment


class ConditionConsistencyClassifier(nn.Module):
    """仅用于训练约束的 CelebA 40 属性辅助分类器。

    它不参与 FS-CVAE 的推理/生成主干。真实图像分支教它识别属性；重建图
    和条件先验生成图分支把属性梯度传回生成模型。正式论文评估时更推荐
    另外训练并冻结一个独立属性分类器，避免“生成器和裁判共同适应”。
    """

    def __init__(self, in_channels, condition_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, condition_dim)

    def forward(self, image):
        # (B,C,H,W) -> (B,128,1,1) -> (B,40) logits；BCE 内部再做 sigmoid。
        return self.classifier(self.features(image).flatten(1))
