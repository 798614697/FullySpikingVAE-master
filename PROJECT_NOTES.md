# 项目批注：Fully Spiking VAE

## 一句话概览

该项目是 AAAI 2022 论文 **Fully Spiking Variational Autoencoder** 的官方实现，核心目标是把 VAE 的编码器、潜变量采样、先验模型和解码器都放进脉冲神经网络时间维度中运行。

## 代码主线

1. `main_fsvae.py` 是 FSVAE 的训练入口：读取 YAML 配置、加载数据、构建模型、训练/测试、保存样本和计算指标。
2. `fsvae_models/fsvae.py` 定义完整 FSVAE：卷积编码器把图像序列编码为脉冲潜变量，后验网络采样 `z`，先验网络建模 `p(z_t | z_<t)`，最后由转置卷积解码回图像。
3. `fsvae_models/snn_layers.py` 提供时序层：`tdConv`、`tdLinear`、`tdBatchNorm`、`LIFSpike` 等。项目大量张量都带最后一维 `T`，表示脉冲时间步。
4. `fsvae_models/fsvae_posterior.py` 建模 `q(z_t | x_<=t, z_<t)`，也就是根据当前和过去的编码信息，以及过去潜变量，采样当前潜变量。
5. `fsvae_models/fsvae_prior.py` 建模 `p(z_t | z_<t)`，训练时可使用 scheduled sampling，生成时从先验逐时间步采样。
6. `datasets/load_dataset_snn.py` 加载 MNIST/FashionMNIST/CIFAR10/CelebA，并把图像归一化到 `[-1, 1]`，对应模型输出端的 `tanh`。

## 关键张量形状

- 图像输入在训练入口中从 `(N, C, H, W)` 扩展为 `(N, C, H, W, T)`。
- 编码器输出仍保留时间维：`(N, C, H, W, T)`。
- 展平后进入潜变量层：`(N, C*H*W, T)` -> `(N, latent_dim, T)`。
- 后验/先验分布张量：`q_z`、`p_z` 形状为 `(N, latent_dim, k, T)`。
- 实际采样出的潜变量脉冲：`sampled_z` 形状为 `(N, latent_dim, T)`。
- 解码器输出在膜电位汇聚后回到普通图像：`(N, C, H, W)`。

## 训练流程批注

- 每个 batch 先把静态图像复制到所有时间步，这里的输入不是泊松编码，而是 direct spike input。
- 损失由重建项和距离项组成，距离项可选 `mmd` 或 `kld`。
- `scheduled: true` 时，每个 epoch 调用 `net.update_p`，逐步提高先验网络使用自身预测历史的概率。
- 测试阶段会注册 hook 统计乘加操作，用于衡量 SNN 计算量。
- 每个 epoch 都会保存重建图、采样图、checkpoint，并计算 Inception Score、Autoencoder Frechet Distance 和 clean FID。

## 阅读提醒

- 这个代码包依赖全局配置 `global_v.network_config`，很多模块初始化时会直接读取它；因此必须先调用 `glv.init(...)` 再构建模型。
- `tdConv`/`tdConvTranspose` 实际用的是 `Conv3d`，但时间维 kernel/stride/padding 固定为 1/1/0，只在空间维做卷积。
- `SpikeAct.backward` 用矩形窗口近似梯度，这是脉冲神经网络中常见的 surrogate gradient 思路。
- `FSVAELarge.__init__` 中使用 `super(FSVAE, self).__init__()`，这是原代码写法；如果后续维护，建议重点检查这里是否符合预期。

