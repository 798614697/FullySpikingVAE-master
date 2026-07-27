# FSVAE / FS-CVAE 中文代码导读

本文档按照程序真正执行的顺序解释代码。建议先读本文件建立全局概念，
再沿文中给出的函数顺序阅读源代码。

## 1. 推荐阅读顺序

1. `NetworkConfigs/CelebA_CVAE.yaml`：先知道模型将使用哪些开关。
2. `main_fsvae.py` 的 `if __name__ == '__main__'`：了解实验如何启动。
3. `datasets/load_dataset_snn.py::load_celebA`：了解一个 batch 的内容。
4. `fsvae_models/fsvae.py::FSVAE.forward`：掌握主干数据流。
5. `condition_encoder.py` 和 `temporal_condition_gate.py`：理解创新条件分支。
6. `fsvae_posterior.py::forward`：理解如何从 q 中采样 z。
7. `fsvae_prior.py`：理解训练先验与无图像生成。
8. `fsvae.py::loss_function_mmd`：理解四项损失如何汇合。
9. `snn_layers.py`：最后下钻到 LIF、tdLinear、tdConv 的实现。

## 2. 文件职责速查

| 文件 | 作用 |
|---|---|
| `main_fsvae.py` | 训练、测试、采样、日志、保存 checkpoint |
| `network_parser.py` | 从 YAML 读取字典 |
| `global_v.py` | 把配置保存成模型可访问的全局状态 |
| `datasets/load_dataset_snn.py` | 加载静态图片及标签/属性 |
| `fsvae_models/fsvae.py` | 编码器、条件分支整合、解码器和损失 |
| `condition_encoder.py` | 静态 y -> 条件脉冲 c_1:T |
| `temporal_condition_gate.py` | 按时间和特征状态控制条件注入强度 |
| `fsvae_posterior.py` | q(z_t|x_<=t,z_<t,c_<=t) |
| `fsvae_prior.py` | p(z_t|z_<t,c_<=t) |
| `temporal_loss.py` | 条件一致性分类器和时间对齐损失 |
| `snn_layers.py` | 硬脉冲、LIF、时域线性/卷积、PSP、输出膜电位 |
| `metrics/` | FID、Inception Score、特征空间 Frechet 距离 |

## 3. 从命令行到模型

训练完整 FS-CVAE：

```bash
python main_fsvae.py celeba_fscvae \
  -config NetworkConfigs/CelebA_CVAE.yaml \
  -device 0
```

`main_fsvae.py` 首先解析出：

- `name=celeba_fscvae`：决定输出目录 `checkpoint/celeba_fscvae/`；
- `config=...yaml`：决定模型结构和训练超参数；
- `device=0`：决定使用 `cuda:0`。

随后执行：

```text
parse(YAML)
-> glv.init(config)
-> load_celebA(data_path)
-> FSVAELarge()
-> AdamW(model.parameters())
-> epoch loop
```

必须先 `glv.init` 再构造模型，因为各层构造函数会读取全局的 `latent_dim`、
`n_steps`、`condition_embed_dim` 等字段。

## 4. 一个 CelebA batch

DataLoader 返回：

```text
real_img: (B,3,64,64)，float，范围 [-1,1]
labels:   (B,40)，CelebA 二值属性
```

`target_type='attr'` 表示 `labels` 来自 `list_attr_celeba.txt`，例如 Smiling、
Male、Eyeglasses、Young 等。它不是单一类别编号。

进入模型前，`main_fsvae.train` 执行 direct input：

```python
spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, T)
```

于是：

```text
(B,3,64,64) -> (B,3,64,64,16)
```

这一步只是把同一幅模拟像素图复制到每个时间步；第一个卷积层和后续 LIF
负责把输入转换为脉冲活动。

## 5. FSVAELarge 图像编码器

五个 stride=2 的 `tdConv + tdBatchNorm + LIFSpike` 依次执行：

```text
(B,  3,64,64,T)
-> (B, 32,32,32,T)
-> (B, 64,16,16,T)
-> (B,128, 8, 8,T)
-> (B,256, 4, 4,T)
-> (B,512, 2, 2,T)
```

展平空间维后得到 `(B,2048,T)`。卷积核时间大小恒为 1，所以卷积本身
不混合不同时间；跨时间记忆来自每层 LIF 的膜电位。

## 6. 静态属性如何变成条件时间脉冲

`TemporalConditionEncoder` 接收 `(B,40)`，先投影为 `(B,64)` 的基础电流，
再逐时间步运行递归 LIF：

```text
y -> W_y y
        + time_embedding[t]
        + W_r c_(t-1)
-> condition membrane u_t
-> threshold
-> c_t
```

最终输出 `(B,64,T)`。因此 c_t 不是把原始标签机械复制 T 次，而是由
条件膜电位历史和前一时刻放电共同决定。

## 7. 时间选择性门控

每个门控接收主干特征 `(B,D,T)` 和条件脉冲 `(B,64,T)`：

```text
condition_current = W_c c_t
gate = sigmoid(feature_scale*h_t + W_g*c_t + time_bias[t])
output = h_t + alpha*gate*condition_current
```

`gate` 是 `(B,D,T)`，所以不同样本、通道、时间步都有独立强度。当前是
连续软门控，不是纯二值事件门控。输出随后进入下一层 LIF，间接改变其
输入电流、膜电位积累和放电时间。

四个位置分别有独立门控参数：

1. encoder gate：调制 `(B,2048,T)` 图像特征；
2. posterior gate：调制 `(B,128,T)` 后验输入；
3. prior gate：调制 `(B,128,T)` 历史潜变量；
4. decoder gate：调制 `(B,128,T)` 解码输入。

## 8. 条件后验 q(z|x,y)

编码特征经 `before_latent_layer` 变成 `(B,128,T)`。后验还需要历史潜变量，
所以其网络实际输入为：

```text
cat([latent_feature, shifted_z_history], channel_dim)
-> (B,256,T)
```

这里拼接的是后验状态和自回归历史，不是静态标签拼接。条件已经通过
`condition_sequence` 和门控进入 latent feature。

后验最后输出 `(B,128*k,T)`，再 reshape 为 `(B,128,k,T)`。默认 k=20。
对 k 维求平均可得到 `[0,1]` 内的近似 Bernoulli 参数；随机选其中一个
population 成员得到实际二值 `sampled_z`。

后验采用两阶段算法：

1. 在 `no_grad` 中逐步预采样历史 z，解决 q_t 需要 z_<t 的问题；
2. 用完整历史重新计算 q_z，让 tdBatchNorm 和真正的损失路径保留梯度。

## 9. 条件先验 p(z|y)

训练时，先验把后验 `sampled_z` 右移一位：

```text
prior input at t: z_(t-1)
t=1 使用全零 z0
```

历史 z 会 detach，之后由 prior gate 使用 c_t 调制，再产生 `(B,128,k,T)`
的 p_z。分布损失要求 p_z 接近 q_z。

Scheduled sampling 会在较晚时间步以一定概率用 prior 自己的输出替换后验
历史，降低 teacher forcing 与实际生成之间的差距。

真正生成时没有 x 和 q：

```text
y -> c_1:T
z0=0
-> p(z1|z0,c1) -> z1
-> p(z2|z0,z1,c_1:2) -> z2
...
-> z_1:T
```

## 10. 解码器

后验重建或先验采样得到的 `(B,128,T)` 先经过 decoder gate，再执行：

```text
(B,128,T)
-> tdLinear
-> (B,2048,T)
-> reshape (B,512,2,2,T)
-> 多层 tdConvTranspose
-> (B,3,64,64,T)
-> MembraneOutputLayer
-> tanh
-> (B,3,64,64)
```

只有最后的 `MembraneOutputLayer` 才压缩 T，它以指数权重累加各时间步输出，
模拟一个不放电读出神经元的末端膜电位。

## 11. 四项损失

完整配置使用：

```text
L = L_rec + L_MMD + 0.1*L_CC + 0.1*L_temp
```

### L_rec

真实图与后验重建图的 MSE，保证基本重建能力。

### L_MMD / L_cond_dist

先将 q_z、p_z 沿 k 求均值为 `(B,128,T)`，再经过 PSP 时间滤波，最后计算
均方距离。它要求 `q(z|x,y)` 能被不看 x 的 `p(z|y)` 模拟。

### L_CC

辅助分类器预测真实图、后验重建图、条件先验生成图的 40 个属性并计算
BCE。真实图教会分类器识别属性，另外两条路径把属性监督传给生成模型。

### L_temp

把 `(B,64,T)` 条件脉冲投影到 `(B,128,T)`，与后验 population mean 同时
比较绝对状态和相邻时间差分，促使条件动态真正对应潜变量动态。

## 12. 配置与消融

| 配置 | 含义 |
|---|---|
| `CelebA.yaml` | 原始无条件 FSVAE |
| `CelebA_CVAE_Latent.yaml` | 静态投影、始终开门、无新增损失 |
| `CelebA_CVAE_Temporal_Additive.yaml` | LIF 时间编码、直接相加、无新增损失（隔离门控作用） |
| `CelebA_CVAE_Temporal.yaml` | LIF 条件时间编码和软门控、无新增损失 |
| `CelebA_CVAE.yaml` | 时间编码、软门控和全部提出损失 |

`Latent -> Temporal_Additive` 隔离时间编码作用，
`Temporal_Additive -> Temporal` 隔离软门控作用，
`Temporal -> Full` 隔离新增损失作用。只有统一训练设置并逐级比较，
才能证明时间编码、选择性注入和新增损失各自有效。

## 13. TensorBoard 中看什么

```bash
tensorboard --logdir checkpoint
```

主要曲线：

- `Train/recons_loss`：重建误差；
- `Train/distance`：q/p 分布距离；
- `Train/condition_consistency`：属性 BCE；
- `Train/temporal_alignment`：时间对齐；
- `Train/latent_firing_rate`：潜变量平均放电率；
- `Train/condition_gate_*/t*`：四个模块在每个时间步的平均门值。

如果所有门值长期相同，说明时间选择性可能没有被真正使用；如果全接近 0，
条件被忽略；如果全接近 1，则退化为始终注入。

## 14. Checkpoint 和输出目录

每个实验输出：

```text
checkpoint/<name>/best.pth          测试 loss 最低的 state_dict
checkpoint/<name>/checkpoint.pth    按间隔保存的最新 state_dict
checkpoint/<name>/imgs/             输入、重建、采样图片
checkpoint/<name>/tb/               TensorBoard event 文件
checkpoint/<name>/<name>.log        文本日志
```

checkpoint 当前不包含 optimizer、epoch 和随机数状态，所以可以加载模型做
推理，但不能精确恢复一次被中断训练的全部状态。

## 15. 当前实现边界

1. 时间门控仍使用 sigmoid 和连续逐元素乘法，不是纯事件门控。
2. 训练用属性分类器与生成器联合学习；论文评价应另用冻结的独立分类器。
3. 训练中的条件先验一致性使用可微 population mean，而非完整离散祖先采样。
4. 现有操作量 hook 没有完整统计门控逐元素运算。
5. 默认关闭昂贵 FID；正式实验需要对所有对照统一开启并使用相同样本数。

这些边界不影响代码运行，但在解释实验结论和撰写论文时必须明确说明。
