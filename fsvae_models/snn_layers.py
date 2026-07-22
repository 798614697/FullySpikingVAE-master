import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import global_v as glv

"""FSVAE 使用的基础时域 SNN 层。

全项目统一把时间 T 放在最后一维。tdLinear/tdConv 对每个时间步共享同一组
权重；真正的跨时间记忆来自 LIF 膜电位，而不是时间卷积核。
"""

dt = 5
a = 0.25
aa = 0.5  
Vth = 0.2
tau = 0.25


class SpikeAct(torch.autograd.Function):
    """ 
        Implementation of the spiking activation function with an approximation of gradient.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        # 前向是不可导的硬阈值事件：膜电位超过 0.2 才输出 spike=1。
        output = torch.gt(input, Vth) 
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors 
        grad_input = grad_output.clone()
        # surrogate gradient：只在阈值附近的小窗口内传递近似梯度；这是
        # STBP 能训练离散脉冲神经元的关键。
        hu = abs(input) < aa
        hu = hu.float() / (2 * aa)
        return grad_input * hu

class LIFSpike(nn.Module):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self):
        super(LIFSpike, self).__init__()

    def forward(self, x):
        # 最后一维是时间步 T；逐时间步更新膜电位并产生脉冲。
        nsteps = x.shape[-1]
        u   = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        # 必须显式按时间循环，因为 u_t 依赖 u_(t-1) 和上一时刻输出。
        for step in range(nsteps):
            u, out[..., step] = self.state_update(u, out[..., max(step-1, 0)], x[..., step])
        return out
    
    def state_update(self, u_t_n1, o_t_n1, W_mul_o_t1_n, tau=tau):
        # u_t = tau*u_(t-1)*(1-o_(t-1)) + input_t；放电后通过乘零软复位。
        u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
        o_t1_n1 = SpikeAct.apply(u_t1_n1)
        return u_t1_n1, o_t1_n1  

class tdLinear(nn.Linear):
    """对 (B,C,T) 的每个时间步应用同一个 nn.Linear。"""
    def __init__(self, 
                in_features,
                out_features,
                bias=True,
                bn=None,
                spike=None):
        assert type(in_features) == int, 'inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape)
        assert type(out_features) == int, 'outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape)

        super(tdLinear, self).__init__(in_features, out_features, bias=bias)

        self.bn = bn
        self.spike = spike
        

    def forward(self, x):
        """
        x : (N,C,T)
        """        
        # F.linear 作用在最后一维，所以先把通道维换到末尾，计算后再换回来。
        x = x.transpose(1, 2) # (N, T, C)
        y = F.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)# (N, C, T)
        
        if self.bn is not None:
            y = y[:,:,None,None,:]
            y = self.bn(y)
            y = y[:,:,0,0,:]
        if self.spike is not None:
            y = self.spike(y)
        return y

class tdConv(nn.Conv3d):
    """对 (B,C,H,W,T) 做空间卷积，时间核恒为 1。

    Conv3d 只被用作处理额外时间维的工程实现；kernel=(kh,kw,1)，所以
    不会在卷积中混合不同时间步，时间依赖仍由后接 LIF 建立。
    """
    def __init__(self, 
                in_channels, 
                out_channels,  
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                bn=None,
                spike=None,
                is_first_conv=False):

        # kernel/stride/padding 的时间维固定为 1/1/0，只在 H/W 上卷积。
        # 因此这里借 Conv3d 处理 (N,C,H,W,T)，但不混合不同时间步。
        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        super(tdConv, self).__init__(in_channels, out_channels, kernel, stride, padding, dilation, groups,
                                        bias=bias)
        self.bn = bn
        self.spike = spike
        self.is_first_conv = is_first_conv

    def forward(self, x):
        x = F.conv3d(x, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
        if self.bn is not None:
            x = self.bn(x)
        if self.spike is not None:
            x = self.spike(x)
        return x
        

class tdConvTranspose(nn.ConvTranspose3d):
    """tdConv 的空间上采样对应层，同样保持时间长度 T 不变。"""
    def __init__(self, 
                in_channels, 
                out_channels,  
                kernel_size,
                stride=1,
                padding=0,
                output_padding=0,
                dilation=1,
                groups=1,
                bias=True,
                bn=None,
                spike=None):

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))


        # output padding
        if type(output_padding) == int:
            output_padding = (output_padding, output_padding, 0)
        elif len(output_padding) == 2:
            output_padding = (output_padding[0], output_padding[1], 0)
        else:
            raise Exception('output_padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        super().__init__(in_channels, out_channels, kernel, stride, padding, output_padding, groups,
                                        bias=bias, dilation=dilation)

        self.bn = bn
        self.spike = spike

    def forward(self, x):
        x = F.conv_transpose3d(x, self.weight, self.bias,
                        self.stride, self.padding, 
                        self.output_padding, self.groups, self.dilation)

        if self.bn is not None:
            x = self.bn(x)
        if self.spike is not None:
            x = self.spike(x)
        return x

class tdBatchNorm(nn.BatchNorm2d):
    """
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        # tdBN 在 batch、空间和时间维上共同统计均值方差，每个通道单独归一化。
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * Vth * (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
        
        return input


class PSP(torch.nn.Module):
    """一阶突触后电位滤波器，用于比较 q/p 的时间脉冲响应而非孤立 bit。"""
    def __init__(self):
        super().__init__()
        self.tau_s = 2

    def forward(self, inputs):
        """
        inputs: (N, C, T)
        """
        # 简单一阶突触滤波，把离散 spike 序列转成随时间衰减/积累的响应。
        syns = None
        syn = 0
        n_steps = inputs.shape[-1]
        for t in range(n_steps):
            syn = syn + (inputs[...,t] - syn) / self.tau_s
            if syns is None:
                syns = syn.unsqueeze(-1)
            else:
                syns = torch.cat([syns, syn.unsqueeze(-1)], dim=-1)

        return syns

class MembraneOutputLayer(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    """
    def __init__(self) -> None:
        super().__init__()
        n_steps = glv.n_steps

        arr = torch.arange(n_steps-1,-1,-1)
        # 越靠后的时间步权重越大；register_buffer 让系数随模型迁移设备，
        # 但不作为可训练参数进入优化器。
        self.register_buffer("coef", torch.pow(0.8, arr)[None,None,None,None,:]) # (1,1,1,1,T)

    def forward(self, x):
        """
        x : (N,C,H,W,T)
        """
        # 按时间衰减加权求和，相当于取输出神经元的末端膜电位。
        out = torch.sum(x*self.coef, dim=-1)
        return out
