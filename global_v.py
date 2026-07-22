import torch

"""项目级全局配置容器。

模型构造函数直接读取 glv.network_config，因此调用顺序必须是：解析 YAML
-> glv.init(...) -> 构造 FSVAE。它简化了原项目接口，但也意味着同一进程中
构造多套不同配置模型时必须谨慎重置全局状态。
"""


dtype = None
n_steps = None
network_config = None
layer_config = None
devices = None
params = {}

def init(n_config, devs):
    # 项目以全局配置驱动模型构建；训练入口必须先调用本函数再实例化网络。
    global dtype, devices, n_steps, tau_s, network_config, layer_config, params
    dtype = torch.float32
    devices = devs
    network_config = n_config
    # 原实现按设备数缩放 batch size 和 lr，单卡时等于配置文件中的值。
    network_config['batch_size'] = network_config['batch_size'] * len(devices)
    # 注意这里使用的是已经缩放后的 batch_size；当前训练入口实际只使用单卡，
    # 所以不会暴露多卡重复缩放问题。
    network_config['lr'] = network_config['lr'] * len(devices) * network_config['batch_size'] / 250
    layer_config = {'threshold': 0.2}
    n_steps = network_config['n_steps']
    
    
