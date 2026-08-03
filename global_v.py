import torch


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
    # Batch size can still be interpreted per device.  Learning-rate scaling is
    # opt-in so an experiment requesting 0.001 actually trains at 0.001.
    network_config['batch_size'] = network_config['batch_size'] * len(devices)
    if network_config.get('scale_lr_by_batch', True):
        network_config['lr'] = network_config['lr'] * len(devices) * network_config['batch_size'] / 250
    layer_config = {'threshold': 0.2}
    n_steps = network_config['n_steps']
    
    
