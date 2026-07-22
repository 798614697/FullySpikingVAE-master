import os
import os.path
import numpy as np
import logging
import argparse

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

import global_v as glv
from network_parser import parse
from datasets import load_dataset_snn
from utils import aboutCudaDevices
from utils import AverageMeter
from utils import CountMulAddSNN
import fsvae_models.fsvae as fsvae
from fsvae_models.snn_layers import LIFSpike
import metrics.inception_score as inception_score
import metrics.clean_fid as clean_fid
import metrics.autoencoder_fid as autoencoder_fid

"""FSVAE/FS-CVAE 实验入口。

本文件不定义网络层，负责把配置、数据、模型、损失、优化器、评估和保存
串成完整训练流程。核心模型结构在 fsvae_models/fsvae.py。
"""


max_accuracy = 0
min_loss = 1000


def config_value(name, default):
    """读取可选 YAML 字段，旧配置缺少该字段时使用 default。"""
    return glv.network_config.get(name, default)


def should_run(epoch, interval, run_last=True):
    """统一控制测试/采样/checkpoint/FID 的执行间隔；interval<=0 表示关闭。"""
    interval = int(interval)
    if interval <= 0:
        return False
    if run_last and epoch == glv.network_config['epochs'] - 1:
        return True
    return epoch % interval == 0


def add_hook(net):
    # 在测试阶段给卷积、线性层和 LIF 激活注册 hook，用于统计 SNN 乘加操作量。
    count_mul_add = CountMulAddSNN()
    hook_handles = []
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.ConvTranspose3d) or isinstance(m, LIFSpike):
            handle = m.register_forward_hook(count_mul_add)
            hook_handles.append(handle)
    return count_mul_add, hook_handles



def write_weight_hist(net, index):
    # 将所有可训练参数写入 TensorBoard，便于观察权重分布是否发散或饱和。
    for n, m in net.named_parameters():
        root, name = os.path.splitext(n)
        writer.add_histogram(root + '/' + name, m, index)

def train(network, trainloader, opti, epoch):
    # FSVAE 的训练循环：静态图像会被复制到 T 个时间步，形成 (N,C,H,W,T)。
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']
    
    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    dist_meter = AverageMeter()
    condition_meter = AverageMeter()
    temporal_meter = AverageMeter()

    mean_q_z = 0
    mean_p_z = 0
    mean_sampled_z = 0

    network = network.train()
    
    for batch_idx, (real_img, labels) in enumerate(trainloader):   
        opti.zero_grad()
        real_img = real_img.to(init_device, non_blocking=True)
        labels = labels.to(init_device, non_blocking=True)
        # direct spike input：不是随机脉冲编码，而是把同一张图像铺到所有时间步。
        spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)
        # MNIST/CIFAR 的 labels 在无条件模型中会被忽略；CelebA FS-CVAE 中
        # labels 是 (B,40) 多属性向量，会送入 TemporalConditionEncoder。
        condition = labels if network.condition_dim else None
        x_recon, q_z, p_z, sampled_z = network(spike_input, condition=condition, scheduled=network_config['scheduled'])
        
        if network_config['loss_func'] == 'mmd':
            losses = network.loss_function_mmd(real_img, x_recon, q_z, p_z)
        elif network_config['loss_func'] == 'kld':
            losses = network.loss_function_kld(real_img, x_recon, q_z, p_z)
        else:
            raise ValueError('unrecognized loss function')
        
        # 一次反传同时更新 SNN 主干、条件编码器、四组门控和辅助分类器。
        losses['loss'].backward()
        
        opti.step()

        loss_meter.update(losses['loss'].detach().cpu().item())
        recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
        dist_meter.update(losses['Distance_Loss'].detach().cpu().item())
        condition_meter.update(losses.get('Condition_Loss', torch.tensor(0.)).detach().cpu().item())
        temporal_meter.update(losses.get('Temporal_Loss', torch.tensor(0.)).detach().cpu().item())

        mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx+1) # (C,k,T)
        mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx+1) # (C,k,T)
        mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (batch_idx+1) # (C,T)

        print(f'Train[{epoch}/{max_epoch}] [{batch_idx}/{len(trainloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')

        if batch_idx == len(trainloader)-1:
            os.makedirs(f'checkpoint/{args.name}/imgs/train/', exist_ok=True)
            torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_input.png')
            torchvision.utils.save_image((x_recon+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_recons.png')
            writer.add_images('Train/input_img', (real_img+1)/2, epoch)
            writer.add_images('Train/recons_img', (x_recon+1)/2, epoch)

    logging.info(f"Train [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} DISTANCE: {dist_meter.avg}")
    writer.add_scalar('Train/loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/recons_loss', recons_meter.avg, epoch)
    writer.add_scalar('Train/distance', dist_meter.avg, epoch)
    writer.add_scalar('Train/condition_consistency', condition_meter.avg, epoch)
    writer.add_scalar('Train/temporal_alignment', temporal_meter.avg, epoch)
    writer.add_scalar('Train/latent_firing_rate', mean_sampled_z.mean().item(), epoch)
    writer.add_scalar('Train/mean_q', mean_q_z.mean().item(), epoch)
    writer.add_scalar('Train/mean_p', mean_p_z.mean().item(), epoch)
    

    writer.add_image('Train/mean_sampled_z', mean_sampled_z.unsqueeze(0), epoch)
    mean_q_z = mean_q_z.permute(1,0,2) # (k,C,T)
    mean_p_z = mean_p_z.permute(1,0,2) # (k,C,T)
    writer.add_image(f'Train/mean_q_z', mean_q_z.mean(0).unsqueeze(0))
    writer.add_image(f'Train/mean_p_z', mean_p_z.mean(0).unsqueeze(0))
    # 每条曲线有 T 个点，可检查门值是否随时间变化或退化为常数。
    for gate_name, gate_by_time in network.condition_gate_means().items():
        for time_index, gate_value in enumerate(gate_by_time):
            writer.add_scalar(f'Train/condition_gate_{gate_name}/t{time_index}',
                              gate_value.item(), epoch)

    return loss_meter.avg


def test(network, testloader, epoch):
    # 测试逻辑与训练基本一致，但不反传，并额外统计计算量。
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']

    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    dist_meter = AverageMeter()
    condition_meter = AverageMeter()
    temporal_meter = AverageMeter()

    mean_q_z = 0
    mean_p_z = 0
    mean_sampled_z = 0

    # hook 统计常规 Linear/Conv/LIF 操作；门控中的逐元素 sigmoid/乘加尚未
    # 完整计入，因此若做严格能耗论文实验还需要扩展 CountMulAddSNN。
    count_mul_add, hook_handles = add_hook(net)

    network = network.eval()
    with torch.no_grad():
        for batch_idx, (real_img, labels) in enumerate(testloader):   
            real_img = real_img.to(init_device, non_blocking=True)
            labels = labels.to(init_device, non_blocking=True)
            # direct spike input
            spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)

            condition = labels if network.condition_dim else None
            x_recon, q_z, p_z, sampled_z = network(spike_input, condition=condition, scheduled=network_config['scheduled'])

            if network_config['loss_func'] == 'mmd':
                losses = network.loss_function_mmd(real_img, x_recon, q_z, p_z)
            elif network_config['loss_func'] == 'kld':
                losses = network.loss_function_kld(real_img, x_recon, q_z, p_z)
            else:
                raise ValueError('unrecognized loss function')

            mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx+1) # (C,k,T)
            mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx+1) # (C,k,T)
            mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (batch_idx+1) # (C,T)
            
            loss_meter.update(losses['loss'].detach().cpu().item())
            recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
            dist_meter.update(losses['Distance_Loss'].detach().cpu().item())
            condition_meter.update(losses.get('Condition_Loss', torch.tensor(0.)).detach().cpu().item())
            temporal_meter.update(losses.get('Temporal_Loss', torch.tensor(0.)).detach().cpu().item())

            print(f'Test[{epoch}/{max_epoch}] [{batch_idx}/{len(testloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')

            if batch_idx == len(testloader)-1:
                os.makedirs(f'checkpoint/{args.name}/imgs/test/', exist_ok=True)
                torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{args.name}/imgs/test/epoch{epoch}_input.png')
                torchvision.utils.save_image((x_recon+1)/2, f'checkpoint/{args.name}/imgs/test/epoch{epoch}_recons.png')
                writer.add_images('Test/input_img', (real_img+1)/2, epoch)
                writer.add_images('Test/recons_img', (x_recon+1)/2, epoch)
                

    logging.info(f"Test [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} DISTANCE: {dist_meter.avg}")
    writer.add_scalar('Test/loss', loss_meter.avg, epoch)
    writer.add_scalar('Test/recons_loss', recons_meter.avg, epoch)
    writer.add_scalar('Test/distance', dist_meter.avg, epoch)
    writer.add_scalar('Test/condition_consistency', condition_meter.avg, epoch)
    writer.add_scalar('Test/temporal_alignment', temporal_meter.avg, epoch)
    writer.add_scalar('Test/latent_firing_rate', mean_sampled_z.mean().item(), epoch)
    writer.add_scalar('Test/mean_q', mean_q_z.mean().item(), epoch)
    writer.add_scalar('Test/mean_p', mean_p_z.mean().item(), epoch)
    writer.add_scalar('Test/mul', count_mul_add.mul_sum.item() / len(testloader), epoch)
    writer.add_scalar('Test/add', count_mul_add.add_sum.item() / len(testloader), epoch)
    
    for handle in hook_handles:
        handle.remove()

    writer.add_image('Test/mean_sampled_z', mean_sampled_z.unsqueeze(0), epoch)
    mean_q_z = mean_q_z.permute(1,0,2) # # (k,C,T)
    mean_p_z = mean_p_z.permute(1,0,2) # # (k,C,T)
    writer.add_image(f'Test/mean_q_z', mean_q_z.mean(0).unsqueeze(0))
    writer.add_image(f'Test/mean_p_z', mean_p_z.mean(0).unsqueeze(0))

    return loss_meter.avg

def sample(network, epoch, batch_size=128, condition=None):
    # 从先验 p(z_t | z_<t) 自回归采样潜变量，再经解码器生成图像。
    network = network.eval()
    with torch.no_grad():
        # FSCVAE 使用测试集真实属性组合，避免独立随机 40 bit 产生冲突条件。
        if condition is not None:
            condition = condition[:batch_size].to(init_device, non_blocking=True)
            batch_size = condition.shape[0]
        sampled_x, sampled_z = network.sample(batch_size, condition=condition)
        writer.add_images('Sample/sample_img', (sampled_x+1)/2, epoch)
        writer.add_image('Sample/mean_sampled_z', sampled_z.mean(0).unsqueeze(0), epoch)
        os.makedirs(f'checkpoint/{args.name}/imgs/sample/', exist_ok=True)
        torchvision.utils.save_image((sampled_x+1)/2, f'checkpoint/{args.name}/imgs/sample/epoch{epoch}_sample.png')

def calc_inception_score(network, epoch, batch_size=256):
    # 生成模型指标。非整 5 epoch 使用较少 batch 以降低训练期间的评估成本。
    network = network.eval()
    with torch.no_grad():
        batch_times = config_value('inception_batch_times', 4)
        inception_mean, inception_std = inception_score.get_inception_score(network, device=init_device, batch_size=batch_size, batch_times=batch_times)
        writer.add_scalar('Sample/inception_score_mean', inception_mean, epoch)
        writer.add_scalar('Sample/inception_score_std', inception_std, epoch)

def calc_clean_fid(network, epoch):
    # clean_fid 使用预先初始化的统计文件，与真实数据分布做 Frechet 距离比较。
    network = network.eval()
    with torch.no_grad():
        num_gen = config_value('fid_num_gen', 5000)
        fid_score = clean_fid.get_clean_fid_score(network, glv.network_config['dataset'], init_device, num_gen)
        writer.add_scalar('Sample/FID', fid_score, epoch)

def calc_autoencoder_frechet_distance(network, epoch):
    # 使用项目提供的自编码器特征空间计算距离，主要用于灰度/小图数据集的补充评价。
    network = network.eval()
    if glv.network_config['dataset'] == "MNIST":
        dataset = 'mnist'
    elif glv.network_config['dataset'] == "FashionMNIST":
        dataset = 'fashion'
    elif glv.network_config['dataset'] == "CelebA":
        dataset = 'celeba'
    elif glv.network_config['dataset'] == "CIFAR10":
        dataset = 'cifar10'
    else:
        raise ValueError()

    with torch.no_grad():
        num_gen = config_value('autoencoder_fid_num_gen', 5000)
        fid_score = autoencoder_fid.get_autoencoder_frechet_distance(network, dataset, init_device, num_gen)
        writer.add_scalar('Sample/AutoencoderDist', fid_score, epoch)
        


if __name__ == '__main__':
    # 示例：python main_fsvae.py celeba_fscvae \
    #          -config NetworkConfigs/CelebA_CVAE.yaml -device 0
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('-config', action='store', dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-device', type=int)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')

    # 当前训练脚本明确要求 CUDA；demo.py 另外支持 CPU 推理。
    if args.device is None:
        init_device = torch.device("cuda:0")
    else:
        init_device = torch.device(f"cuda:{args.device}")
    
    os.makedirs(f'checkpoint/{args.name}', exist_ok=True)
    writer = SummaryWriter(log_dir=f'checkpoint/{args.name}/tb')
    logging.basicConfig(filename=f'checkpoint/{args.name}/{args.name}.log', level=logging.INFO)
    
    logging.info("start parsing settings")
    
    params = parse(args.config)
    network_config = params['Network']
    
    logging.info("finish parsing settings")
    logging.info(network_config)
    print(network_config)
        
    # Check whether a GPU is available
    if torch.cuda.is_available():
        c_device = aboutCudaDevices()
        print(c_device.info())
        print("selected device: ", args.device)
    else:
        raise Exception("only support gpu")
    
    # glv.init 会把配置写入全局变量；模型层初始化时会读取这些全局配置。
    glv.init(network_config, [args.device])

    dataset_name = glv.network_config['dataset']
    data_path = glv.network_config['data_path']
    
    logging.info("dataset loading...")
    if dataset_name == "MNIST":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_mnist(data_path)
    elif dataset_name == "FashionMNIST":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_fashionmnist(data_path)
        
    elif dataset_name == "CIFAR10":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_cifar10(data_path)
        
    elif dataset_name == "CelebA":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_celebA(data_path)
        
    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")

    # 为可视化采样缓存一个测试 batch 的真实相关属性组合。这里只缓存标签，
    # 不会把测试图片送入生成路径。
    sample_conditions = None
    if network_config.get('conditional', False):
        _, sample_conditions = next(iter(test_loader))

    if network_config['model'] == 'FSVAE':
        net = fsvae.FSVAE()
    elif network_config['model'] == 'FSVAE_large':
        net = fsvae.FSVAELarge()
    else:
        raise Exception('not defined model')

    net = net.to(init_device)
    
    # checkpoint 只保存 state_dict，不含 optimizer/epoch，因此这里适合加载
    # 同结构模型权重，但不能完整恢复 AdamW 动量和训练轮数。
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)    
    optimizer = torch.optim.AdamW(net.parameters(), 
                                lr=glv.network_config['lr'], 
                                betas=(0.9, 0.999), 
                                weight_decay=0.001)
    
    best_loss = 1e8
    # 每个 epoch 的顺序：可选权重直方图 -> scheduled p -> train -> test
    # -> 保存 -> 条件采样 -> 可选昂贵生成指标。
    for e in range(glv.network_config['epochs']):
        
        if should_run(e, config_value('weight_hist_interval', 1)):
            write_weight_hist(net, e)
        if network_config['scheduled']:
            # scheduled sampling 概率随 epoch 线性增大，让 prior 逐渐适应自己的历史输出。
            net.update_p(e, glv.network_config['epochs'])
            logging.info("update p")
        train_loss = train(net, train_loader, optimizer, e)

        if should_run(e, config_value('test_interval', 1)):
            test_loss = test(net, test_loader, e)
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(net.state_dict(), f'checkpoint/{args.name}/best.pth')

        if should_run(e, config_value('checkpoint_interval', 1)):
            torch.save(net.state_dict(), f'checkpoint/{args.name}/checkpoint.pth')

        if should_run(e, config_value('sample_interval', 1)):
            sample(net, e, batch_size=config_value('sample_batch_size', 128),
                   condition=sample_conditions)

        if should_run(e, config_value('metric_interval', 1)):
            if config_value('enable_inception_score', True):
                calc_inception_score(net, e, batch_size=config_value('inception_batch_size', 256))
            if config_value('enable_autoencoder_fid', True):
                calc_autoencoder_frechet_distance(net, e)
            if config_value('enable_clean_fid', True):
                calc_clean_fid(net, e)
        
    writer.close()
