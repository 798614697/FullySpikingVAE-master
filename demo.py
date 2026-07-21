import os
import os.path
import argparse

import torch
import torchvision

import global_v as glv
from datasets import load_dataset_snn
import fsvae_models.fsvae as fsvae
import metrics.inception_score as inception_score
import metrics.clean_fid as clean_fid
import metrics.autoencoder_fid as autoencoder_fid

def test_sample(network, testloader):
    n_steps = glv.network_config['n_steps']

    network = network.eval()
    with torch.no_grad():
        real_img, labels = next(iter(testloader))
        real_img = real_img.to(init_device, non_blocking=True)
        labels = labels.to(init_device, non_blocking=True)
        # direct spike input
        spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)

        x_recon, q_z, p_z, sampled_z = network(spike_input, scheduled=False)

        torchvision.utils.save_image((real_img+1)/2, f'demo_imgs/demo_input.png')
        torchvision.utils.save_image((x_recon+1)/2, f'demo_imgs/demo_recons.png')

def sample(network):
    network = network.eval()
    with torch.no_grad():
        sampled_x, sampled_z = network.sample(glv.network_config['batch_size'])
        torchvision.utils.save_image((sampled_x+1)/2, f'demo_imgs/demo_sample.png')
        

def calc_inception_score(network):
    network = network.eval()
    with torch.no_grad():
        inception_mean, inception_std = inception_score.get_inception_score(network, 
                                                            device=init_device, 
                                                            batch_size=64, 
                                                            batch_times=40)
    return inception_mean

def calc_clean_fid(network):
    network = network.eval()
    with torch.no_grad():
        num_gen=5000
        fid_score = clean_fid.get_clean_fid_score(network, glv.network_config['dataset'], init_device, num_gen)
    return fid_score

def calc_autoencoder_frechet_distance(network):
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
        score = autoencoder_fid.get_autoencoder_frechet_distance(network, dataset, init_device, 5000)
    return score 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data', help='path of CelebA dataset')
    parser.add_argument('--checkpoint', default='./demo_checkpoint/fsvae_celeba_demo.pth')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sample-only', action='store_true',
                        help='skip CelebA loading and only save demo_sample.png')
    parser.add_argument('--skip-metrics', action='store_true',
                        help='skip inception/FID/autoencoder Frechet metrics')
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'cuda':
        return torch.device('cuda:0')
    return torch.device(device_arg)


if __name__ == '__main__':
    args = parse_args()

    init_device = resolve_device(args.device)
    
    network_config = {"batch_size": args.batch_size, "n_steps": 16, "dataset": "CelebA",
                        "in_channels": 3, "latent_dim": 128, "input_size": 64, 
                        "k": 20, "loss_func": "mmd", "lr": 0.001}
    
    glv.init(network_config, devs=[0])

    dataset_name = glv.network_config['dataset']
    data_path = args.data_path # specify the path of dataset
    
    test_loader = None
    if not args.sample_only:
        # load celeba dataset
        data_path = os.path.expanduser(data_path)
        _, test_loader = load_dataset_snn.load_celebA(data_path)
        
    net = fsvae.FSVAELarge()
    net = net.to(init_device)
    
    checkpoint = torch.load(args.checkpoint, map_location=init_device)
    net.load_state_dict(checkpoint)    

    if not args.skip_metrics:
        print("calculating inception score...")
        inception_s = calc_inception_score(net)
        print("calculating fid score...")
        fid_score = calc_clean_fid(net)
        autoencoder_frechet_distance = calc_autoencoder_frechet_distance(net)

    if test_loader is not None:
        test_sample(net, test_loader)

    sample(net)

    print("###############################")
    if not args.skip_metrics:
        print(f"Inception score: {inception_s}")
        print(f'FID score: {fid_score}')
        print(f'Autoencoder Frechet score: {autoencoder_frechet_distance}')
    if test_loader is not None:
        print('save demo_imgs/demo_input.png')
        print('save demo_imgs/demo_recons.png')
    print('save demo_imgs/demo_sample.png')
    print("###############################")
    
