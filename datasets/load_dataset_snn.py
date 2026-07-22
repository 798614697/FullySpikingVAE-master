import os
import torchvision.datasets
import torchvision.transforms as transforms
import torch
import global_v as glv

"""SNN 训练数据加载器。

这里仍返回普通静态图像；把图像复制到 T 个时间步的 direct input 操作发生
在 main_fsvae.py。所有图像映射到 [-1,1]，与 decoder 末端 tanh 对齐。
"""

def load_mnist(data_path):
    print("loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def load_fashionmnist(data_path):
    print("loading Fashion MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    return trainloader, testloader

def load_cifar10(data_path):
    print("loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    return trainloader, testloader

def load_celebA(data_path):
    """返回 CelebA train/test DataLoader。

    torchvision 会读取 data/celeba/list_attr_celeba.txt；target_type='attr'
    明确要求第二个返回值是 40 维属性，而不是 identity/bbox/landmarks。
    当前 torchvision 返回 {0,1}，模型也兼容旧版本的 {-1,1}。
    """
    print("loading CelebA")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    # 训练集可随机翻转以增强泛化；属性（如 Male/Smiling）不随水平翻转改变。
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange])
    # 测试集不使用随机变换，保证 FSVAE/FSCVAE 的重建指标可复现。
    transform_test = transforms.Compose([
        transforms.CenterCrop(148),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.CelebA(root=data_path,
                                            split='train',
                                            target_type='attr',
                                            download=True,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=8, pin_memory=True)

    testset = torchvision.datasets.CelebA(root=data_path,
                                            split='test',
                                            target_type='attr',
                                            download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=8, pin_memory=True)
    return trainloader, testloader



