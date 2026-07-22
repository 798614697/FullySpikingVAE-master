from cleanfid import fid

"""用 clean-fid 比较生成图分布和项目预先注册的真实数据统计。

FS-CVAE 调用 model.sample 时若未显式给 y，会走随机属性兼容路径；严格的
条件生成评价应另外按真实属性分布采样，并增加属性一致率。
"""

def get_clean_fid_score(model, dataset, device, num_gen=5000):
    """适配 SNN 模型的 clean-fid 入口；返回值越低通常表示分布越接近。"""
    if dataset.lower() == "mnist":
        dataset_name = 'mnist_test'
    elif dataset.lower() == "fashionmnist":
        dataset_name = 'fashion_test'
    elif dataset.lower() == 'celeba':
        dataset_name = 'celeba_valid'
    elif dataset.lower() == 'cifar10':
        dataset_name = 'cifar10_test'
    else:
        raise ValueError()

    # clean-fid 要求一个接收 dummy latent 的生成器；FSVAE 自己从 prior 采样，
    # 因此这里只使用 z.shape[0] 获取所需 batch size。
    def sample_from_vae(z):
        """
        z : dummy latent value (batch_size, z_dim)
        """
        batch_size = z.shape[0]
        sampled_x, _ = model.sample(batch_size)
        sampled_x = (sampled_x+1)/2 # 0 to 1
        sampled_x = 255 * sampled_x # 0 to 255
        if sampled_x.shape[1] == 1:
            sampled_x = sampled_x.repeat(1,3,1,1) # gray to RGB
        return sampled_x

    score = fid.compute_fid(gen=sample_from_vae, dataset_name=dataset_name,
            num_gen=num_gen, dataset_split="custom", batch_size=256, device=device, z_dim=2)

    return score

def get_clean_fid_score_ann(model, dataset, device, num_gen=5000):
    if dataset.lower() == "mnist":
        dataset_name = 'mnist_test'
    elif dataset.lower() == "fashionmnist":
        dataset_name = 'fashion_test'
    elif dataset.lower() == 'celeba':
        dataset_name = 'celeba_valid'
    elif dataset.lower() == 'cifar10':
        dataset_name = 'cifar10_test'
    else:
        raise ValueError()

    # function that accepts a latent and returns an image in range[0,255]
    def sample_from_vae(z):
        """
        z : dummy latent value (batch_size, z_dim)
        """
        batch_size = z.shape[0]
        sampled_x= model.sample(batch_size, device)
        sampled_x = (sampled_x+1)/2 # 0 to 1
        sampled_x = 255 * sampled_x # 0 to 255
        if sampled_x.shape[1] == 1:
            sampled_x = sampled_x.repeat(1,3,1,1) # gray to RGB
        return sampled_x

    score = fid.compute_fid(gen=sample_from_vae, dataset_name=dataset_name,
            num_gen=num_gen, dataset_split="custom", batch_size=256, device=device, z_dim=2)

    return score
