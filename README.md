# Fully Spiking Variational Autoencoder
official implementation of Fully Spiking Variational Autoencoder

Accepted to **AAAI2022**!!

paper: https://ojs.aaai.org/index.php/AAAI/article/view/20665/20424

arxiv: https://arxiv.org/abs/2110.00375

![overview](./imgs/overview.png?raw=true)

# Get started

1. install dependencies

```
pip install -r requirements.txt
```

2. initialize the fid stats

```
python init_fid_stats.py
```

# Demo
The following command calculates the Inception score & FID of FSVAE trained on CelebA. After that, it outputs `demo_input.png`, `demo_recons.png`, and `demo_sample.png`.
```
python demo.py
```
If CelebA cannot be downloaded, you can still generate samples from the demo checkpoint without loading the dataset:
```
python demo.py --sample-only --skip-metrics
```

# Training Fully Spiking VAE
```
python main_fsvae exp_name -config NetworkConfigs/dataset_name.yaml
```

Training settings are defined in `NetworkConfigs/*.yaml`.

args:
- name: [required] experiment name
- config: [required] config file path
- checkpoint: checkpoint path (if use pretrained model) 
- device: device id of gpu, default 0

You can watch the logs with below command and access http://localhost:8009/ 

```
tensorboard --logdir checkpoint --bind_all --port 8009
```

# Training ANN VAE
As a comparison method, we prepared vanilla VAEs of the same network architecture built with ANN, and trained on the same settings.

```
python main_ann_vae exp_name -dataset dataset_name
```

args: 
- name: [required] experiment name
- dataset:[required] dataset name [mnist, fashion, celeba, cifar10]
- batch_size: default 250
- latent_dim: default 128
- checkpoint: checkpoint path (if use pretrained model) 
- device: device id of gpu, default 0

# Evaluation
![results](imgs/results.png)

## ASC-FSVAE conditional experiments

The `experiment/fsvae-innovation` branch adds four controlled CelebA runs:

- A: original FSVAE baseline.
- B: true attributes condition both posterior and prior.
- C: B plus a frozen image-attribute guidance loss.
- D: C plus an auxiliary spiking attribute head on the image encoder.

All configurations use an effective batch size of 64 (micro-batch 8), 100 epochs,
seed 2024, and learning rates 0.001 for epochs 0--29 then 0.0001.  Pretrain the
shared classifier and run the four experiments with `scripts/run_remote_screening.sh`.
Conditional sampling requires a complete CelebA 40-attribute vector; it never
falls back to a random condition. `evaluate_asc.py` computes reconstruction MSE
and writes fixed-seed, single-attribute counterfactual grids for the six core
attributes.

# Reconstructed Images
![mnist_recons](imgs/mnist_recons_appendix.png)
![fashion_recons](imgs/fashion_recons_appendix.png)
![cifar_recons](imgs/cifar_recons_appendix.png)
![celeb_recons](imgs/celeb_recons_appendix.png)

# Generated Images
![mnist](imgs/mnist_generated_images_appendix.png)
![fashion](imgs/fashion_generated_images_appendix.png)
![cifar](imgs/cifar_generated_images_appendix.png)
![celeb](imgs/celeb_generated_images_appendix.png)
