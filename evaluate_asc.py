"""Deterministic reconstruction and one-attribute counterfactual evaluation."""
import argparse
import json
import os

import numpy as np
import torch
import torchvision
import yaml

import global_v as glv
from asc_utils import isolated_rng, seed_everything
from datasets.load_dataset_snn import load_celeba_splits
from fsvae_models.asc_fsvae import ASCFSVAELarge, normalize_attributes
from fsvae_models.fsvae import FSVAELarge


CORE_ATTRIBUTES = ('Smiling', 'Male', 'Blond_Hair', 'Bangs', 'Eyeglasses', 'Wearing_Hat')
HAIR_ATTRIBUTES = ('Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair')


def load_model(checkpoint_path, experiment, device):
    state = torch.load(checkpoint_path, map_location=device)
    model = FSVAELarge() if experiment == 'A' else ASCFSVAELarge(
        use_encoder_attribute_head=(experiment == 'D'))
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state


def reconstruction_mse(model, experiment, loader, device, n_steps, micro_batch_size):
    squared_error = pixels = 0
    with torch.no_grad():
        for images, attributes in loader:
            for start in range(0, images.shape[0], micro_batch_size):
                micro_images = images[start:start + micro_batch_size].to(device)
                micro_attributes = attributes[start:start + micro_batch_size].to(device)
                spike_input = micro_images.unsqueeze(-1).expand(-1, -1, -1, -1, n_steps)
                if experiment == 'A':
                    recon = model(spike_input, scheduled=False)[0]
                else:
                    recon = model(spike_input, micro_attributes, scheduled=False)[0]
                squared_error += (recon - micro_images).square().sum().item()
                pixels += micro_images.numel()
    return squared_error / pixels


def select_base_conditions(dataset, attr_names, seed, per_direction=4):
    rng = np.random.default_rng(seed)
    attributes = normalize_attributes(dataset.attr).cpu().numpy().astype(np.int64)
    selected = {}
    for name in CORE_ATTRIBUTES:
        index = attr_names.index(name)
        selected[name] = []
        for value in (0, 1):
            candidates = np.flatnonzero(attributes[:, index] == value)
            choices = rng.choice(candidates, size=per_direction, replace=False)
            selected[name].extend((int(choice), value) for choice in choices)
    return selected


def toggled_condition(condition, target_name, attr_names):
    result = condition.clone()
    target = attr_names.index(target_name)
    result[target] = 1.0 - result[target]
    if target_name in HAIR_ATTRIBUTES and result[target] == 1:
        for name in HAIR_ATTRIBUTES:
            if name != target_name:
                result[attr_names.index(name)] = 0
    return result


def save_counterfactual_pairs(model, dataset, attr_names, device, output_dir,
                              seed=2024, sampling_seeds=(2024, 2025, 2026)):
    selections = select_base_conditions(dataset, attr_names, seed)
    manifest = []
    os.makedirs(output_dir, exist_ok=True)
    for name, bases in selections.items():
        pairs = []
        for dataset_index, original_value in bases:
            condition = normalize_attributes(dataset.attr[dataset_index]).to(device)
            changed = toggled_condition(condition, name, attr_names)
            for sampling_seed in sampling_seeds:
                with isolated_rng(sampling_seed + dataset_index):
                    original_image, _ = model.sample(condition.unsqueeze(0))
                with isolated_rng(sampling_seed + dataset_index):
                    changed_image, _ = model.sample(changed.unsqueeze(0))
                pairs.extend([original_image.cpu(), changed_image.cpu()])
                manifest.append({'attribute': name, 'dataset_index': dataset_index,
                                 'direction': f'{original_value}->{1-original_value}',
                                 'sampling_seed': sampling_seed})
        torchvision.utils.save_image((torch.cat(pairs) + 1) / 2,
                                     os.path.join(output_dir, f'{name}.png'), nrow=6)
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, choices=list('ABCD'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='valid', choices=('valid', 'test'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)['Network']
    seed_everything(args.seed)
    glv.init(config, [0])
    device = torch.device(args.device)
    loaders, attr_names = load_celeba_splits(config['data_path'], config['batch_size'],
                                             args.num_workers, download=False)
    model, state = load_model(args.checkpoint, args.experiment, device)
    mse = reconstruction_mse(model, args.experiment, loaders[args.split], device,
                             config['n_steps'], config.get('micro_batch_size',
                                                          config['batch_size']))
    os.makedirs(args.output_dir, exist_ok=True)
    results = {'experiment': args.experiment, 'split': args.split,
               'reconstruction_mse': mse, 'checkpoint_epoch': state['epoch']}
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as handle:
        json.dump(results, handle, indent=2)
    if args.experiment != 'A':
        save_counterfactual_pairs(model, loaders[args.split].dataset, list(attr_names),
                                  device, os.path.join(args.output_dir, 'paired'), args.seed)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
