"""Evaluate an FSCVAE checkpoint and create paired condition-control grids."""

import argparse
import json
import os
import random

import numpy as np
import torch
import torchvision

import global_v as glv
from datasets import load_dataset_snn
import fsvae_models.fscvae as fscvae
import metrics.clean_fid as clean_fid
import metrics.inception_score as inception_score
from network_parser import parse


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-gen", type=int, default=5000)
    parser.add_argument("--is-batch-size", type=int, default=250)
    parser.add_argument("--is-batches", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=["Smiling", "Eyeglasses", "Male", "Blond_Hair"],
    )
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-is", action="store_true")
    parser.add_argument("--skip-pairs", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RealConditionSampler:
    """Supply real CelebA attribute combinations to existing metric helpers."""

    def __init__(self, sample_function, conditions, device):
        self.sample_function = sample_function
        self.conditions = conditions
        self.device = device
        self.offset = 0

    def __call__(self, batch_size):
        indices = (torch.arange(batch_size) + self.offset) % len(self.conditions)
        condition = self.conditions[indices].to(self.device)
        self.offset = (self.offset + batch_size) % len(self.conditions)
        return self.sample_function(batch_size, condition=condition)


def load_model(config, checkpoint, device):
    params = parse(config)["Network"]
    glv.init(params, [device.index or 0])
    model_cls = fscvae.FSCVAELarge if params["model"] in ("FSCVAE_large", "FSVAE_large") else fscvae.FSCVAE
    model = model_cls().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return params, model


def toggle_attribute(condition, index, name, attr_index):
    changed = condition.clone()
    changed[:, index] = 1.0 - changed[:, index]
    # Hair colours are mutually exclusive in a controlled edit.
    if name.endswith("_Hair") and changed[:, index].item() == 1:
        for other in ("Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"):
            if other in attr_index and other != name:
                changed[:, attr_index[other]] = 0
    return changed


@torch.no_grad()
def paired_controls(model, conditions, attr_names, args, device):
    attr_index = {name: i for i, name in enumerate(attr_names)}
    missing = [name for name in args.attributes if name not in attr_index]
    if missing:
        raise ValueError(f"unknown CelebA attributes: {missing}")

    results = {}
    for attr_number, name in enumerate(args.attributes):
        rows = []
        fixed_z_l1 = []
        paired_prior_l1 = []
        index = attr_index[name]
        for pair_number in range(args.pairs):
            base = conditions[pair_number:pair_number + 1].to(device).float()
            changed = toggle_attribute(base, index, name, attr_index)

            # Fixed-z comparison isolates the condition path through the decoder.
            set_seed(10000 + pair_number)
            base_sequence = model.condition_encoder(base)
            latent = model.prior.sample(1, base_sequence)
            base_fixed = model.decode(latent, base_sequence)
            changed_sequence = model.condition_encoder(changed)
            changed_fixed = model.decode(latent, changed_sequence)

            # Resetting the seed pairs the random draws while allowing p(z|y) to change.
            set_seed(20000 + pair_number)
            base_prior, _ = model.sample(1, condition=base)
            set_seed(20000 + pair_number)
            changed_prior, _ = model.sample(1, condition=changed)

            fixed_z_l1.append((base_fixed - changed_fixed).abs().mean().item())
            paired_prior_l1.append((base_prior - changed_prior).abs().mean().item())
            rows.extend([base_fixed, changed_fixed, base_prior, changed_prior])

        grid = torch.cat(rows, dim=0)
        path = os.path.join(args.output_dir, f"paired_{name}.png")
        torchvision.utils.save_image((grid + 1) / 2, path, nrow=4)
        results[name] = {
            "columns": ["fixed_z_base", "fixed_z_toggled", "prior_base", "prior_toggled"],
            "fixed_z_mean_l1": float(np.mean(fixed_z_l1)),
            "paired_prior_mean_l1": float(np.mean(paired_prior_l1)),
            "image": path,
        }
    return results


def main():
    args = arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    set_seed(1234)
    params, model = load_model(args.config, args.checkpoint, device)
    _, test_loader = load_dataset_snn.load_celebA(os.path.expanduser(params["data_path"]))
    dataset = test_loader.dataset
    conditions = (dataset.attr > 0).float()
    attr_names = list(dataset.attr_names)
    report = {"checkpoint": args.checkpoint, "num_gen": args.num_gen}

    if not args.skip_metrics:
        original_sample = model.sample
        sampler = RealConditionSampler(original_sample, conditions, device)
        model.sample = sampler
        set_seed(1234)
        if not args.skip_fid:
            report["fid"] = clean_fid.get_clean_fid_score(
                model, params["dataset"], device, args.num_gen
            )
        if not args.skip_is:
            sampler.offset = 0
            set_seed(1234)
            mean, std = inception_score.get_inception_score(
                model, device, args.is_batch_size, args.is_batches
            )
            report["inception_score_mean"] = float(mean)
            report["inception_score_std"] = float(std)
        model.sample = original_sample

    if not args.skip_pairs:
        report["paired_controls"] = paired_controls(
            model, conditions, attr_names, args, device
        )

    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
