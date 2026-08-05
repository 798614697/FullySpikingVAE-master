import argparse
import json
import os

import torch
import torch.nn.functional as F
import yaml

import global_v as glv
from asc_utils import (atomic_torch_save, celeba_pos_weight,
                       load_trusted_checkpoint, seed_everything)
from attribute_classifier import AttributeClassifier
from datasets.load_dataset_snn import load_celeba_splits


def evaluate(model, loader, device, pos_weight):
    model.eval()
    total_loss = total_count = 0
    true_positive = false_negative = true_negative = false_positive = None
    with torch.no_grad():
        for images, attributes in loader:
            images = images.to(device, non_blocking=True)
            targets = (attributes.to(device, non_blocking=True) > 0).float()
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            predictions = logits >= 0
            target_bool = targets.bool()
            batch_tp = (predictions & target_bool).sum(0)
            batch_fn = ((~predictions) & target_bool).sum(0)
            batch_tn = ((~predictions) & (~target_bool)).sum(0)
            batch_fp = (predictions & (~target_bool)).sum(0)
            if true_positive is None:
                true_positive, false_negative = batch_tp, batch_fn
                true_negative, false_positive = batch_tn, batch_fp
            else:
                true_positive += batch_tp
                false_negative += batch_fn
                true_negative += batch_tn
                false_positive += batch_fp
            total_loss += loss.item() * images.shape[0]
            total_count += images.shape[0]
    positive_recall = true_positive.float() / (true_positive + false_negative).clamp_min(1)
    negative_recall = true_negative.float() / (true_negative + false_positive).clamp_min(1)
    return {'loss': total_loss / total_count,
            'macro_positive_recall': positive_recall.mean().item(),
            'macro_balanced_accuracy': ((positive_recall + negative_recall) / 2).mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='NetworkConfigs/ASC_CelebA_D.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--resume')
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)['Network']
    seed_everything(args.seed)
    glv.init(config, [0])
    device = torch.device(args.device)
    loaders, attr_names = load_celeba_splits(config['data_path'], config['batch_size'],
                                             args.num_workers, download=False)
    pos_weight = celeba_pos_weight(loaders['train'].dataset,
                                   config.get('pos_weight_cap', 10.0)).to(device)
    model = AttributeClassifier(len(attr_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best = float('inf')
    start_epoch = 0
    if args.resume:
        state = load_trusted_checkpoint(args.resume, device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch'] + 1
        best = state.get('best_valid_loss', state['metrics']['loss'])
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = count = 0
        for batch_index, (images, attributes) in enumerate(loaders['train']):
            images = images.to(device, non_blocking=True)
            targets = (attributes.to(device, non_blocking=True) > 0).float()
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(images), targets,
                                                       pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.shape[0]
            count += images.shape[0]
            if (batch_index + 1) % 200 == 0:
                print(json.dumps({'epoch': epoch, 'batch': batch_index + 1,
                                  'batches': len(loaders['train']),
                                  'train_running_loss': running / count}), flush=True)
        metrics = evaluate(model, loaders['valid'], device, pos_weight)
        print(json.dumps({'epoch': epoch, 'train_loss': running / count, **metrics}), flush=True)
        improved = metrics['loss'] < best
        best = min(best, metrics['loss'])
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'epoch': epoch, 'attr_names': list(attr_names),
                   'pos_weight': pos_weight.cpu(), 'metrics': metrics, 'config': config,
                   'best_valid_loss': best}
        atomic_torch_save(payload, os.path.join(args.output_dir, 'latest_resume.pt'))
        if improved:
            atomic_torch_save(payload, os.path.join(args.output_dir, 'best_valid.pt'))
    with open(os.path.join(args.output_dir, 'complete.json'), 'w') as handle:
        json.dump({'epochs': args.epochs, 'best_valid_loss': best}, handle)


if __name__ == '__main__':
    main()
