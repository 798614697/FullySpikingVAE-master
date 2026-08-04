import argparse
import json
import os

import torch
import torch.nn.functional as F
import torchvision
import yaml

import global_v as glv
from asc_utils import (atomic_torch_save, celeba_pos_weight, isolated_rng,
                       restore_rng_state, rng_state, seed_everything)
from attribute_classifier import load_frozen_classifier
from datasets.load_dataset_snn import load_celeba_splits
from fsvae_models.asc_fsvae import ASCFSVAELarge, normalize_attributes, spike_attribute_bce
from fsvae_models.fsvae import FSVAELarge


def build_model(experiment):
    if experiment == 'A':
        return FSVAELarge()
    return ASCFSVAELarge(use_encoder_attribute_head=(experiment == 'D'))


def forward_model(model, experiment, spike_input, condition, scheduled):
    if experiment == 'A':
        recon, q_z, p_z, sampled_z = model(spike_input, scheduled=scheduled)
        return recon, q_z, p_z, sampled_z, None
    return model(spike_input, condition, scheduled=scheduled)


def loss_terms(model, experiment, images, attributes, output, pos_weight, classifier, config):
    recon, q_z, p_z, _, attribute_spikes = output
    base = model.loss_function_mmd(images, recon, q_z, p_z)
    zero = base['loss'].new_zeros(())
    spike_loss = zero
    image_loss = zero
    if experiment == 'D':
        spike_loss, _ = spike_attribute_bce(attribute_spikes, attributes, pos_weight)
    if experiment in ('C', 'D'):
        # Classifier weights are frozen, but autograd through its input is required.
        image_loss = F.binary_cross_entropy_with_logits(
            classifier(recon), normalize_attributes(attributes), pos_weight=pos_weight)
    total = (base['loss'] + config.get('lambda_spike_attribute', 0.1) * spike_loss
             + config.get('lambda_image_attribute', 0.1) * image_loss)
    return {'loss': total, 'reconstruction': base['Reconstruction_Loss'],
            'distance': base['Distance_Loss'], 'spike_attribute': spike_loss,
            'image_attribute': image_loss}


def run_epoch(model, loader, experiment, device, config, pos_weight,
              classifier=None, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, attributes) in enumerate(loader):
            if training:
                optimizer.zero_grad(set_to_none=True)
            batch = images.shape[0]
            micro_batch = int(config.get('micro_batch_size', batch))
            for start in range(0, batch, micro_batch):
                micro_images = images[start:start + micro_batch].to(device, non_blocking=True)
                micro_attributes = attributes[start:start + micro_batch].to(device, non_blocking=True)
                spike_input = micro_images.unsqueeze(-1).expand(
                    -1, -1, -1, -1, config['n_steps'])
                output = forward_model(model, experiment, spike_input, micro_attributes,
                                       config['scheduled'] if training else False)
                losses = loss_terms(model, experiment, micro_images, micro_attributes,
                                    output, pos_weight, classifier, config)
                micro_count = micro_images.shape[0]
                if training:
                    (losses['loss'] * (micro_count / batch)).backward()
                count += micro_count
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + value.detach().item() * micro_count
            if training:
                optimizer.step()
                if (batch_index + 1) % 100 == 0:
                    progress = {name: value / count for name, value in totals.items()}
                    print(json.dumps({'batch': batch_index + 1,
                                      'batches': len(loader), 'train_running': progress}),
                          flush=True)
    return {name: value / count for name, value in totals.items()}


def checkpoint_payload(model, optimizer, scheduler, epoch, best_total, best_recon,
                       config, experiment, seed, attr_names, pos_weight):
    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'epoch': epoch,
            'best_total': best_total, 'best_reconstruction': best_recon,
            'rng_state': rng_state(), 'config': config, 'experiment': experiment,
            'seed': seed, 'attr_names': list(attr_names), 'pos_weight': pos_weight.cpu()}


def save_samples(model, experiment, loader, device, output_dir, epoch, seed, n_steps):
    images, conditions = next(iter(loader))
    conditions = conditions[:16].to(device)
    model.eval()
    with isolated_rng(seed + 100000 + epoch), torch.no_grad():
        if experiment == 'A':
            generated, _ = model.sample(conditions.shape[0])
        else:
            generated, _ = model.sample(conditions)
    path = os.path.join(output_dir, 'samples')
    os.makedirs(path, exist_ok=True)
    torchvision.utils.save_image((generated + 1) / 2,
                                 os.path.join(path, f'epoch_{epoch:03d}.png'), nrow=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, choices=list('ABCD'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--classifier-checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--resume')
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)['Network']
    expected = {'A': (False, False), 'B': (False, False),
                'C': (True, False), 'D': (True, True)}[args.experiment]
    actual = (config.get('use_image_attribute_loss', False),
              config.get('use_encoder_attribute_head', False))
    if actual != expected:
        raise ValueError(f'config flags {actual} do not match experiment {args.experiment}: {expected}')
    seed_everything(args.seed)
    glv.init(config, [0])
    device = torch.device(args.device)
    loaders, attr_names = load_celeba_splits(config['data_path'], config['batch_size'],
                                             args.num_workers, download=False)
    pos_weight = celeba_pos_weight(loaders['train'].dataset,
                                   config.get('pos_weight_cap', 10.0)).to(device)
    model = build_model(args.experiment).to(device)
    classifier = None
    if args.experiment in ('C', 'D'):
        if not args.classifier_checkpoint:
            raise ValueError('C/D require --classifier-checkpoint')
        classifier, classifier_state = load_frozen_classifier(args.classifier_checkpoint, device)
        if list(attr_names) != list(classifier_state['attr_names']):
            raise ValueError('classifier and dataset attribute order differ')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                  betas=(0.9, 0.999), weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.get('lr_milestones', [30]), gamma=0.1)
    start_epoch = 0
    best_total = best_recon = float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        best_total, best_recon = state['best_total'], state['best_reconstruction']
        restore_rng_state(state['rng_state'])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'resolved_config.yaml'), 'w') as handle:
        yaml.safe_dump({'Network': config, 'experiment': args.experiment, 'seed': args.seed}, handle)
    for epoch in range(start_epoch, config['epochs']):
        if config['scheduled']:
            model.update_p(epoch, config['epochs'])
        train_metrics = run_epoch(model, loaders['train'], args.experiment, device,
                                  config, pos_weight, classifier, optimizer)
        validation_interval = int(config.get('validation_interval', 1))
        should_validate = (epoch % validation_interval == 0
                           or epoch == config['epochs'] - 1)
        valid_metrics = None
        if should_validate:
            # Evaluation has an isolated RNG so it cannot perturb training RNG.
            with isolated_rng(args.seed + 50000 + epoch):
                valid_metrics = run_epoch(model, loaders['valid'], args.experiment, device,
                                          config, pos_weight, classifier)
        used_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        print(json.dumps({'epoch': epoch, 'lr': used_lr,
                          'train': train_metrics, 'valid': valid_metrics}), flush=True)
        improved_total = valid_metrics is not None and valid_metrics['loss'] < best_total
        improved_recon = (valid_metrics is not None
                          and valid_metrics['reconstruction'] < best_recon)
        if valid_metrics is not None:
            best_total = min(best_total, valid_metrics['loss'])
            best_recon = min(best_recon, valid_metrics['reconstruction'])
        payload = checkpoint_payload(model, optimizer, scheduler, epoch, best_total,
                                     best_recon, config, args.experiment, args.seed,
                                     attr_names, pos_weight)
        atomic_torch_save(payload, os.path.join(args.output_dir, 'latest_resume.pt'))
        if improved_total:
            atomic_torch_save(payload, os.path.join(args.output_dir, 'best_total.pt'))
        if improved_recon:
            atomic_torch_save(payload, os.path.join(args.output_dir, 'best_reconstruction.pt'))
        if epoch % config.get('checkpoint_interval', 5) == 0:
            atomic_torch_save(payload, os.path.join(args.output_dir,
                                                    f'epoch_{epoch + 1:03d}.pt'))
        if epoch % config.get('sample_interval', 5) == 0:
            save_samples(model, args.experiment, loaders['valid'], device,
                         args.output_dir, epoch + 1, args.seed, config['n_steps'])


if __name__ == '__main__':
    main()
