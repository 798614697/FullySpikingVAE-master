"""Attribute-Supervised Conditional Fully Spiking VAE."""
import torch
import torch.nn as nn

import global_v as glv
from .fsvae import FSVAELarge
from .snn_layers import LIFSpike, tdBatchNorm, tdLinear


def normalize_attributes(condition):
    """Normalize CelebA {-1, 1} (or already {0, 1}) attributes to float {0, 1}."""
    condition = condition.float()
    if condition.numel() and condition.min() < 0:
        condition = (condition + 1.0) / 2.0
    return condition.clamp(0.0, 1.0)


class ConditionEncoder(nn.Module):
    def __init__(self, condition_dim=40, embedding_dim=64):
        super().__init__()
        self.condition_dim = condition_dim
        self.embedding_dim = embedding_dim
        self.layer = tdLinear(condition_dim, embedding_dim, bias=True,
                              bn=None, spike=LIFSpike())

    def forward(self, condition, n_steps):
        if condition is None:
            raise ValueError('a complete condition tensor is required for ASC-FSVAE sampling')
        condition = normalize_attributes(condition)
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                f'condition must have shape (B,{self.condition_dim}), got {tuple(condition.shape)}')
        return self.layer(condition.unsqueeze(-1).expand(-1, -1, n_steps))


class ConditionalPriorBernoulliSTBP(nn.Module):
    """p(z_t | z_<t, e_t), with condition present at every time step."""
    def __init__(self, k=20, condition_dim=64):
        super().__init__()
        self.channels = glv.network_config['latent_dim']
        self.k = k
        self.n_steps = glv.network_config['n_steps']
        self.layers = nn.Sequential(
            tdLinear(self.channels + condition_dim, self.channels * 2, bias=True,
                     bn=tdBatchNorm(self.channels * 2, alpha=2), spike=LIFSpike()),
            tdLinear(self.channels * 2, self.channels * 4, bias=True,
                     bn=tdBatchNorm(self.channels * 4, alpha=2), spike=LIFSpike()),
            tdLinear(self.channels * 4, self.channels * k, bias=True,
                     bn=tdBatchNorm(self.channels * k, alpha=2), spike=LIFSpike()))
        self.register_buffer('initial_input', torch.zeros(1, self.channels, 1))

    def _distribution(self, history, condition_embedding):
        return self.layers(torch.cat([history, condition_embedding], dim=1))

    def forward(self, z, condition_embedding, scheduled=False, p=0.0):
        batch_size = z.shape[0]
        z = z.detach()
        history = self.initial_input.repeat(batch_size, 1, 1)
        if scheduled and self.training:
            # Decisions are shared across the batch, matching the original FSVAE schedule.
            import random
            with torch.no_grad():
                for t in range(self.n_steps - 1):
                    if t >= 5 and random.random() < p:
                        out = self._distribution(history, condition_embedding[..., :t + 1])[..., -1]
                        prob = out.view(batch_size, self.channels, self.k).mean(-1)
                        z_t = (prob + 1e-3 * torch.randn_like(prob) > 0.5).float()
                    else:
                        z_t = z[..., t]
                    history = torch.cat([history, z_t.unsqueeze(-1)], dim=-1)
        else:
            history = torch.cat([history, z[..., :-1]], dim=-1)
        out = self._distribution(history.detach(), condition_embedding)
        return out.view(batch_size, self.channels, self.k, self.n_steps)

    def sample(self, condition_embedding):
        batch_size = condition_embedding.shape[0]
        history = self.initial_input.repeat(batch_size, 1, 1)
        for t in range(self.n_steps):
            out = self._distribution(history, condition_embedding[..., :t + 1])[..., -1]
            indices = (torch.randint(self.k, (batch_size * self.channels,), device=out.device)
                       + torch.arange(0, batch_size * self.channels * self.k,
                                      self.k, device=out.device))
            z_t = out.reshape(-1)[indices].view(batch_size, self.channels, 1)
            history = torch.cat([history, z_t], dim=-1)
        return history[..., 1:]


class ConditionalPosteriorBernoulliSTBP(nn.Module):
    """q(z_t | x_<=t, z_<t, e_t)."""
    def __init__(self, k=20, condition_dim=64):
        super().__init__()
        self.channels = glv.network_config['latent_dim']
        self.k = k
        self.n_steps = glv.network_config['n_steps']
        self.layers = nn.Sequential(
            tdLinear(self.channels * 2 + condition_dim, self.channels * 2, bias=True,
                     bn=tdBatchNorm(self.channels * 2, alpha=2), spike=LIFSpike()),
            tdLinear(self.channels * 2, self.channels * 4, bias=True,
                     bn=tdBatchNorm(self.channels * 4, alpha=2), spike=LIFSpike()),
            tdLinear(self.channels * 4, self.channels * k, bias=True,
                     bn=tdBatchNorm(self.channels * k, alpha=2), spike=LIFSpike()))
        self.register_buffer('initial_input', torch.zeros(1, self.channels, 1))

    def forward(self, x, condition_embedding):
        batch_size = x.shape[0]
        indices_per_step = []
        with torch.no_grad():
            history = self.initial_input.repeat(batch_size, 1, 1)
            for t in range(self.n_steps - 1):
                inputs = torch.cat([x[..., :t + 1].detach(), history,
                                    condition_embedding[..., :t + 1].detach()], dim=1)
                out = self.layers(inputs)[..., -1]
                indices = (torch.randint(self.k, (batch_size * self.channels,), device=x.device)
                           + torch.arange(0, batch_size * self.channels * self.k,
                                          self.k, device=x.device))
                indices_per_step.append(indices)
                z_t = out.reshape(-1)[indices].view(batch_size, self.channels, 1)
                history = torch.cat([history, z_t], dim=-1)

        q_flat = self.layers(torch.cat([x, history.detach(), condition_embedding], dim=1))
        sampled = []
        for t in range(self.n_steps):
            if t == self.n_steps - 1:
                indices = (torch.randint(self.k, (batch_size * self.channels,), device=x.device)
                           + torch.arange(0, batch_size * self.channels * self.k,
                                          self.k, device=x.device))
            else:
                indices = indices_per_step[t]
            sampled.append(q_flat[..., t].reshape(-1)[indices].view(batch_size, self.channels, 1))
        q_z = q_flat.view(batch_size, self.channels, self.k, self.n_steps)
        return torch.cat(sampled, dim=-1), q_z


class SpikeAttributeHead(nn.Module):
    def __init__(self, latent_dim=128, attribute_dim=40, hidden_dim=256, population=10):
        super().__init__()
        self.attribute_dim = attribute_dim
        self.population = population
        self.layers = nn.Sequential(
            tdLinear(latent_dim, hidden_dim, bias=True,
                     bn=tdBatchNorm(hidden_dim), spike=LIFSpike()),
            tdLinear(hidden_dim, attribute_dim * population, bias=True,
                     bn=tdBatchNorm(attribute_dim * population), spike=LIFSpike()))

    def forward(self, latent_x):
        spikes = self.layers(latent_x)
        return spikes.view(spikes.shape[0], self.attribute_dim,
                           self.population, spikes.shape[-1])


class ASCFSVAELarge(FSVAELarge):
    """Conditional q/p; the original decoder remains D(z)."""
    def __init__(self, use_encoder_attribute_head=False):
        super().__init__()
        condition_dim = int(glv.network_config.get('condition_dim', 40))
        embedding_dim = int(glv.network_config.get('condition_embedding_dim', 64))
        self.condition_dim = condition_dim
        self.condition_encoder = ConditionEncoder(condition_dim, embedding_dim)
        self.prior = ConditionalPriorBernoulliSTBP(self.k, embedding_dim)
        self.posterior = ConditionalPosteriorBernoulliSTBP(self.k, embedding_dim)
        self.attribute_head = None
        if use_encoder_attribute_head:
            self.attribute_head = SpikeAttributeHead(
                self.latent_dim, condition_dim,
                int(glv.network_config.get('attribute_hidden_dim', 256)),
                int(glv.network_config.get('condition_population', 10)))

    def encode(self, x, condition, scheduled=False):
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1, end_dim=3)
        latent_x = self.before_latent_layer(features)
        embedding = self.condition_encoder(condition, self.n_steps)
        sampled_z, q_z = self.posterior(latent_x, embedding)
        p_z = self.prior(sampled_z, embedding, scheduled, self.p)
        attribute_spikes = None if self.attribute_head is None else self.attribute_head(latent_x)
        return sampled_z, q_z, p_z, attribute_spikes

    def forward(self, x, condition, scheduled=False):
        sampled_z, q_z, p_z, attribute_spikes = self.encode(x, condition, scheduled)
        return self.decode(sampled_z), q_z, p_z, sampled_z, attribute_spikes

    def sample(self, condition):
        embedding = self.condition_encoder(condition, self.n_steps)
        sampled_z = self.prior.sample(embedding)
        return self.decode(sampled_z), sampled_z


def spike_attribute_bce(spikes, targets, pos_weight, eps=1e-5):
    targets = normalize_attributes(targets)
    probability = spikes.mean(dim=(2, 3))
    # Add epsilon inside log instead of clamping: an all-silent population must
    # still receive surrogate-gradient signal from positive attributes.
    loss = -(pos_weight * targets * torch.log(probability + eps)
             + (1.0 - targets) * torch.log(1.0 - probability + eps)).mean()
    return loss, probability
