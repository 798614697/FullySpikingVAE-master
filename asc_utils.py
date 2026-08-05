import os
import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def isolated_rng(seed):
    state = rng_state()
    seed_everything(seed)
    try:
        yield
    finally:
        restore_rng_state(state)


def atomic_torch_save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + '.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_trusted_checkpoint(path, map_location):
    """Load a checkpoint produced by this training pipeline.

    Resume files include Python and NumPy RNG state, so they are intentionally
    not weights-only checkpoints. Never use this helper for untrusted files.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def celeba_pos_weight(dataset, cap=10.0):
    targets = dataset.attr.float()
    targets = (targets > 0).float()
    positive = targets.sum(0)
    negative = targets.shape[0] - positive
    return (negative / positive.clamp_min(1)).clamp(max=cap)
