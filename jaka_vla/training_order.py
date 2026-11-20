"""Reproducible full shuffled passes with a resumable frame cursor."""
import numpy as np


def sample_plan(indices, count, seed, offset=0):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError('Eligible frame indices must be nonempty and unique')
    if count < 1 or offset < 0:
        raise ValueError('Invalid sample count or offset')
    rng = np.random.default_rng(seed)
    order = []
    remaining = offset
    while len(order) < count:
        block = rng.permutation(indices)
        if remaining >= len(block):
            remaining -= len(block)
            continue
        order.extend(block[remaining:].tolist())
        remaining = 0
    return np.asarray(order[:count], dtype=np.int64)
