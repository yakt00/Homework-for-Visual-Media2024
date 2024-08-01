"""Data loader."""

import os

import torch
from dataset_cirr import CIRR


# Default data directory (/path/pycls/pycls/datasets/data)

def _construct_loader(_DATA_DIR, split, mode, transform, blip_transform, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    # Construct the dataset
    dataset = CIRR(_DATA_DIR, split, mode, transform, blip_transform)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=8,
        pin_memory=False,
        drop_last=drop_last,
    )
    return loader

def construct_loader(_DATA_DIR, split, mode, transform, blip_transform, batch_size):
    """Test loader wrapper."""
    return _construct_loader(
        _DATA_DIR=_DATA_DIR,
        split=split,
        mode = mode,
        transform = transform,
        blip_transform = blip_transform,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
