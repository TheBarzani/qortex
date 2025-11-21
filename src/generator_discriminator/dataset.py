from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SwaptionGANDataset(Dataset):
    """
    Simple dataset for conditional GAN on swaption surfaces.

    - context: current-day surface (scaled), shape (N, context_dim)
    - target:  next-day surface (scaled), shape (N, target_dim)

    Both context and target are expected as numpy arrays.
    """

    def __init__(self, context: np.ndarray, target: np.ndarray):
        assert context.shape[0] == target.shape[0], "Mismatched number of samples"
        self.context = torch.tensor(context, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        return self.context.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.context[idx], self.target[idx]