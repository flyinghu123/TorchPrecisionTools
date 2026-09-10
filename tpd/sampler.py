"""
Tensor sampling utilities.
Supports uniform and random sampling with configurable seed.
"""

import numpy as np
import torch

from .config import Config


class TensorSampler:
    """Samples elements from tensors with uniform or random strategy."""

    def __init__(self):
        self._rng = None
        if Config.SAMPLE_MODE == "random":
            if Config.SAMPLE_SEED is not None:
                self._rng = np.random.default_rng(Config.SAMPLE_SEED)
            else:
                self._rng = np.random.default_rng()

    def sample(self, tensor: torch.Tensor) -> list[float]:
        """
        Sample elements from a tensor.
        Returns a list of sampled values as Python floats.
        """
        if tensor.numel() == 0:
            return []

        # Flatten tensor
        flat = tensor.detach().cpu().float().numpy().flatten()
        total_elements = len(flat)
        sample_count = min(Config.SAMPLE_COUNT, total_elements)

        if sample_count == 0:
            return []

        if Config.SAMPLE_MODE == "uniform":
            # Uniform sampling: evenly spaced indices
            indices = np.linspace(0, total_elements - 1, sample_count, dtype=int)
        else:  # random
            if self._rng is None:
                # Should not happen, but fallback
                self._rng = np.random.default_rng()
            indices = self._rng.integers(0, total_elements, size=sample_count)

        sampled_values = flat[indices].tolist()
        return sampled_values


def safe_sample(tensor: torch.Tensor) -> list[float]:
    """Safely sample from a tensor, handling edge cases."""
    try:
        sampler = TensorSampler()
        return sampler.sample(tensor)
    except Exception as e:
        return [float("nan")]  # Return NaN on error
