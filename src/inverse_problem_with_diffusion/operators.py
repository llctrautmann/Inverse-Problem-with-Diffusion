"""
Forward operators A for inverse problems, y = A(x) + noise.

All operators act on image batches (B, C, H, W) with values in [0, 1] and are
differentiable, so they can be used as guidance in dps.py. Each factory returns
a function; `make_operator` builds one by name from config values.
"""

import math
from typing import Callable

import torch
import torch.nn.functional as F


def gaussian_blur(sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Blur with a Gaussian of standard deviation `sigma` pixels (bigger = harder)."""
    radius = max(1, math.ceil(3 * sigma))

    def operator(x):
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-(t**2) / (2 * sigma**2))
        k = k / k.sum()
        c = x.shape[1]
        k_h = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        k_v = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k_h, groups=c)
        return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k_v, groups=c)

    return operator


def downsample(factor: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Average-pool by `factor`; the measurement is `factor` times smaller per side."""

    def operator(x):
        return F.avg_pool2d(x, factor)

    return operator


def random_mask(keep_fraction: float, seed: int = 0) -> Callable[[torch.Tensor], torch.Tensor]:
    """Keep a random `keep_fraction` of the pixels and zero the rest (fixed by `seed`)."""
    cache = {}

    def operator(x):
        shape = tuple(x.shape[-2:])
        if shape not in cache:
            g = torch.Generator().manual_seed(seed)
            cache[shape] = (torch.rand(1, 1, *shape, generator=g) < keep_fraction).float()
        return x * cache[shape].to(x.device, x.dtype)

    return operator


def box_mask(side_fraction: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Zero a centred square whose side is `side_fraction` of the image side."""

    def operator(x):
        h, w = x.shape[-2:]
        bh, bw = round(h * side_fraction), round(w * side_fraction)
        top, left = (h - bh) // 2, (w - bw) // 2
        mask = torch.ones(1, 1, h, w, device=x.device, dtype=x.dtype)
        mask[..., top : top + bh, left : left + bw] = 0
        return x * mask

    return operator


OPERATORS = {
    "gaussian_blur": gaussian_blur,
    "downsample": downsample,
    "random_mask": random_mask,
    "box_mask": box_mask,
}


def make_operator(name: str, **params) -> Callable[[torch.Tensor], torch.Tensor]:
    if name not in OPERATORS:
        raise ValueError(f"Unknown operator '{name}'. Choose from {sorted(OPERATORS)}.")
    return OPERATORS[name](**params)


def measure(operator, x, noise_sigma: float, generator=None) -> torch.Tensor:
    """Simulate a measurement y = A(x) + N(0, noise_sigma^2)."""
    y = operator(x)
    noise = torch.randn(y.shape, generator=generator).to(y.device, y.dtype)
    return y + noise_sigma * noise
