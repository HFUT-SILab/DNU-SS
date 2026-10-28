"""STRI, SERT, TIMR and TIMR Gate (Section III-E)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_sum_normalize(value):
    return value / (value.sum(dim=-1, keepdim=True) + 1e-6)


def spatial_minmax_normalize(value):
    low = value.min(dim=-1, keepdim=True)[0]
    high = value.max(dim=-1, keepdim=True)[0]
    return (value - low) / (high - low + 1e-6)


class STRI(nn.Module):
    """Spatio-Temporal Response Informativeness: [T,B,C,H,W] -> [T,B,N]."""

    def __init__(self, window=3):
        super().__init__()
        self.window = window

    def forward(self, responses):
        timesteps, batch, channels, height, width = responses.shape
        folded = responses.reshape(timesteps * batch, channels, height, width)
        average = F.avg_pool2d(folded, self.window, stride=1, padding=self.window // 2)
        silent = (folded.norm(dim=1) <= 1e-6) & (average.norm(dim=1) <= 1e-6)
        cosine = (
            F.normalize(folded, dim=1, eps=1e-6) * F.normalize(average, dim=1, eps=1e-6)
        ).sum(dim=1)
        spatial = (1.0 - cosine).clamp_min(0.0).masked_fill(silent, 0.0)
        spatial = spatial_sum_normalize(spatial.flatten(1)).reshape(
            timesteps, batch, -1
        )
        if timesteps == 1:
            temporal = torch.zeros_like(spatial)
        else:
            difference = (responses[1:] - responses[:-1]).abs().sum(dim=2)
            temporal = spatial_sum_normalize(
                difference.reshape((timesteps - 1) * batch, -1)
            ).reshape(timesteps - 1, batch, -1)
            temporal = torch.cat([torch.zeros_like(temporal[:1]), temporal], dim=0)
        return spatial + temporal


class SERT(nn.Module):
    """Soft Early-Response Timing with an explicit inactive-response mask."""

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, responses):
        timesteps = responses.size(0)
        peak = responses.abs().amax(dim=2).flatten(2)
        active = peak.sum(dim=0) > 1e-6
        weights = torch.softmax(peak / self.temperature, dim=0)
        step = torch.arange(
            timesteps, device=responses.device, dtype=responses.dtype
        ).view(timesteps, 1, 1)
        centroid = (weights * step).sum(dim=0)
        early = 1.0 - centroid / max(1, timesteps - 1)
        return spatial_minmax_normalize(early) * active.to(early.dtype)


class TIMR(nn.Module):
    """Time-Integrated Mean Response; decay is fixed, not task-trainable."""

    def __init__(self, decay=0.6):
        super().__init__()
        self.register_buffer(
            "decay_logit", torch.logit(torch.tensor(float(decay)).clamp(1e-4, 1 - 1e-4))
        )

    def forward(self, responses):
        timesteps = responses.size(0)
        step = torch.arange(timesteps, device=responses.device, dtype=responses.dtype)
        weights = torch.sigmoid(self.decay_logit) ** step
        weights = weights / (weights.sum() + 1e-6)
        activity = responses.abs().mean(dim=2).flatten(2)
        return spatial_minmax_normalize(
            (activity * weights.view(timesteps, 1, 1)).sum(dim=0)
        )


class TIMRGate(nn.Module):
    """Residual multiplicative modulation of SERT by normalized TIMR."""

    def __init__(self, strength=0.1):
        super().__init__()
        self.strength = strength

    def forward(self, sert, timr):
        return spatial_minmax_normalize(sert * (1.0 + self.strength * timr))
