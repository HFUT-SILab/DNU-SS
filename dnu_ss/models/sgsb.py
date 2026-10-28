"""Saliency-Guided Spiking Branch (Sections III-D and III-E)."""

import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron, surrogate
from .prior import PalmprintSaliencyPrior
from .spgtr import SPGTR


class SpatialAlignment(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(64, channels, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, features):
        return self.bn(self.conv(features))


def encoding_backend(requested):
    if requested == "torch":
        return "torch"
    try:
        import cupy

        if torch.cuda.is_available() and cupy.cuda.runtime.getDeviceCount() > 0:
            import numpy as np

            # SpikingJelly 0.0.0.0.14 generates kernels using these removed aliases.
            for name, value in [("int", int), ("float", float), ("bool", bool)]:
                np.__dict__.setdefault(name, value)
            return "cupy"
    except Exception:
        pass
    if requested == "cupy":
        raise RuntimeError(
            "CuPy LIF backend requested but unavailable; choose torch or install matching CuPy/CUDA"
        )
    return "torch"


class SGSB(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.timesteps = config.timesteps
        self.resize_large = nn.Upsample(
            size=(config.token_grid, config.token_grid),
            mode="bilinear",
            align_corners=False,
        )
        self.resize_small = nn.Upsample(
            size=(config.token_grid, config.token_grid),
            mode="bilinear",
            align_corners=False,
        )
        self.align_large = SpatialAlignment(config.embedding_dim)
        self.align_small = SpatialAlignment(config.embedding_dim)
        self.use_prior = config.use_prior
        if self.use_prior:
            self.prior = PalmprintSaliencyPrior(
                config.token_grid,
                config.prior_gray_window,
                config.prior_coherence_window,
                config.prior_weights,
            )
            initial_ratio = config.prior_gain_initial / config.prior_gain_maximum
            self.prior_gain_logit = nn.Parameter(
                torch.tensor(float(math.log(initial_ratio / (1.0 - initial_ratio))))
            )
        self.prior_gain_maximum = config.prior_gain_maximum
        self.prior_strength = 1.0
        self.last_prior = None
        self.spike_encoder = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(alpha=2.0),
            step_mode="m",
            backend=encoding_backend(config.lif_backend),
            store_v_seq=False,
        )
        self.spgtr_blocks = nn.ModuleList(
            [SPGTR(config, ratio) for ratio in config.skip_ratios]
        )
        self.descriptor_projection = nn.Linear(config.embedding_dim, 1024)

    def effective_prior_gain(self):
        if not self.use_prior:
            return self.align_large.conv.weight.new_tensor(0.0)
        return (
            torch.sigmoid(self.prior_gain_logit)
            * self.prior_gain_maximum
            * self.prior_strength
        )

    def forward(self, large, small, roi):
        functional.reset_net(self)
        aligned = self.align_large(self.resize_large(large)) + self.align_small(
            self.resize_small(small)
        )
        if self.use_prior:
            prior = self.prior(roi, large)
            aligned = aligned * (1.0 + self.effective_prior_gain() * prior)
        else:
            prior = aligned.new_zeros(
                aligned.size(0), 1, aligned.size(2), aligned.size(3)
            )
        self.last_prior = prior.detach()
        repeated = (
            aligned.unsqueeze(0).expand(self.timesteps, *aligned.shape).contiguous()
        )
        responses = self.spike_encoder(repeated)
        for block in self.spgtr_blocks:
            responses = block(responses, prior)
        spatial_average = responses.mean(dim=[3, 4])
        temporal_average = spatial_average.mean(dim=0)
        return self.descriptor_projection(temporal_average)
