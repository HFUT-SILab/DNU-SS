"""Dual-Scale Gabor Texture Stem (Section III-B)."""

import torch.nn as nn
import torch.nn.functional as F
from .layers import LearnableGaborConv2d, EfficientChannelAttention


class DGTStem(nn.Module):
    """ROI [B,1,128,128] -> shared F_L [B,64,30,30], F_S [B,64,14,14]."""

    def __init__(self):
        super().__init__()
        self.gabor_large = LearnableGaborConv2d(
            1, 36, 17, stride=2, padding=8, init_ratio=0.4
        )
        self.gabor_small = LearnableGaborConv2d(
            1, 36, 9, stride=2, padding=4, init_ratio=0.25
        )
        self.eca_large = EfficientChannelAttention(36)
        self.eca_small = EfficientChannelAttention(36)
        self.small_alignment = nn.Conv2d(36, 36, 5, stride=2, padding=2)
        self.aggregate_large = nn.Conv2d(36, 64, 5)
        self.aggregate_small = nn.Conv2d(36, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.modulation_strength = 0.8
        self.mask_steepness = 18.0
        self.eps = 1e-6

    def forward(self, roi):
        large = self.gabor_large(roi)
        small = self.gabor_small(roi)
        mean = large.mean(dim=(2, 3), keepdim=True)
        std = large.std(dim=(2, 3), keepdim=True, unbiased=False)
        mask = ((mean - large) / (std + self.eps) * self.mask_steepness).sigmoid()
        local_mean = F.avg_pool2d(small, 3, stride=1, padding=1)
        small = small + self.modulation_strength * mask * (local_mean - small)
        large = self.eca_large(large)
        small = self.small_alignment(self.eca_small(small))
        return self.pool(F.relu(self.aggregate_large(large))), self.pool(
            F.relu(self.aggregate_small(small))
        )
