"""Texture Enhancement Branch (Section III-C)."""

import torch
import torch.nn as nn


class TEB(nn.Module):
    """Refine the shared grids, concatenate 7424 coordinates, project to 1024."""

    def __init__(self, defer_projection=False):
        super().__init__()
        self.refine_large = nn.Conv2d(64, 32, 3, stride=2)
        self.refine_small = nn.Conv2d(64, 32, 3, stride=2)
        if not defer_projection:
            self.initialize_projection()

    def initialize_projection(self):
        if hasattr(self, "fc1"):
            raise RuntimeError("TEB projection is already initialized")
        self.fc1 = nn.Linear(7424, 2048)
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, large, small):
        local_large = self.refine_large(large).reshape(large.size(0), -1)
        local_small = self.refine_small(small).reshape(small.size(0), -1)
        return self.fc2(self.fc1(torch.cat([local_large, local_small], dim=1)))
