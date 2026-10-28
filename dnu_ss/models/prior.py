"""Palmprint Saliency Prior (Section III-D1); fixed fusion, stopped gradients."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PalmprintSaliencyPrior(nn.Module):
    def __init__(
        self, grid=15, gray_window=7, coherence_window=7, weights=(0.4, 0.2, 0.4)
    ):
        super().__init__()
        self.grid = grid
        self.gray_window = gray_window
        self.coherence_window = coherence_window
        self.weights = tuple(weights)
        self.eps = 1e-6
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )

    def normalize(self, value):
        flat = value.reshape(value.shape[0], -1)
        low = flat.min(dim=1)[0].view(-1, 1, 1, 1)
        high = flat.max(dim=1)[0].view(-1, 1, 1, 1)
        return (value - low) / (high - low + self.eps)

    def resize(self, value):
        return F.interpolate(
            value, (self.grid, self.grid), mode="bilinear", align_corners=False
        )

    @torch.no_grad()
    def forward(self, roi, large_features):
        mean = F.avg_pool2d(
            roi,
            self.gray_window,
            stride=1,
            padding=self.gray_window // 2,
            count_include_pad=False,
        )
        gray = self.normalize(self.resize(self.normalize((roi - mean).abs())))
        gx = F.conv2d(roi, self.sobel_x, padding=1)
        gy = F.conv2d(roi, self.sobel_y, padding=1)
        magnitude = self.normalize(torch.sqrt(gx * gx + gy * gy + self.eps))
        denominator = gx * gx + gy * gy + self.eps
        vx = (gx * gx - gy * gy) / denominator
        vy = (2.0 * gx * gy) / denominator
        vx = F.avg_pool2d(
            vx,
            self.coherence_window,
            stride=1,
            padding=self.coherence_window // 2,
            count_include_pad=False,
        )
        vy = F.avg_pool2d(
            vy,
            self.coherence_window,
            stride=1,
            padding=self.coherence_window // 2,
            count_include_pad=False,
        )
        coherence = torch.sqrt(vx * vx + vy * vy + self.eps).clamp(0, 1)
        gradient = self.normalize(self.resize(magnitude * coherence))
        directional = self.normalize(
            self.resize(large_features.abs().mean(dim=1, keepdim=True))
        )
        weights = self.sobel_x.new_tensor(self.weights).to(
            device=roi.device, dtype=roi.dtype
        )
        prior = weights[0] * gray + weights[1] * gradient
        prior = prior + weights[2] * directional
        return self.normalize(prior)
