"""Spatial-indexed LIF, SpikingSSA and SpikingMLP for SPGTR."""

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate


class SpatialIndexedLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.tau = 2.0
        self.v_threshold = 1.0
        self.v_reset = 0.0
        self.surrogate = surrogate.ATan(alpha=2.0)

    def forward(self, x_seq, spatial_indices, num_positions):
        T, B, K, C = x_seq.shape
        v_full = x_seq.new_zeros(B, num_positions, C)
        spikes = []
        for t in range(T):
            idx = spatial_indices[t]
            idx_exp = idx.unsqueeze(-1).expand(B, K, C)
            v_sel = torch.gather(v_full, 1, idx_exp)
            v_sel = v_sel + (x_seq[t] - (v_sel - self.v_reset)) / self.tau
            spike = self.surrogate(v_sel - self.v_threshold)
            v_sel = self.v_reset * spike + (1.0 - spike) * v_sel
            v_full = v_full.scatter(1, idx_exp, v_sel)
            spikes.append(spike)
        return torch.stack(spikes, dim=0)


class SpikingSSA(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.q_linear = nn.Linear(dim, dim, bias=False)
        self.k_linear = nn.Linear(dim, dim, bias=False)
        self.v_linear = nn.Linear(dim, dim, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.q_lif = SpatialIndexedLIF()
        self.k_lif = SpatialIndexedLIF()
        self.v_lif = SpatialIndexedLIF()
        self.attention_lif = SpatialIndexedLIF()
        self.proj_linear = nn.Linear(dim, dim, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.projection_lif = SpatialIndexedLIF()

    @staticmethod
    def _bn(bn, x):
        T, B, K, D = x.shape
        flat = x.reshape(T * B * K, D)
        flat = bn(flat)
        return flat.reshape(T, B, K, D)

    def forward(self, x_seq, spatial_indices, num_positions):
        T, B, K, D = x_seq.shape
        q_pre = self._bn(self.q_bn, self.q_linear(x_seq))
        k_pre = self._bn(self.k_bn, self.k_linear(x_seq))
        v_pre = self._bn(self.v_bn, self.v_linear(x_seq))
        assert num_positions is not None
        q = self.q_lif(q_pre, spatial_indices, num_positions)
        k = self.k_lif(k_pre, spatial_indices, num_positions)
        v = self.v_lif(v_pre, spatial_indices, num_positions)
        q = q.reshape(T * B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(T * B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(T * B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = self.scale * (q @ (k.transpose(-2, -1) @ v))
        out = out.permute(0, 2, 1, 3).reshape(T, B, K, D)
        out = self.attention_lif(out, spatial_indices, num_positions)
        out = self._bn(self.proj_bn, self.proj_linear(out))
        out = self.projection_lif(out, spatial_indices, num_positions)
        return out


class SpikingMLP(nn.Module):
    def __init__(self, dim, hidden_ratio=4):
        super().__init__()
        hidden = dim * hidden_ratio
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lif1 = SpatialIndexedLIF()
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)
        self.lif2 = SpatialIndexedLIF()

    @staticmethod
    def _bn(bn, x):
        T, B, K, C = x.shape
        flat = bn(x.reshape(T * B * K, C))
        return flat.reshape(T, B, K, C)

    def forward(self, x_seq, spatial_indices, num_positions):
        h = self.fc1(x_seq)
        h = self._bn(self.bn1, h)
        assert num_positions is not None
        h = self.lif1(h, spatial_indices, num_positions)
        h = self.fc2(h)
        h = self._bn(self.bn2, h)
        h = self.lif2(h, spatial_indices, num_positions)
        return h
