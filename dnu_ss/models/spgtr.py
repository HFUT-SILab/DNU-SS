"""Saliency Prior-Guided Token Retention (Section III-E)."""

import torch
import torch.nn as nn
from .scoring import STRI, SERT, TIMR, TIMRGate, spatial_sum_normalize
from .spiking import SpikingSSA, SpikingMLP


class SPGTR(nn.Module):
    """Per-step ranking, selected interaction, and full-grid bypass reassembly."""

    def __init__(self, config, skip_ratio):
        super().__init__()
        self.temporal_mode = config.temporal_mode
        self.skip_ratio = skip_ratio
        self.prior_weight = config.prior_score_weight
        self.sert_weight = config.sert_score_weight
        self.stri = STRI(config.stri_window)
        if skip_ratio > 0 and self.temporal_mode != "none":
            self.sert = SERT(config.sert_temperature)
            if self.temporal_mode == "timr_gated_sert":
                self.timr = TIMR(config.timr_decay)
                self.timr_gate = TIMRGate(config.timr_gate_strength)
        self.ssa = SpikingSSA(config.embedding_dim, config.num_heads)
        self.mlp = SpikingMLP(config.embedding_dim, config.mlp_ratio)
        self.capture_scores = False
        self.last_selection = None
        self.last_scores = None

    def forward(self, responses, prior):
        timesteps, batch, channels, height, width = responses.shape
        positions = height * width
        count = max(1, int((1.0 - self.skip_ratio) * positions))
        with torch.no_grad():
            stri = self.stri(responses)
            prior_score = spatial_sum_normalize(prior.reshape(batch, -1))
            temporal_weight = (
                self.sert_weight
                if self.skip_ratio > 0 and self.temporal_mode != "none"
                else 0.0
            )
            joint = max(
                0.0, 1.0 - self.prior_weight - temporal_weight
            ) * spatial_sum_normalize(stri)
            joint = joint + self.prior_weight * prior_score.unsqueeze(0).expand(
                timesteps, batch, positions
            )
            sert = timr = gated_sert = None
            if self.skip_ratio > 0 and self.temporal_mode != "none":
                sert = self.sert(responses)
                if self.temporal_mode == "timr_gated_sert":
                    timr = self.timr(responses)
                    gated_sert = self.timr_gate(sert, timr)
                else:
                    gated_sert = sert
                joint = joint + temporal_weight * spatial_sum_normalize(
                    gated_sert
                ).unsqueeze(0).expand(timesteps, batch, positions)
            joint = spatial_sum_normalize(joint)
            selected = (
                joint.reshape(timesteps * batch, positions).topk(count, dim=1).indices
            )
        self.last_selection = selected.reshape(timesteps, batch, count).detach()
        if self.capture_scores:
            self.last_scores = dict(
                stri=stri,
                prior=prior,
                sert=sert,
                timr=timr,
                gated_sert=gated_sert,
                joint=joint,
            )
        else:
            self.last_scores = None
        flat = responses.permute(0, 1, 3, 4, 2).reshape(
            timesteps * batch, positions, channels
        )
        indices = selected.unsqueeze(-1).expand(timesteps * batch, count, channels)
        retained = torch.gather(flat, 1, indices).reshape(
            timesteps, batch, count, channels
        )
        interacted = self.ssa(retained, self.last_selection, positions) + retained
        updated = self.mlp(interacted, self.last_selection, positions) + interacted
        mask = torch.zeros(
            timesteps * batch,
            positions,
            1,
            dtype=responses.dtype,
            device=responses.device,
        )
        mask.scatter_(1, selected.unsqueeze(-1), 1.0)
        scattered = torch.zeros_like(flat)
        scattered.scatter_(
            1, indices, updated.reshape(timesteps * batch, count, channels)
        )
        output = flat * (1.0 - mask) + scattered * mask
        return output.reshape(timesteps, batch, height, width, channels).permute(
            0, 1, 4, 2, 3
        )
