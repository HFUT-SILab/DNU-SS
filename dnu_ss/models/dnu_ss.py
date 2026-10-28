"""Complete DNU-SS architecture, using the paper's module boundaries and names."""

import torch.nn as nn
import torch.nn.functional as F
from ..config import ModelConfig
from .dgt_stem import DGTStem
from .teb import TEB
from .sgsb import SGSB
from .layers import ArcFaceClassifier
from .initialization import consume_sgsb_initialization


class DNUSS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.dgt_stem = DGTStem()
        self.teb = TEB(defer_projection=True)
        if self.config.use_sgsb:
            self.sgsb = SGSB(self.config)
        else:
            consume_sgsb_initialization(self.config)
            self.sgsb = None
        self.teb.initialize_projection()
        self.dropout = nn.Dropout(self.config.dropout)
        self.arcface = ArcFaceClassifier(1024, self.config.num_classes, s=30.0, m=0.5)

    def describe(self, roi):
        """Unnormalized fused descriptor [B,1024], shared by train and evaluate."""
        large, small = self.dgt_stem(roi)
        texture = self.teb(large, small)
        if self.sgsb is None:
            return texture
        spiking = self.sgsb(large, small, roi)
        weight = self.config.teb_fusion_weight
        return texture * weight + spiking * (1 - weight)

    def forward(self, roi, labels=None):
        """ArcFace logits [B,num_classes]; training requires identity labels."""
        return self.arcface(self.dropout(self.describe(roi)), labels)

    def extract_features(self, roi):
        """L2-normalized verification descriptor; caller selects eval/no_grad."""
        return F.normalize(self.describe(roi), p=2, dim=1, eps=1e-8)
