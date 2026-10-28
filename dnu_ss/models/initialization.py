"""Preserve reference initialization without retaining inactive network modules."""

import math
import torch
from torch import nn


def consume_sgsb_initialization(config):
    """Advance the CPU initializer stream used by the reference TEB-only run.

    The reference constructed, froze and bypassed SGSB. Consume its weight
    draws without constructing or retaining a second branch. Normalization,
    neurons and fixed scoring buffers do not draw random numbers.
    """
    dim = config.embedding_dim

    def weight(shape):
        nn.init.kaiming_uniform_(torch.empty(shape), a=math.sqrt(5))

    def linear(inputs, outputs, bias=False):
        weight((outputs, inputs))
        if bias:
            bound = 1 / math.sqrt(inputs)
            nn.init.uniform_(torch.empty(outputs), -bound, bound)

    for _ in range(2):
        weight((dim, 64, 3, 3))
    for _ in config.skip_ratios:
        for _ in range(4):
            linear(dim, dim)
        linear(dim, dim * config.mlp_ratio)
        linear(dim * config.mlp_ratio, dim)
    linear(dim, 1024, bias=True)


def training_parameters(model):
    """Reference traversal order, expressed using the paper's native modules."""
    yield from model.dgt_stem.parameters()
    yield from model.teb.refine_large.parameters()
    yield from model.teb.refine_small.parameters()
    if model.sgsb is not None:
        yield from model.sgsb.parameters()
    yield from model.teb.fc1.parameters()
    yield from model.teb.fc2.parameters()
    yield from model.arcface.parameters()
