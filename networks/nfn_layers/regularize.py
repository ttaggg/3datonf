import numpy as np
import torch
from torch import nn
from einops.layers.torch import Rearrange

from networks.nfn_layers.common import WeightSpaceFeatures
from networks.nfn_layers.layer_utils import shape_wsfeat_symmetry, unshape_wsfeat_symmetry


class SimpleLayerNorm(nn.Module):

    def __init__(self, network_spec, channels):
        super().__init__()
        self.network_spec = network_spec
        self.channels = channels
        self.w_norms, self.v_norms = nn.ModuleList(), nn.ModuleList()
        for i in range(len(network_spec)):
            eff_channels = int(channels *
                               np.prod(network_spec.weight_spec[i].shape[2:]))
            self.w_norms.append(ChannelLayerNorm(eff_channels))
            self.v_norms.append(ChannelLayerNorm(channels))

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        out_weights, out_biases = [], []
        for i in range(len(self.network_spec)):
            weight, bias, _ = wsfeat[i]
            out_weights.append(self.w_norms[i](weight))
            out_biases.append(self.v_norms[i](bias))

        out_angle = None
        if wsfeat.angle is not None:
            out_angle = wsfeat.angle

        return unshape_wsfeat_symmetry(
            WeightSpaceFeatures(out_weights, out_biases, out_angle),
            self.network_spec)

    def __repr__(self):
        return f"SimpleLayerNorm(channels={self.channels})"



class ChannelLayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.channels_last = Rearrange("b c ... -> b ... c")
        self.channels_first = Rearrange("b ... c -> b c ...")

    def forward(self, x):
        # x.shape = (b, c, ...)
        x = self.channels_last(x)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        out = self.channels_first(x * self.gamma + self.beta)
        return out
