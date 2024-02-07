import numpy as np
import torch
from torch import nn
from networks.nfn_layers.layer_utils import shape_wsfeat_symmetry
from networks.nfn_layers.common import WeightSpaceFeatures, NetworkSpec


class FlattenWeights(nn.Module):

    def __init__(self, network_spec):
        super().__init__()
        self.network_spec = network_spec

    def forward(self, wsfeat):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        outs = []
        for i in range(len(self.network_spec)):
            w, b, _ = wsfeat[i]
            outs.append(torch.flatten(w, start_dim=2).transpose(1, 2))
            outs.append(b.transpose(1, 2))
        return torch.cat(outs, dim=1)  # (B, N, C)


class UnflattenWeights(nn.Module):

    def __init__(self, network_spec: NetworkSpec):
        super().__init__()
        self.network_spec = network_spec
        self.num_wts, self.num_bs = [], []
        for weight_spec, bias_spec in zip(self.network_spec.weight_spec,
                                          self.network_spec.bias_spec):
            self.num_wts.append(np.prod(weight_spec.shape))
            self.num_bs.append(np.prod(bias_spec.shape))

    def forward(self, x: torch.Tensor) -> WeightSpaceFeatures:
        # x.shape == (bs, num weights and biases, n_chan)
        n_chan = x.shape[2]
        out_weights, out_biases = [], []
        curr_idx = 0
        for i, (weight_spec, bias_spec) in enumerate(
                zip(self.network_spec.weight_spec,
                    self.network_spec.bias_spec)):
            num_wts, num_bs = self.num_wts[i], self.num_bs[i]
            # reshape to (bs, 1, *weight_spec.shape) where 1 is channels.
            wt = x[:, curr_idx:curr_idx + num_wts].transpose(1, 2).reshape(
                -1, n_chan, *weight_spec.shape)
            out_weights.append(wt)
            curr_idx += num_wts
            bs = x[:, curr_idx:curr_idx + num_bs].transpose(1, 2).reshape(
                -1, n_chan, *bias_spec.shape)
            out_biases.append(bs)
            curr_idx += num_bs
        return WeightSpaceFeatures(out_weights, out_biases)


class LearnedScale(nn.Module):

    def __init__(self, network_spec: NetworkSpec, init_scale):
        super().__init__()
        self.weight_scales = nn.ParameterList()
        self.bias_scales = nn.ParameterList()
        for _ in range(len(network_spec)):
            self.weight_scales.append(
                nn.Parameter(torch.tensor(init_scale, dtype=torch.float32)))
            self.bias_scales.append(
                nn.Parameter(torch.tensor(init_scale, dtype=torch.float32)))

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        out_weights, out_biases = [], []
        for i, (weight, bias) in enumerate(zip(wsfeat.weights, wsfeat.biases)):
            out_weights.append(weight * self.weight_scales[i])
            out_biases.append(bias * self.bias_scales[i])
        return WeightSpaceFeatures(out_weights, out_biases, wsfeat.angle)


class TupleOp(nn.Module):

    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        out_weights = [self.op(w) for w in wsfeat.weights]
        out_bias = [self.op(b) for b in wsfeat.biases]

        out_angle = None
        if wsfeat.angle is not None:
            out_angle = wsfeat.angle

        return WeightSpaceFeatures(out_weights, out_bias, out_angle)

    def __repr__(self):
        return f"TupleOp({self.op})"
