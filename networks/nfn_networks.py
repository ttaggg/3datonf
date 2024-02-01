"""Modified fron NFN."""

import torch
from typing import Type, Optional, Union, Tuple, NamedTuple
from torch import nn
from networks.nfn_layers import HNPLinear, NPLinear, HNPLinearAngle, TupleOp, Pointwise, SimpleLayerNorm, LearnedScale, HNPLinearAngleMerge
from networks.nfn_layers import GaussianFourierFeatureTransform, IOSinusoidalEncoding
from networks.nfn_layers.common import NetworkSpec, ArraySpec, WeightSpaceFeatures

MODE2LAYER = {
    "PT":
        Pointwise,
    "NP":
        NPLinear,
    "NP-PosEmb":
        lambda *args, **kwargs: NPLinear(*args, io_embed=True, **kwargs),
    "HNP":
        HNPLinear,
    "HNP_angle":
        HNPLinearAngle,
}

InpEncTypes = Optional[Union[Type[GaussianFourierFeatureTransform],
                             Type[Pointwise]]]


class TransferNet(nn.Module):

    def __init__(
        self,
        weight_shapes,
        bias_shapes,
        hidden_chan,
        hidden_layers,
        inp_enc_cls: InpEncTypes = None,
        pos_enc_cls: Optional[Type[IOSinusoidalEncoding]] = None,
        mode="full",
        lnorm=False,
        out_scale=0.01,
        dropout=0,
    ):
        super().__init__()
        network_spec = NetworkSpec(
            weight_spec=[ArraySpec(x) for x in weight_shapes],
            bias_spec=[ArraySpec(x) for x in bias_shapes])
        layers = []
        in_channels = 1
        if inp_enc_cls == 'gaussian':
            inp_enc = GaussianFourierFeatureTransform(network_spec, in_channels)
            layers.append(inp_enc)
            in_channels = inp_enc.out_channels
        if pos_enc_cls is not None:
            pos_enc = pos_enc_cls(network_spec)
            layers.append(pos_enc)
            in_channels = pos_enc.num_out_chan(in_channels)
        layer_cls = MODE2LAYER[mode]
        layers.append(
            layer_cls(network_spec,
                      in_channels=in_channels,
                      out_channels=hidden_chan))
        if lnorm:
            layers.append(SimpleLayerNorm(network_spec, hidden_chan))
        layers.append(TupleOp(nn.ReLU()))
        if dropout > 0:
            layers.append(TupleOp(nn.Dropout(dropout)))
        for _ in range(hidden_layers - 1):
            layers.append(
                layer_cls(network_spec,
                          in_channels=hidden_chan,
                          out_channels=hidden_chan))
            if lnorm:
                layers.append(SimpleLayerNorm(network_spec, hidden_chan))
            layers.append(TupleOp(nn.ReLU()))
        layers.append(
            layer_cls(network_spec, in_channels=hidden_chan, out_channels=1))
        layers.append(LearnedScale(network_spec, out_scale))
        self.hnet = nn.Sequential(*layers)

    def forward(self, params):
        params = WeightSpaceFeatures(params.weights, params.biases)
        return self.hnet(params)


def reduction(x: torch.tensor, dim=1, keepdim=False):
    x, _ = torch.max(x, dim=dim, keepdim=keepdim)
    return x

def pool(weights, biases):
    
    weights = [w.permute(0, 2, 3, 1) for w in weights] # !!! change
    biases = [b.permute(0, 2, 1) for b in biases] # !!! change
    
    first_w, last_w = weights[0], weights[-1]
    pooled_first_w = first_w.permute(0, 2, 1, 3).flatten(start_dim=2)
    pooled_last_w = last_w.flatten(start_dim=2)
    pooled_first_w = reduction(pooled_first_w, dim=1)
    pooled_last_w = reduction(pooled_last_w, dim=1)
    last_b = biases[-1]
    pooled_last_b = last_b.flatten(start_dim=1)
    pooled_weights = torch.cat(
            [
                reduction(w.permute(0, 3, 1, 2).flatten(start_dim=2),
                                dim=2) for w in weights[1:-1]
            ],
            dim=-1,
    )
    pooled_weights = torch.cat(
            (pooled_weights, pooled_first_w, pooled_last_w), dim=-1)
    pooled_biases = torch.cat(
            [reduction(b, dim=1) for b in biases[:-1]],
            dim=-1)

    pooled_biases = torch.cat((pooled_biases, pooled_last_b), dim=-1)

    pooled_all = torch.cat(
            [pooled_weights, pooled_biases], dim=-1
        )
    return pooled_all


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    angle_encoded: torch.Tensor
    
class NfnSiamese(nn.Module):

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._classifier = nn.Sequential(
            nn.Linear(136, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )

    def forward(self, lhs_weights, lhs_biases, rhs_weights, rhs_biases, angle_encoded):

        # We just set dummy angle for pretraining.
        dummy_angle = torch.zeros_like(angle_encoded)
        lhs_params = Batch(lhs_weights, lhs_biases, dummy_angle)
        rhs_params = Batch(rhs_weights, rhs_biases, dummy_angle)

        lhs = self._model(lhs_params)
        rhs = self._model(rhs_params)
        
        lhs_out = pool(lhs.weights, lhs.biases)
        rhs_out = pool(rhs.weights, rhs.biases)
        
        combined = torch.cat([lhs_out, rhs_out], dim=-1)
        combined = combined.flatten(start_dim=1)
        
        angle = self._classifier(combined)

        return angle


class TransferRotateNet(nn.Module):

    def __init__(
        self,
        weight_shapes,
        bias_shapes,
        hidden_chan,
        hidden_layers,
        inp_enc_cls: InpEncTypes = None,
        pos_enc_cls: Optional[Type[IOSinusoidalEncoding]] = None,
        mode="full",
        lnorm=False,
        out_scale=0.01,
        dropout=0,
    ):
        super().__init__()
        network_spec = NetworkSpec(
            weight_spec=[ArraySpec(x) for x in weight_shapes],
            bias_spec=[ArraySpec(x) for x in bias_shapes])
        self.network_spec = network_spec
        layers = []
        in_channels = 1
        if inp_enc_cls == 'gaussian':
            inp_enc = GaussianFourierFeatureTransform(network_spec, in_channels)
            layers.append(inp_enc)
            in_channels = inp_enc.out_channels
        if pos_enc_cls is not None:
            pos_enc = pos_enc_cls(network_spec)
            layers.append(pos_enc)
            in_channels = pos_enc.num_out_chan(in_channels)
        layers.append(
            HNPLinearAngle(network_spec,
                           in_channels=in_channels,
                           out_channels=hidden_chan))
        if lnorm:
            layers.append(SimpleLayerNorm(network_spec, hidden_chan))
        layers.append(TupleOp(nn.ReLU()))
        if dropout > 0:
            layers.append(TupleOp(nn.Dropout(dropout)))
        for _ in range(hidden_layers - 1):
            layers.append(
                HNPLinearAngle(network_spec,
                               in_channels=hidden_chan,
                               out_channels=hidden_chan))
            if lnorm:
                layers.append(SimpleLayerNorm(network_spec, hidden_chan))
            layers.append(TupleOp(nn.ReLU()))
        layers.append(
            HNPLinearAngle(network_spec, in_channels=hidden_chan, out_channels=1)) # !!! DEBUG

        layers.append(LearnedScale(network_spec, out_scale))
        self.hnet = nn.Sequential(*layers)

    def forward(self, batch):
        params = WeightSpaceFeatures(batch.weights, batch.biases, batch.angle_encoded)
        return self.hnet(params)


class TransferRotateMergeNet(nn.Module):
    def __init__(
        self,
        weight_shapes,
        bias_shapes,
        hidden_chan,
        hidden_layers,
        inp_enc_cls: InpEncTypes = None,
        pos_enc_cls: Optional[Type[IOSinusoidalEncoding]] = None,
        mode="full",
        lnorm=False,
        out_scale=0.01,
        dropout=0,
    ):
        super().__init__()
        network_spec = NetworkSpec(
            weight_spec=[ArraySpec(x) for x in weight_shapes],
            bias_spec=[ArraySpec(x) for x in bias_shapes])
        layers = []
        in_channels = 1
        if inp_enc_cls == 'gaussian':
            inp_enc = GaussianFourierFeatureTransform(network_spec, in_channels)
            layers.append(inp_enc)
            in_channels = inp_enc.out_channels
        if pos_enc_cls is not None:
            pos_enc = pos_enc_cls(network_spec)
            layers.append(pos_enc)
            in_channels = pos_enc.num_out_chan(in_channels)
        layers.append(
            HNPLinearAngleMerge(network_spec,
                                in_channels=in_channels,
                                out_channels=hidden_chan))
        if lnorm:
            layers.append(SimpleLayerNorm(network_spec, hidden_chan))
        layers.append(TupleOp(nn.ReLU()))
        if dropout > 0:
            layers.append(TupleOp(nn.Dropout(dropout)))
        for _ in range(hidden_layers - 1):
            layers.append(
                HNPLinearAngleMerge(network_spec,
                                    in_channels=hidden_chan,
                                    out_channels=hidden_chan))
            if lnorm:
                layers.append(SimpleLayerNorm(network_spec, hidden_chan))
            layers.append(TupleOp(nn.ReLU()))
        layers.append(
            HNPLinearAngleMerge(network_spec, in_channels=hidden_chan, out_channels=1))

        layers.append(LearnedScale(network_spec, out_scale))
        self.hnet = nn.Sequential(*layers)

    def forward(self, batch):
        params = WeightSpaceFeatures(batch.weights, batch.biases,
                                     batch.angle_encoded)
        return self.hnet(params)
