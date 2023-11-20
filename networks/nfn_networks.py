"""Modified fron NFN."""

from typing import Type, Optional, Union
from torch import nn
from networks.nfn_layers import HNPLinear, NPLinear, TupleOp, Pointwise, SimpleLayerNorm, LearnedScale
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
        if inp_enc_cls is not None:
            inp_enc = inp_enc_cls(network_spec, in_channels)
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
        # print(len(params))
        # print(len(params[0]))
        # print(params[0][0].shape)
        # print(params[0][1].shape)
        # 1/0
        params = WeightSpaceFeatures(params.weights, params.biases)
        return self.hnet(params)
