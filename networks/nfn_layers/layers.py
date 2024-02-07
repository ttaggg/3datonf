import numpy as np
import torch
from torch import nn
from einops.layers.torch import Rearrange
from networks.nfn_layers.common import WeightSpaceFeatures, NetworkSpec
from networks.nfn_layers.layer_utils import set_init_, shape_wsfeat_symmetry, unshape_wsfeat_symmetry


class Pointwise(nn.Module):
    """Assumes full row/col exchangeability of weights in each layer."""

    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels):
        super().__init__()
        self.network_spec = network_spec
        self.in_channels = in_channels
        self.out_channels = out_channels
        # register num_layers in_channels -> out_channels linear layers
        self.weight_maps, self.bias_maps = nn.ModuleList(), nn.ModuleList()
        filter_facs = [
            int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec
        ]
        for i in range(len(network_spec)):
            fac_i = filter_facs[i]
            self.weight_maps.append(
                nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
            self.bias_maps.append(nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, wsfeat: WeightSpaceFeatures):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        # weights is a list of tensors, each with shape (B, C_in, nrow, ncol)
        # each tensor is reshaped to (B, nrow * ncol, C_in), passed through a linear
        # layer, then reshaped back to (B, C_out, nrow, ncol)
        out_weights, out_biases = [], []
        for i in range(len(self.network_spec)):
            weight, bias, angle = wsfeat[i]
            out_weights.append(self.weight_maps[i](weight))
            out_biases.append(self.bias_maps[i](bias))
        return unshape_wsfeat_symmetry(
            WeightSpaceFeatures(out_weights, out_biases, angle),
            self.network_spec)

    def __repr__(self):
        return f"Pointwise(in_channels={self.in_channels}, out_channels={self.out_channels})"


class HNPLinearAngleSum(nn.Module):

    def __init__(
        self,
        network_spec: NetworkSpec,
        in_channels,
        out_channels,
        input_dim,
        hidden_size,
        add_features,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.add_features = add_features
        self.network_spec = network_spec
        n_in, n_out = network_spec.get_io()
        self.n_in, self.n_out = n_in, n_out
        self.c_in, self.c_out = in_channels, out_channels
        self.L = len(network_spec)
        filter_facs = [
            int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec
        ]
        n_rc_inp = n_in * filter_facs[0] + n_out * filter_facs[
            -1] + self.L + n_out
        for i in range(self.L):
            n_rc_inp += filter_facs[i]

        self._angle_emb = {}
        self._angle_emb_merge = {}

        self._angle_emb_bias = {}
        self._angle_emb_merge_bias = {}

        for i in range(self.L):
            fac_i = filter_facs[i]
            if i == 0:
                fac_ip1 = filter_facs[i + 1]
                rpt_in = (fac_i * n_in + fac_ip1 + 1)
                if i + 1 == self.L - 1:
                    rpt_in += n_out * fac_ip1
                self.w0_rpt = nn.Conv1d(rpt_in * in_channels,
                                        n_in * fac_i * out_channels, 1)
                self.w0_rbdcst = nn.Linear(n_rc_inp * in_channels,
                                           n_in * fac_i * out_channels)

                self.bias_0 = nn.Conv1d(rpt_in * in_channels, out_channels, 1)
                self.bias_0_rc = nn.Linear(n_rc_inp * in_channels, out_channels)
                set_init_(self.bias_0, self.bias_0_rc)

                ## WEIGHTS
                self._angle_emb[i] = nn.Sequential(
                    nn.Linear(input_dim, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, n_in * fac_i * out_channels))
                self.add_module(f"layer_{i}_angle", self._angle_emb[i])

                # BIASES
                self._angle_emb_bias[i] = nn.Sequential(
                    nn.Linear(input_dim, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, out_channels))
                self.add_module(f"layer_{i}_angle_bias",
                                self._angle_emb_bias[i])

            elif i == self.L - 1:
                fac_im1 = filter_facs[i - 1]
                cpt_in = (fac_i * n_out + fac_im1)
                if i - 1 == 0:
                    cpt_in += n_in * fac_im1
                self.wfin_cpt = nn.Conv1d(cpt_in * in_channels,
                                          n_out * fac_i * out_channels, 1)
                self.wfin_cbdcst = nn.Linear(n_rc_inp * in_channels,
                                             n_out * fac_i * out_channels)
                set_init_(self.wfin_cpt, self.wfin_cbdcst)

                self.bias_fin_rc = nn.Linear(n_rc_inp * in_channels,
                                             n_out * out_channels)

                # WEIGHTS
                self._angle_emb[i] = nn.Sequential(
                    nn.Linear(input_dim, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, n_out * fac_i * out_channels))
                self.add_module(f"layer_{i}_angle", self._angle_emb[i])

                # BIASES
                self._angle_emb_bias[i] = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(),
                    nn.Linear(64, out_channels))
                self.add_module(f"layer_{i}_angle_bias",
                                self._angle_emb_bias[i])

            else:
                self.add_module(
                    f"layer_{i}",
                    nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
                self.add_module(
                    f"layer_{i}_rc",
                    nn.Linear(n_rc_inp * in_channels, fac_i * out_channels))

                # WEIGHTS
                self._angle_emb[i] = nn.Sequential(
                    nn.Linear(input_dim, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size,
                              add_features * n_out * fac_i * out_channels))
                self.add_module(f"layer_{i}_angle", self._angle_emb[i])

                # BIASES
                self._angle_emb_bias[i] = nn.Sequential(
                    nn.Linear(input_dim, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, out_channels))
                self.add_module(f"layer_{i}_angle_bias",
                                self._angle_emb_bias[i])

                fac_im1, fac_ip1 = filter_facs[i - 1], filter_facs[i + 1]
                row_in, col_in = (fac_im1 + fac_i + 1) * in_channels, (
                    fac_ip1 + fac_i + 1) * in_channels
                if i == 1:
                    row_in += n_in * filter_facs[0] * in_channels
                if i == self.L - 2:
                    col_in += n_out * filter_facs[-1] * in_channels
                self.add_module(f"layer_{i}_r",
                                nn.Conv1d(row_in, fac_i * out_channels, 1))
                self.add_module(f"layer_{i}_c",
                                nn.Conv1d(col_in, fac_i * out_channels, 1))
                set_init_(
                    getattr(self, f"layer_{i}"),
                    getattr(self, f"layer_{i}_rc"),
                    getattr(self, f"layer_{i}_r"),
                    getattr(self, f"layer_{i}_c"),
                )

                self.add_module(f"bias_{i}", nn.Conv1d(col_in, out_channels, 1))
                self.add_module(f"bias_{i}_rc",
                                nn.Linear(n_rc_inp * in_channels, out_channels))
                set_init_(getattr(self, f"bias_{i}"),
                          getattr(self, f"bias_{i}_rc"))
        self.rearr1_wt1 = Rearrange("b c_in nrow ncol -> b (c_in ncol) nrow")
        self.rearr1_wtL = Rearrange("b c_in n_out nrow -> b (c_in n_out) nrow")
        self.rearr1_outwt = Rearrange(
            "b (c_out ncol) nrow -> b c_out nrow ncol", ncol=n_in)
        self.rearrL_wtL = Rearrange("b c_in nrow ncol -> b (c_in nrow) ncol")
        self.rearrL_wt1 = Rearrange("b c_in ncol n_in -> b (c_in n_in) ncol")
        self.rearrL_outwt = Rearrange(
            "b (c_out nrow) ncol -> b c_out nrow ncol", nrow=n_out)
        self.rearrL_outbs = Rearrange("b (c_out nrow) -> b c_out nrow",
                                      nrow=n_out)
        self.rearr2_wt1 = Rearrange("b c ncol n_in -> b (c n_in) ncol")
        self.rearrLm1_wtL = Rearrange("b c n_out nrow -> b (c n_out) nrow")

    def forward(self, wsfeat: WeightSpaceFeatures):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        weights, biases = wsfeat.weights, wsfeat.biases
        angle = wsfeat.angle

        col_means = [w.mean(dim=-1) for w in weights
                    ]  # shapes: (B, C_in, nrow_i=ncol_i+1)
        row_means = [w.mean(dim=-2) for w in weights
                    ]  # shapes: (B, C_in, ncol_i=nrow_i-1)
        rc_means = [w.mean(dim=(-1, -2)) for w in weights]  # shapes: (B, C_in)
        bias_means = [b.mean(dim=-1) for b in biases]  # shapes: (B, C_in)
        rm0 = torch.flatten(row_means[0],
                            start_dim=-2)  # b c_in ncol -> b (c_in ncol)
        cmL = torch.flatten(col_means[-1],
                            start_dim=-2)  # b c_in nrowL -> b (c_in nrowL)
        final_bias = torch.flatten(biases[-1],
                                   start_dim=-2)  # b c_in nrow -> b (c_in nrow)
        # (B, C_in * (2 * L + n_in + 2 * n_out))
        rc_inp = torch.cat(rc_means + bias_means + [rm0, cmL, final_bias],
                           dim=-1)

        out_weights, out_biases = [], []
        for i in range(self.L):
            weight, bias, angle = wsfeat[i]
            if i == 0:
                rpt = [self.rearr1_wt1(weight), row_means[1], bias]
                if i + 1 == self.L - 1:
                    rpt.append(self.rearr1_wtL(weights[-1]))

                rpt = torch.cat(rpt, dim=-2)  # repeat ptwise across rows
                z1 = self.w0_rpt(rpt)
                z2 = self.w0_rbdcst(rc_inp)[..., None]  # (b, c_out * ncol, 1)
                z = z1 + z2

                z3 = self._angle_emb[i](angle)
                z3 = torch.unsqueeze(z3, -1)
                z3 = torch.squeeze(z3, -3)
                z = z + z3

                z = self.rearr1_outwt(z)
                u1 = self.bias_0(rpt)  # (B, C_out, nrow)
                u2 = self.bias_0_rc(rc_inp).unsqueeze(-1)  # (B, C_out, 1)
                u = u1 + u2

                u3 = self._angle_emb_bias[i](angle)
                u3 = torch.unsqueeze(u3, -1)
                u3 = torch.squeeze(u3, -3)
                u = u + u3

            elif i == self.L - 1:
                cpt = [self.rearrL_wtL(weight), col_means[-2]]  # b c_in ncol
                if i - 1 == 0:
                    cpt.append(self.rearrL_wt1(weights[0]))
                z1 = self.wfin_cpt(torch.cat(cpt,
                                             dim=-2))  # (B, C_out * nrow, ncol)
                z2 = self.wfin_cbdcst(rc_inp)[..., None]  # (b, c_out * nrow, 1)
                z = z1 + z2

                z3 = self._angle_emb[i](angle)
                z3 = torch.unsqueeze(z3, -1)
                z3 = torch.squeeze(z3, -3)
                z = z + z3

                z = self.rearrL_outwt(z)
                u = self.rearrL_outbs(self.bias_fin_rc(rc_inp))

                u3 = self._angle_emb_bias[i](angle)
                u3 = torch.unsqueeze(u3, -1)
                u3 = torch.squeeze(u3, -3)
                u = u + u3

            else:
                z1 = getattr(self,
                             f"layer_{i}")(weight)  # (B, C_out, nrow, ncol)
                z2 = getattr(self, f"layer_{i}_rc")(rc_inp)[..., None, None]
                row_bdcst = [row_means[i], col_means[i - 1], biases[i - 1]]
                col_bdcst = [col_means[i], bias, row_means[i + 1]]
                if i == 1:
                    row_bdcst.append(self.rearr2_wt1(weights[0]))
                if i == len(weights) - 2:
                    col_bdcst.append(self.rearrLm1_wtL(weights[-1]))
                row_bdcst = torch.cat(row_bdcst,
                                      dim=-2)  # (B, (3 + ?n_in) * C_in, ncol)
                col_bdcst = torch.cat(col_bdcst,
                                      dim=-2)  # (B, (3 + ?n_out) * C_in, nrow)
                z3 = getattr(self, f"layer_{i}_r")(row_bdcst).unsqueeze(
                    -2)  # (B, C_out, 1, ncol)
                z4 = getattr(self, f"layer_{i}_c")(col_bdcst).unsqueeze(
                    -1)  # (B, C_out, nrow, 1)
                z = z1 + z2 + z3 + z4

                z5 = self._angle_emb[i](angle)

                z5 = torch.reshape(z5, (z.shape[0], z.shape[1], z.shape[2], 1))
                z = z + z5

                u1 = getattr(self, f"bias_{i}")(col_bdcst)  # (B, C_out, nrow)
                u2 = getattr(self, f"bias_{i}_rc")(rc_inp).unsqueeze(
                    -1)  # (B, C_out, 1)
                u = u1 + u2

                u3 = self._angle_emb_bias[i](angle)
                u3 = torch.unsqueeze(u3, -1)
                u3 = torch.squeeze(u3, -3)
                u = u + u3

            out_weights.append(z)
            out_biases.append(u)
        return unshape_wsfeat_symmetry(
            WeightSpaceFeatures(out_weights, out_biases, angle),
            self.network_spec)

    def __repr__(self):
        return f"HNPLinear(in_channels={self.c_in}, out_channels={self.c_out}, n_in={self.n_in}, n_out={self.n_out}, num_layers={self.L})"
