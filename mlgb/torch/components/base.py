# coding=utf-8
# author=uliontse

"""
Copyright 2020 UlionTse

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from mlgb.torch.configs import (
    numpy,
    torch,
    PoolModeList,
    EDCNModeList,
)
from mlgb.torch.functions import (
    FlattenLayer,
    FlattenAxesLayer,
    BatchNormalizationLayer,
    LayerNormalizationLayer,
    ActivationLayer,
    InitializerLayer,
    SimplePoolingLayer,
)
from mlgb.torch.components.linears import (
    DeepNeuralNetworkLayer,
    FeedForwardNetworkLayer,
    DNN3dParallelLayer,
    ConvolutionalNeuralNetworkLayer,
)
from mlgb.error import MLGBError


__all__ = [
    'MaskBlockLayer',
    'ResidualUnitLayer',
    'CrossNetworkLayer',
    'RegulationModuleLayer',
    'BridgeModuleLayer',
    'FactorEstimatingNetworkLayer',
    'LogarithmicTransformationLayer',
    'CompressedInteractionNetworkLayer',
    'SqueezeExcitationNetworkLayer',
    'LocalActivationUnitLayer',
]


class ResidualUnitLayer(torch.nn.Module):
    def __init__(self, dcm_if_dnn=True, dcm_if_ln=False,
                 dnn_hidden_unit=32, dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
        super().__init__()
        self.dcm_if_dnn = dcm_if_dnn
        self.dcm_if_ln = dcm_if_ln
        self.dnn_activation = dnn_activation
        self.dnn_dropout = dnn_dropout
        self.dnn_if_bn = dnn_if_bn
        self.dnn_if_ln = dnn_if_ln

        if self.dcm_if_dnn:
            self.dnn_fn = DeepNeuralNetworkLayer(
                dnn_hidden_units=[dnn_hidden_unit],
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                device=device,
            )
        if self.dcm_if_ln:
            self.ln_fn = LayerNormalizationLayer(axis=1, device=device)
        self.activation_fn = ActivationLayer(
            activation=self.dnn_activation,
            device=device,
        )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 2:
                raise MLGBError
    
            self.stack_fn = DeepNeuralNetworkLayer(
                dnn_hidden_units=[x.shape[-1]],
                dnn_activation=None,
                dnn_dropout=self.dnn_dropout,
                dnn_if_bn=self.dnn_if_bn,
                dnn_if_ln=self.dnn_if_ln,
                device=self.device,
            )
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x_dnn = self.dnn_fn(x) if self.dcm_if_dnn else x
        x_stack = self.stack_fn(x_dnn)

        x = torch.add(x, x_stack)  # x_stack = x_true - x_pred
        x = self.ln_fn(x) if self.dcm_if_ln else x
        x = self.activation_fn(x)
        return x


class CrossNetworkLayer(torch.nn.Module):
    def __init__(self, dcn_version='v2', dcn_layer_num=1, dcn_initializer=None, device='cpu'):
        super().__init__()
        if dcn_version not in ('v1', 'v2'):
            raise MLGBError

        self.dcn_version = dcn_version
        self.dcn_layer_num = dcn_layer_num
        self.dcn_unit = 1  # dcn_weight__length_of_latent_vector
        self.initializer_fn = InitializerLayer(
            initializer=dcn_initializer,
            activation=None,
        ).get()
        self.initializer_zeros_fn = InitializerLayer(
            initializer='zeros',
            activation=None,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 2:
                raise MLGBError

            self.embed_dim = x.shape[-1]
            self.dcn_weight_shap_map = {
                'v1': [self.embed_dim, self.dcn_unit],
                'v2': [self.embed_dim, self.embed_dim],
            }
            self.dcn_weight_list = [
                self.initializer_fn(torch.nn.parameter.Parameter(
                    data=torch.empty(
                        size=self.dcn_weight_shap_map[self.dcn_version],
                        device=self.device,
                    ),
                    requires_grad=True,
                ))
                for _ in range(self.dcn_layer_num)
            ]
            self.dcn_bias_list = [
                self.initializer_zeros_fn(torch.nn.parameter.Parameter(
                    data=torch.empty(
                        size=[self.dcn_unit],
                        device=self.device,
                    ),
                    requires_grad=True,
                ))
                for _ in range(self.dcn_layer_num)
            ]

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x_i = x_0 = x
        for i in range(self.dcn_layer_num):
            if self.dcn_version == 'v2':
                x_i = x_0 * (x_i @ self.dcn_weight_list[i] + self.dcn_bias_list[i]) + x_i
            else:
                x_c = torch.einsum('ijk,ik->ij', torch.unsqueeze(x_0, dim=-1), x_i @ self.dcn_weight_list[i])
                x_i = x_c + self.dcn_bias_list[i] + x_i
        return x_i


class RegulationModuleLayer(torch.nn.Module):
    def __init__(self, fgu_tau_ratio=1.0, fgu_initializer='ones', device='cpu'):
        super().__init__()
        if not (0 < fgu_tau_ratio <= 1.0):
            raise MLGBError  # field-wise gating unit

        self.fgu_tau_ratio = fgu_tau_ratio
        self.initializer_fn = InitializerLayer(
            initializer=fgu_initializer,
            activation=None,
        ).get()
        self.flatten_fn = FlattenLayer()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            _, self.fields_width, self.embed_dim = x.shape
            self.fgu_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[1, self.fields_width, 1],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        fgu_w = self.fgu_weight * self.fgu_tau_ratio
        fgu_score = torch.softmax(fgu_w, dim=1)

        fgu_outputs = x * fgu_score
        fgu_outputs = self.flatten_fn(fgu_outputs)
        return fgu_outputs


class BridgeModuleLayer(torch.nn.Module):
    def __init__(self, bdg_mode='EDCN:attention_pooling', bdg_layer_num=1,
                 dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
        super().__init__()
        if bdg_mode not in EDCNModeList:
            raise MLGBError

        self.bdg_mode = bdg_mode
        self.bdg_layer_num = bdg_layer_num
        self.dnn_activation = dnn_activation
        self.dnn_dropout = dnn_dropout
        self.dnn_if_bn = dnn_if_bn
        self.dnn_if_ln = dnn_if_ln
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 2:
                raise MLGBError
            if x[0].ndim != 2:
                raise MLGBError
            if x[0].shape != x[1].shape:
                raise MLGBError

            _, self.inputs_width = x[0].shape
            bdg_dnn_hidden_units = [self.inputs_width] * self.bdg_layer_num

            if self.bdg_mode in ('EDCN:attention_pooling', 'EDCN:concatenation'):
                self.dnn_fn = DeepNeuralNetworkLayer(
                    dnn_hidden_units=bdg_dnn_hidden_units,
                    dnn_activation=self.dnn_activation,
                    dnn_dropout=self.dnn_dropout,
                    dnn_if_bn=self.dnn_if_bn,
                    dnn_if_ln=self.dnn_if_ln,
                    device=self.device,
                )
            if self.bdg_mode == 'EDCN:attention_pooling':
                self.dnn2_fn = DeepNeuralNetworkLayer(
                    dnn_hidden_units=bdg_dnn_hidden_units,
                    dnn_activation=self.dnn_activation,
                    dnn_dropout=self.dnn_dropout,
                    dnn_if_bn=self.dnn_if_bn,
                    dnn_if_ln=self.dnn_if_ln,
                    device=self.device,
                )
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_c, x_d = x  # x_cross, x_deep

        if self.bdg_mode == 'EDCN:pointwise_addition':
            x = x_c + x_d
        elif self.bdg_mode == 'EDCN:hadamard_product':
            x = x_c * x_d
        elif self.bdg_mode == 'EDCN:concatenation':
            x = torch.concat([x_c, x_d], dim=-1)
            x = self.dnn_fn(x)
        elif self.bdg_mode == 'EDCN:attention_pooling':
            att_c = torch.log_softmax(self.dnn_fn(x_c), dim=1)  # dead softmax.
            att_d = torch.log_softmax(self.dnn2_fn(x_d), dim=1)
            x = x_c * att_c + x_d * att_d
        else:
            raise MLGBError
        return x


class CompressedInteractionNetworkLayer(torch.nn.Module):
    def __init__(self, cin_interaction_num=4, cin_interaction_ratio=1.0,
                 cnn_filter_num=64, cnn_kernel_size=64, cnn_activation='relu',
                 device='cpu'):
        super().__init__()
        if not (0.5 <= cin_interaction_ratio <= 1.0):
            raise MLGBError('0.5 <= cin_interaction_ratio <= 1.0')

        self.cin_interaction_num = cin_interaction_num
        self.cin_interaction_ratio = cin_interaction_ratio
        self.cnn_fn_list = torch.nn.ModuleList([
            ConvolutionalNeuralNetworkLayer(
                cnn_conv_mode='Conv1D',
                cnn_filter_nums=[cnn_filter_num],
                cnn_kernel_heights=[cnn_kernel_size],
                cnn_activation=cnn_activation,
                cnn_if_max_pool=False,
                device=device
            )
            for _ in range(cin_interaction_num)
        ])
        self.flatten_axes_fn_list = torch.nn.ModuleList([
            FlattenAxesLayer(axes=[1, 2], device=device)
            for _ in range(cin_interaction_num)
        ])
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x_0 = x_i = x
        x_list = []  # feature_map_list
        for i in range(self.cin_interaction_num):
            x_i = torch.einsum('bie,bje->bije', x_0, x_i)  # (b, f_0, f_i, e)  # torch: channels_first
            x_i = self.flatten_axes_fn_list[i](x_i)  # (b, f_0 * f_i, e)
            x_i = self.cnn_fn_list[i](x_i)  # (b, cnn_filter_num, e)

            x_list.append(x_i)
            x_i = self.get_center_x(x_i, self.cin_interaction_ratio)

        x = torch.concat(x_list, dim=1)
        x = torch.sum(x, dim=2, keepdim=False)  # (b, cin_interaction_num * cnn_filter_num)
        return x

    def get_center_x(self, x, center_ratio):
        f = x.shape[1]
        boundary_i = int(f * (1 - center_ratio) // 2)
        boundary_j = int(f - boundary_i)
        return x[:, boundary_i:boundary_j, :]


class SqueezeExcitationNetworkLayer(torch.nn.Module):
    def __init__(self, sen_pool_mode='Pooling:average', sen_reduction_factor=2, sen_activation='relu', sen_initializer=None,
                 device='cpu'):
        super().__init__()
        if sen_pool_mode not in PoolModeList:
            raise MLGBError
        if not (isinstance(sen_reduction_factor, int) and sen_reduction_factor >= 1):
            raise MLGBError('sen_reduction_factor')

        self.sen_pool_mode = sen_pool_mode
        self.sen_reduction_factor = sen_reduction_factor
        self.activation_fn = ActivationLayer(
            activation=sen_activation,
            device=device,
        )
        self.initializer_fn = InitializerLayer(
            initializer=sen_initializer,
            activation=sen_activation,
        ).get()
        self.pooling_fn = SimplePoolingLayer(
            pool_mode=sen_pool_mode,
            pool_axis=2,
            device=device,
        )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            _, self.fields_width, self.embed_dim = x.shape
            self.fields_unit = max(1, self.fields_width // self.sen_reduction_factor)

            self.sen_w1 = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.fields_width, self.fields_unit],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.sen_w2 = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.fields_unit, self.fields_width],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x_z = self.pooling_fn(x)  # (batch_size, fields_width)
        x_z = self.activation_fn(x_z @ self.sen_w1)
        x_z = self.activation_fn(x_z @ self.sen_w2)  # (batch_size, fields_width)
        x_z = torch.unsqueeze(x_z, dim=2)
        x = x * x_z  # re_weight == attention, (batch_size, fields_width, embed_dim), dim=3.
        return x


class LocalActivationUnitLayer(torch.nn.Module):
    def __init__(self, lau_version='v4', lau_hidden_units=(16,), lau_activation='dice', device='cpu'):
        super().__init__()
        if lau_version not in ('v1', 'v2', 'v3', 'v4'):
            raise MLGBError
        if not len(lau_hidden_units) > 0:
            raise MLGBError('lau_hidden_units')

        self.lau_version = lau_version
        self.dnn_parallel_fn = DNN3dParallelLayer(
            dnn_hidden_units=lau_hidden_units + (1,),
            dnn_activation=lau_activation,
            dnn_if_output2d=False,
            device=device,
        )
        self.flatten_axes_fn = FlattenAxesLayer(axes=[2, 3], device=device)
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 2:
                raise MLGBError
            if not (x[0].ndim == x[1].ndim == 3):
                raise MLGBError
            if x[0].shape != x[1].shape:
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_q, x_k = x

        if self.lau_version in ('v3', 'v4'):
            x_op = torch.einsum('bfi,bfj->bfij', x_q, x_k)
            x_op = self.flatten_axes_fn(x_op)  # (b, f, e*e)
            x = torch.concat([x_op, x_q, x_k], dim=2)  # (b, f, (e+2)*e)
        else:
            x_ip = x_q * x_k
            x_sub = x_q - x_k
            x = torch.concat([x_ip, x_sub, x_q, x_k], dim=2)  # (b, f, 4*e)

        x = self.dnn_parallel_fn(x)  # (b, f, 1)
        return x


class FactorEstimatingNetworkLayer(torch.nn.Module):
    def __init__(self, ifm_mode_if_dual=False,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
        super().__init__()
        self.ifm_mode_if_dual = ifm_mode_if_dual
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_length = len(self.dnn_hidden_units)
        self.dnn_activation = dnn_activation
        self.dnn_dropout = dnn_dropout
        self.dnn_if_bn = dnn_if_bn
        self.dnn_if_ln = dnn_if_ln
        self.flatten_fn = FlattenLayer(device=device)
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            self.fields_width = x.shape[1]

            self.dnn_fn = DeepNeuralNetworkLayer(
                dnn_hidden_units=self.dnn_hidden_units + (self.fields_width,),
                dnn_if_bias=[True] * self.dnn_length + [False],
                dnn_activation=self.dnn_activation,
                dnn_dropout=self.dnn_dropout,
                dnn_if_bn=self.dnn_if_bn,
                dnn_if_ln=self.dnn_if_ln,
                device=self.device,
            )
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = self.flatten_fn(x)  # (b, f*e)
        x = self.dnn_fn(x)

        if not self.ifm_mode_if_dual:
            x = torch.softmax(x, dim=1) * self.fields_width  # (b, f)

        x = torch.unsqueeze(x, dim=2)  # (b, f, 1)
        return x


class LogarithmicTransformationLayer(torch.nn.Module):
    def __init__(self, ltl_clip_min=1e-4, ltl_unit=32, ltl_initializer=None, device='cpu'):
        super().__init__()
        self.ltl_clip_min = ltl_clip_min
        self.ltl_unit = ltl_unit
        self.initializer_fn = InitializerLayer(
            initializer=ltl_initializer,
            activation=None,
        ).get()
        self.initializer_zeros_fn = InitializerLayer(
            initializer='zeros',
            activation=None,
        ).get()
        self.flatten_fn = FlattenLayer(device=device)
        self.bn_fn_list = torch.nn.ModuleList([
            BatchNormalizationLayer(axis=1, device=device)
            for _ in range(2)
        ])
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            self.fields_width = x.shape[1]

            self.ltl_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.fields_width, self.ltl_unit],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.ltl_bias = self.initializer_zeros_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.ltl_unit],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = torch.abs(x)
        x = torch.clip(x, min=self.ltl_clip_min, max=numpy.inf)
        x = self.bn_fn_list[0](x)  # (b, f, e)
        x = torch.einsum('bfe,fu->beu', x, self.ltl_weight) + self.ltl_bias
        x = torch.exp(x)
        x = self.bn_fn_list[1](x)
        x = self.flatten_fn(x)  # (b, u*e)
        return x


class MaskBlockLayer(torch.nn.Module):
    def __init__(self, ffn_activation='relu', ffn_if_bn=False, ffn_dropout=0.0, ffn_initializer=None, device='cpu'):
        super().__init__()
        self.igm_fn = FeedForwardNetworkLayer(
            ffn_linear_if_twice=True,
            ffn_if_bias=True,
            ffn_activation=ffn_activation,
            ffn_dropout=ffn_dropout,
            ffn_if_bn=ffn_if_bn,
            ffn_if_ln=False,
            ffn_initializer=ffn_initializer,
            device=device,
        )
        self.ln_hid_fn = FeedForwardNetworkLayer(
            ffn_linear_if_twice=False,
            ffn_if_bias=False,
            ffn_activation=ffn_activation,
            ffn_dropout=ffn_dropout,
            ffn_if_bn=ffn_if_bn,
            ffn_if_ln=True,  #
            ffn_initializer=ffn_initializer,
            device=device,
        )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 2:
                raise MLGBError
            if x[0].ndim != 3:
                raise MLGBError
            if x[0].shape != x[1].shape:
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x1, x2 = x

        x2 = self.igm_fn(x2)
        x = x1 * x2
        x = self.ln_hid_fn(x)
        return x












