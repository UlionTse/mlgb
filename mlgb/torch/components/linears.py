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
    BiGRUModeList,
)
from mlgb.torch.functions import (
    IdentityLayer,
    FlattenLayer,
    BatchNormalizationLayer,
    LayerNormalizationLayer,
    ActivationLayer,
    InitializerLayer,
)
from mlgb.error import MLGBError


__all__ = [
    'TaskLayer',
    'LinearLayer',
    'DeepNeuralNetworkLayer',
    'FeedForwardNetworkLayer',
    'Linear3dParallelLayer',
    'DNN3dParallelLayer',
    'Linear2dParallelLayer',
    'ConvolutionalNeuralNetworkLayer',
    'GatedRecurrentUnitLayer',
    'BiGatedRecurrentUnitLayer',
    'CapsuleNetworkLayer',
]


class LinearLayer(torch.nn.Module):
    # tf: default linear_initializer=None='glorot_uniform'; torch: default linear_initializer=None='he_uniform'.
    def __init__(self, linear_unit=1, linear_if_bias=True, linear_activation=None, device='cpu'):
        super().__init__()
        self.linear_unit = linear_unit
        self.linear_if_bias = linear_if_bias
        self.linear_activation = linear_activation
        if self.linear_activation:
            self.activation_fn = ActivationLayer(
                activation=linear_activation,
                device=device,
            )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 2:
                raise MLGBError

            f = x.shape[1]
            self.linear_fn = torch.nn.Linear(
                in_features=f,
                out_features=self.linear_unit if self.linear_unit else f,
                bias=self.linear_if_bias,
                device=self.device,
            )
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = self.linear_fn(x)
        if self.linear_activation:
            x = self.activation_fn(x)
        return x


class TaskLayer(torch.nn.Module):
    def __init__(self, task='multiclass:10', task_multiclass_if_project=False, task_multiclass_if_softmax=False,
                 task_multiclass_temperature_ratio=None,
                 task_linear_if_identity=False, task_linear_if_weighted=False, task_linear_if_bias=True,
                 device='cpu'):
        super().__init__()
        self.task = task
        self.task_name = task.split(':')[0]
        if self.task_name not in ('binary', 'regression', 'multiclass'):
            raise MLGBError
        if task_multiclass_temperature_ratio and not (1e-2 <= task_multiclass_temperature_ratio <= 1.0):
            raise MLGBError

        self.task_linear_if_identity = task_linear_if_identity
        self.task_linear_if_weighted = task_linear_if_weighted
        self.task_multiclass_if_project = task_multiclass_if_project
        self.task_multiclass_if_softmax = task_multiclass_if_softmax
        self.task_multiclass_temperature_ratio = task_multiclass_temperature_ratio
        self.task_activation_dict = {
            'regression': None,
            'binary': 'sigmoid',
            'multiclass': 'softmax',
        }
        self.activation_fn = ActivationLayer(
            activation=self.task_activation_dict[self.task_name],
            device=device,
        )
        self.linear_fn = LinearLayer(
            linear_unit=1 if self.task_name != 'multiclass' else int(task.split(':')[1]),
            linear_if_bias=task_linear_if_bias,
            linear_activation=None,
            device=device,
        )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 2:
                raise MLGBError
            if self.task_name == 'multiclass' and not self.task_multiclass_if_project:
                if int(self.task.split(':')[1]) != x.shape[1]:
                    raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        if self.task_name != 'multiclass':
            if not self.task_linear_if_identity:
                if self.task_linear_if_weighted:
                    x = self.linear_fn(x)
                else:
                    x = torch.sum(x, dim=1, keepdim=True)  # only apply after product of matching model.
            x = self.activation_fn(x)
        else:
            if self.task_multiclass_if_project:
                x = self.linear_fn(x)

            if self.task_multiclass_temperature_ratio:
                x /= self.task_multiclass_temperature_ratio  # zoom data for better performance in softmax_loss.

            if self.task_multiclass_if_softmax:
                x = torch.softmax(x, dim=1)
        return x


class DeepNeuralNetworkLayer(torch.nn.Module):
    def __init__(self, dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_if_bias=True,
                 device='cpu'):
        super().__init__()
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_length = len(self.dnn_hidden_units)
        self.dnn_dropouts = [dnn_dropout] * self.dnn_length if isinstance(dnn_dropout, float) else dnn_dropout
        self.dnn_bns = [dnn_if_bn] * self.dnn_length if isinstance(dnn_if_bn, bool) else dnn_if_bn
        self.dnn_lns = [dnn_if_ln] * self.dnn_length if isinstance(dnn_if_ln, bool) else dnn_if_ln
        self.dnn_bias = [dnn_if_bias] * self.dnn_length if isinstance(dnn_if_bias, bool) else dnn_if_bias
        self.dnn_fn_list = torch.nn.ModuleList([
            torch.nn.ModuleList([
                LinearLayer(
                    linear_unit=self.dnn_hidden_units[i],
                    linear_if_bias=self.dnn_bias[i],
                    linear_activation=None,
                    device=device,
                ),
                BatchNormalizationLayer(axis=1, device=device) if self.dnn_bns[i] else IdentityLayer(device=device),
                LayerNormalizationLayer(axis=1, device=device) if self.dnn_lns[i] else IdentityLayer(device=device),
                torch.nn.Dropout(p=self.dnn_dropouts[i]),
                ActivationLayer(activation=dnn_activation, device=device),
            ])
            for i in range(self.dnn_length)
        ])
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 2:
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        for dnn_fn in self.dnn_fn_list:
            for fn in dnn_fn:
                x = fn(x)

        return x


class FeedForwardNetworkLayer(torch.nn.Module):
    def __init__(self, ffn_linear_if_twice=True, ffn_if_bias=True, ffn_activation='relu',
                 ffn_dropout=0.0, ffn_if_bn=False, ffn_if_ln=False, ffn_initializer=None,
                 ffn_last_factor=None, ffn_last_activation=None,
                 device='cpu'):
        super().__init__()
        self.ffn_linear_if_twice = ffn_linear_if_twice
        self.ffn_if_bias = ffn_if_bias
        self.ffn_last_factor = ffn_last_factor
        self.ffn_last_activation = ffn_last_activation

        self.bn_fn = BatchNormalizationLayer(axis=1, device=device) if ffn_if_bn else IdentityLayer(device=device)
        self.ln_fn = LayerNormalizationLayer(axis=1, device=device) if ffn_if_ln else IdentityLayer(device=device)
        self.drop_fn = torch.nn.Dropout(p=ffn_dropout)
        self.activation_fn = ActivationLayer(
            activation=ffn_activation,
            device=device,
        )
        if self.ffn_linear_if_twice and self.ffn_last_activation:
            self.activation_last_fn = ActivationLayer(
                activation=ffn_last_activation,
                device=device,
            )
        self.initializer_fn = InitializerLayer(
            initializer=ffn_initializer,
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
            if x.ndim != 3:
                raise MLGBError

            _, f, e = x.shape
            self.ffn_weight_list = [
                self.initializer_fn(torch.nn.parameter.Parameter(
                    data=torch.empty(
                        size=[f, e],
                        device=self.device,
                    ),
                    requires_grad=True,
                ))
                for _ in range(2 if self.ffn_linear_if_twice else 1)
            ]
            if self.ffn_if_bias:
                self.ffn_bias_list = [
                    self.initializer_zeros_fn(torch.nn.parameter.Parameter(
                        data=torch.empty(
                            size=[f, 1],
                            device=self.device,
                        ),
                        requires_grad=True,
                    ))
                    for _ in range(2 if self.ffn_linear_if_twice else 1)
                ]
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = x * self.ffn_weight_list[0] + (self.ffn_bias_list[0] if self.ffn_if_bias else 0.0)
        x = self.bn_fn(x)
        x = self.ln_fn(x)
        x = self.drop_fn(x)
        x = self.activation_fn(x)

        if self.ffn_linear_if_twice:
            x = x * self.ffn_weight_list[1] + (self.ffn_bias_list[1] if self.ffn_if_bias else 0.0)
            if self.ffn_last_activation:
                x = self.activation_last_fn(x)
            if self.ffn_last_factor:
                x = x * self.ffn_last_factor
        return x


class Linear3dParallelLayer(torch.nn.Module):  # vs Dense or EinsumDense with n-d
    def __init__(self, linear_unit=1, linear_activation=None, linear_dropout=0.0, 
                 linear_if_bn=False, linear_if_ln=False, linear_initializer=None, linear_if_bias=True, 
                 device='cpu'):
        super().__init__()
        self.linear_unit = linear_unit
        self.linear_if_bias = linear_if_bias

        self.bn_fn = BatchNormalizationLayer(axis=1, device=device) if linear_if_bn else IdentityLayer(device=device)
        self.ln_fn = LayerNormalizationLayer(axis=1, device=device) if linear_if_ln else IdentityLayer(device=device)
        self.drop_fn = torch.nn.Dropout(p=linear_dropout)
        self.activation_fn = ActivationLayer(activation=linear_activation, device=device)
        self.initializer_fn = InitializerLayer(
            initializer=linear_initializer,
            activation=linear_activation,
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
            if x.ndim != 3:
                raise MLGBError

            _, self.linear_parallel_num, self.inputs_width = x.shape

            self.linear_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.linear_parallel_num, self.inputs_width, self.linear_unit],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.linear_bias = self.initializer_zeros_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.linear_parallel_num, self.linear_unit],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = torch.einsum('ijk,jkl->ijl', x, self.linear_weight) + (self.linear_bias if self.linear_if_bias else 0.0)
        x = self.bn_fn(x)
        x = self.ln_fn(x)
        x = self.drop_fn(x)
        x = self.activation_fn(x)  # (batch_size, linear_parallel_num, linear_unit)
        return x


class DNN3dParallelLayer(torch.nn.Module):
    def __init__(self, dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_initializer=None, dnn_if_bias=True, dnn_if_output2d=False, 
                 device='cpu'):
        super().__init__()
        self.dnn_if_output2d = dnn_if_output2d
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_length = len(self.dnn_hidden_units)
        self.dnn_dropouts = [dnn_dropout] * self.dnn_length if isinstance(dnn_dropout, float) else dnn_dropout
        self.dnn_bns = [dnn_if_bn] * self.dnn_length if isinstance(dnn_if_bn, bool) else dnn_if_bn
        self.dnn_lns = [dnn_if_ln] * self.dnn_length if isinstance(dnn_if_ln, bool) else dnn_if_ln
        self.dnn_bias = [dnn_if_bias] * self.dnn_length if isinstance(dnn_if_bias, bool) else dnn_if_bias

        self.dnn_fn_list = torch.nn.ModuleList([
            Linear3dParallelLayer(
                linear_activation=dnn_activation,
                linear_if_bias=self.dnn_bias[i],
                linear_unit=self.dnn_hidden_units[i],
                linear_dropout=self.dnn_dropouts[i],
                linear_if_bn=self.dnn_bns[i],
                linear_if_ln=self.dnn_lns[i],
                linear_initializer=dnn_initializer,
                device=device,
            ) for i in range(self.dnn_length)
        ])
        self.flatten_fn = FlattenLayer(device=device)
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

        for dnn_fn in self.dnn_fn_list:
            x = dnn_fn(x)

        if self.dnn_if_output2d:
            x = self.flatten_fn(x)
        return x


class Linear2dParallelLayer(torch.nn.Module):
    def __init__(self, linear_parallel_num=1, linear_activation=None, linear_if_bias=True, linear_initializer=None,
                 device='cpu'):
        super().__init__()
        if linear_parallel_num < 1:
            raise MLGBError

        self.linear_parallel_num = linear_parallel_num
        self.linear_unit = 1

        self.linear3d_parallel_fn = Linear3dParallelLayer(
            linear_unit=self.linear_unit,
            linear_activation=linear_activation,
            linear_if_bias=linear_if_bias,
            linear_dropout=0.0,
            linear_if_bn=False,
            linear_if_ln=False,
            linear_initializer=linear_initializer,
            device=device,
        )
        self.flatten_fn = FlattenLayer()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 2:
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = torch.stack([x] * self.linear_parallel_num, dim=1)  # (batch_size, linear_parallel_num, inputs_width)
        x = self.linear3d_parallel_fn(x)  # (batch_size, linear_parallel_num, linear_unit)
        x = self.flatten_fn(x)
        return x


class ConvolutionLayer(torch.nn.Module):
    def __init__(self, cnn_conv_mode='Conv1D', cnn_filter_num=64, cnn_kernel_height=32, cnn_kernel_width=1,
                 cnn_activation='tanh', cnn_if_max_pool=True, cnn_pool_size=2, device='cpu'):
        super().__init__()
        if cnn_conv_mode not in ('Conv1D', 'Conv2D'):
            raise MLGBError

        self.cnn_conv_mode = cnn_conv_mode
        self.cnn_if_max_pool = cnn_if_max_pool
        self.cnn_filter_num = cnn_filter_num
        self.cnn_kernel_height = cnn_kernel_height
        self.cnn_kernel_width = cnn_kernel_width
        self.cnn_pool_size = cnn_pool_size
        self.activation_fn = ActivationLayer(
            activation=cnn_activation,
            device=device,
        )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim not in (3, 4):
                raise MLGBError
            if self.cnn_conv_mode == 'Conv1D' and x.ndim != 3:
                raise MLGBError
            if self.cnn_conv_mode == 'Conv2D' and x.ndim != 4:
                raise MLGBError

            if self.cnn_conv_mode == 'Conv1D':
                self.conv_fn = torch.nn.Conv1d(
                    in_channels=x.shape[1],  # torch: data_format='channels_first'.
                    out_channels=self.cnn_filter_num,
                    kernel_size=(self.cnn_kernel_height,),
                    stride=(1,),
                    bias=True,
                    padding='same',
                    padding_mode='zeros',
                    device=self.device,
                )  # (b, e, f) -> (b, cnn_filter_num, f)
                if self.cnn_if_max_pool:
                    self.max_pool_fn = torch.nn.MaxPool1d(
                        kernel_size=(self.cnn_pool_size,),
                        stride=(self.cnn_pool_size,),
                        padding=0,
                    )  # (b, cnn_filter_num, f) -> (b, cnn_filter_num, f // cnn_pool_size)
            else:
                self.conv_fn = torch.nn.Conv2d(
                    in_channels=x.shape[1],  # torch: data_format='channels_first'.
                    out_channels=self.cnn_filter_num,
                    kernel_size=(self.cnn_kernel_height, self.cnn_kernel_width),
                    stride=(1, 1),
                    bias=True,
                    padding='same',
                    padding_mode='zeros',
                    device=self.device,
                )  # (b, 1, f, e) -> (b, cnn_filter_num, f, e)
                if self.cnn_if_max_pool:
                    self.max_pool_fn = torch.nn.MaxPool2d(
                        kernel_size=(self.cnn_pool_size, 1),
                        stride=(self.cnn_pool_size, 1),
                        padding=0,
                    )  # (b, cnn_filter_num, f, e) -> (b, cnn_filter_num, f // cnn_pool_size, e)
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = self.conv_fn(x)
        x = self.activation_fn(x)
        x = self.max_pool_fn(x) if self.cnn_if_max_pool else x
        return x


class ConvolutionalNeuralNetworkLayer(torch.nn.Module):
    def __init__(self, cnn_conv_mode='Conv1D', cnn_filter_nums=(64, 32), cnn_kernel_heights=(64, 32), cnn_kernel_widths=1,
                 cnn_activation='tanh', cnn_if_max_pool=True, cnn_pool_sizes=2, device='cpu'):
        super().__init__()
        if cnn_conv_mode not in ('Conv1D', 'Conv2D'):
            raise MLGBError

        self.cnn_conv_mode = cnn_conv_mode
        self.cnn_length = len(cnn_filter_nums)
        self.cnn_kernel_heights = [cnn_kernel_heights] * self.cnn_length if isinstance(cnn_kernel_heights, int) else cnn_kernel_heights
        self.cnn_kernel_widths = [cnn_kernel_widths] * self.cnn_length if isinstance(cnn_kernel_widths, int) else cnn_kernel_widths
        self.cnn_pool_sizes = [cnn_pool_sizes] * self.cnn_length if isinstance(cnn_pool_sizes, int) else cnn_pool_sizes

        self.cnn_fn_list = torch.nn.ModuleList([
            ConvolutionLayer(
                cnn_conv_mode=cnn_conv_mode,
                cnn_filter_num=cnn_filter_nums[i],
                cnn_kernel_height=self.cnn_kernel_heights[i],
                cnn_kernel_width=self.cnn_kernel_widths[i],
                cnn_activation=cnn_activation,
                cnn_if_max_pool=cnn_if_max_pool,
                cnn_pool_size=self.cnn_pool_sizes[i],
                device=device,
            )
            for i in range(self.cnn_length)
        ])
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim not in (3, 4):
                raise MLGBError
            if self.cnn_conv_mode == 'Conv1D' and x.ndim != 3:
                raise MLGBError
            if self.cnn_conv_mode == 'Conv2D' and x.ndim != 4:
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        for cnn_fn in self.cnn_fn_list:
            x = cnn_fn(x)
        return x


class BaseGatedRecurrentUnitLayer(torch.nn.Module):
    def __init__(self, gru_hidden_unit=32, gru_dropout=0.0, gru_if_bias=True, device='cpu'):
        super().__init__()
        self.gru_hidden_unit = gru_hidden_unit
        self.gru_dropout = gru_dropout
        self.gru_if_bias = gru_if_bias
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            self.base_gru_fn = torch.nn.GRU(
                batch_first=True,
                input_size=x.shape[2],  # embed_dim
                hidden_size=self.gru_hidden_unit,
                dropout=self.gru_dropout,
                bias=self.gru_if_bias,
                bidirectional=False,
                num_layers=1,
                device=self.device,
            )
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = self.base_gru_fn(x)[0]
        return x


class GatedRecurrentUnitLayer(torch.nn.Module):
    def __init__(self, gru_hidden_units=(64, 32), gru_dropout=0.0, gru_if_bias=True, device='cpu'):
        super().__init__()
        self.gru_length = len(gru_hidden_units)
        self.gru_fn_list = torch.nn.ModuleList([
            BaseGatedRecurrentUnitLayer(
                gru_hidden_unit=gru_hidden_units[i],
                gru_dropout=gru_dropout,
                gru_if_bias=gru_if_bias,
                device=device,
            )
            for i in range(self.gru_length)
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

        for gru_fn in self.gru_fn_list:
            x = gru_fn(x)

        x = x[:, -1, :]
        return x


class BiGatedRecurrentUnitLayer(torch.nn.Module):
    def __init__(self, gru_bi_mode='Frontward', gru_hidden_units=(64, 32), gru_dropout=0.0, gru_if_bias=True, device='cpu'):
        super().__init__()
        if gru_bi_mode not in BiGRUModeList:
            raise MLGBError

        self.gru_bi_mode = gru_bi_mode

        self.gru_fn_list = torch.nn.ModuleList([
            GatedRecurrentUnitLayer(
                gru_hidden_units=gru_hidden_units,
                gru_dropout=gru_dropout,
                gru_if_bias=gru_if_bias,
                device=device,
            )
            for _ in range(1 if self.gru_bi_mode in ('Frontward', 'Backward') else 2)
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
        x_reverse = torch.flip(x, dims=[1])  # x[:, ::-1, :]

        if self.gru_bi_mode == 'Frontward':
            x = self.gru_fn_list[0](x)
        elif self.gru_bi_mode == 'Backward':
            x = self.gru_fn_list[0](x[:, ::-1, :])
        elif self.gru_bi_mode == 'Frontward+Backward':
            x = self.gru_fn_list[0](x) + self.gru_fn_list[1](x_reverse)
        elif self.gru_bi_mode == 'Frontward-Backward':
            x = self.gru_fn_list[0](x) - self.gru_fn_list[1](x_reverse)
        elif self.gru_bi_mode == 'Frontward*Backward':
            x = self.gru_fn_list[0](x) * self.gru_fn_list[1](x_reverse)
        elif self.gru_bi_mode == 'Frontward,Backward':
            x = torch.stack([self.gru_fn_list[0](x), self.gru_fn_list[1](x_reverse)], dim=1)
        else:
            raise MLGBError

        x = x if self.gru_bi_mode == 'Frontward,Backward' else torch.unsqueeze(x, dim=1)
        return x


class CapsuleNetworkLayer(torch.nn.Module):
    def __init__(self, capsule_num=3, capsule_activation='squash', capsule_l2=0.0, capsule_initializer=None,
                 capsule_interest_num_if_dynamic=False, capsule_input_sequence_pad_mode='pre',
                 capsule_routing_initializer='random_normal', device='cpu'):
        super().__init__()
        if capsule_input_sequence_pad_mode not in ('pre', 'post'):
            raise MLGBError

        self.capsule_num = capsule_num
        self.capsule_l2 = capsule_l2
        self.capsule_interest_num_if_dynamic = capsule_interest_num_if_dynamic
        self.capsule_input_sequence_pad_mode = capsule_input_sequence_pad_mode
        self.capsule_initializer_fn = InitializerLayer(
            initializer=capsule_initializer,
            activation=capsule_activation,
        ).get()
        self.capsule_routing_initializer_fn = InitializerLayer(
            initializer=capsule_routing_initializer,
            activation=None,
        ).get()
        self.activation_fn = ActivationLayer(
            activation=capsule_activation,
            device=device,
        )
        self.mask_fn = MaskLayer(
            att_if_mask=True,
            device=device,
        )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            _, seq_len, embed_dim = x.shape
            if self.capsule_interest_num_if_dynamic:
                seq_len = self.get_dynamic_interest_num(seq_len, embed_dim)
                self.seq_len = seq_len

            self.capsule_bilinear_weight = self.capsule_initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[seq_len, embed_dim, embed_dim],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.capsule_routing_weight = self.capsule_routing_initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[1, seq_len, embed_dim],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        if self.capsule_interest_num_if_dynamic:
            x = x[:, -self.seq_len:, :] if self.capsule_input_sequence_pad_mode == 'pre' else x[:, :self.seq_len, :]

        w = self.capsule_routing_weight
        w_b = self.capsule_bilinear_weight
        for i in range(self.capsule_num):
            w = self.mask_fn(w)
            w = torch.softmax(w, dim=1)
            x_h = torch.einsum('bfe,fee->bfe', x, w_b) * w  # high_level_capsule: x_h = w * w_b * x
            x_h = self.activation_fn(x_h)  # squash

            w_i = torch.einsum('bfe,fee->bfe', x_h, w_b) * x  # routing_logit: w_i = x_h * w_b * x
            w_i = torch.reduce_sum(w_i, axis=0, keepdims=True)
            w = w + w_i  # if `w` isn't be updated(not bp), it's like `w` of RNN.
            x = x_h
        return x

    def get_dynamic_interest_num(self, seq_len, embed_dim):
        seq_k = max(1, min(seq_len, int(numpy.log2(embed_dim))))
        return seq_k















































