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

from mlgb.tf.configs import (
    numpy,
    tf,
    Dense,
    L1L2,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    Flatten,
    Conv1D,
    Conv2D,
    MaxPool1D,
    MaxPool2D,
    GRU,
    BiGRUModeList,
)
from mlgb.tf.functions import (
    IdentityLayer,
    ActivationLayer,
    InitializerLayer,
    MaskLayer,
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


class TaskLayer(tf.keras.layers.Layer):
    def __init__(self, task='multiclass:10', task_multiclass_if_project=False, task_multiclass_if_softmax=False,
                 task_multiclass_temperature_ratio=None,
                 task_linear_if_identity=False, task_linear_if_weighted=False, task_linear_if_bias=True,
                 task_linear_initializer=None, seed=None):
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
        self.activation_fn = ActivationLayer(self.task_activation_dict[self.task_name])
        self.task_linear_initializer = InitializerLayer(
            initializer=task_linear_initializer,
            activation=self.task_activation_dict[self.task_name],
            seed=seed,
        ).get()
        self.linear_fn = Dense(
            units=1 if self.task_name != 'multiclass' else int(task.split(':')[1]),
            activation=None,
            use_bias=task_linear_if_bias,
            kernel_initializer=self.task_linear_initializer,
        )

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 2:
            raise MLGBError
        if self.task_name == 'multiclass' and not self.task_multiclass_if_project:
            if int(self.task.split(':')[1]) != input_shape[1]:
                raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        if self.task_name != 'multiclass':
            if not self.task_linear_if_identity:
                if self.task_linear_if_weighted:
                    x = self.linear_fn(x)
                else:
                    x = tf.reduce_sum(x, axis=1, keepdims=True)  # only apply after product of matching model.
            x = self.activation_fn(x)
        else:
            if self.task_multiclass_if_project:
                x = self.linear_fn(x)

            if self.task_multiclass_temperature_ratio:
                x /= self.task_multiclass_temperature_ratio  # zoom data for better performance in softmax_loss.

            if self.task_multiclass_if_softmax:
                x = tf.nn.softmax(x, axis=1)
        return x


class LinearV1Layer(tf.keras.layers.Layer):
    def __init__(self, linear_l1=0.0, linear_l2=0.0, linear_initializer=None, linear_activation=None,
                 linear_if_bias=True, linear_unit=1, seed=None):
        super().__init__()
        self.linear_if_bias = linear_if_bias
        self.linear_l1 = linear_l1
        self.linear_l2 = linear_l2
        self.linear_unit = linear_unit  # linear_weight__length_of_latent_vector

        self.linear_initializer = InitializerLayer(
            initializer=linear_initializer,
            activation=linear_activation,
            seed=seed,
        ).get()
        self.activation_fn = ActivationLayer(linear_activation)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 2:
            raise MLGBError

        self.linear_weight = self.add_weight(
            name='linear_weight',
            shape=[input_shape[-1], self.linear_unit],
            initializer=self.linear_initializer,
            regularizer=L1L2(self.linear_l1, self.linear_l2),
            trainable=True,
        )
        self.linear_bias = self.add_weight(
            name='linear_bias',
            shape=[self.linear_unit],
            initializer='zeros',
            regularizer=L1L2(self.linear_l1, self.linear_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x = tf.matmul(x, self.linear_weight) + (self.linear_bias if self.linear_if_bias else 0.0)
        x = self.activation_fn(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'linear_weight': self.linear_weight.numpy(),
                'linear_bias': self.linear_bias.numpy(),
            }
        )
        return config


class LinearV2Layer(tf.keras.layers.Layer):
    # tf: default linear_initializer=None='glorot_uniform'; torch: default linear_initializer=None='he_uniform'.
    def __init__(self, linear_l1=0.0, linear_l2=0.0, linear_initializer=None, linear_activation=None,
                 linear_if_bias=True, linear_unit=1, seed=None):
        super().__init__()
        self.linear_unit = linear_unit
        self.linear_if_bias = linear_if_bias
        self.linear_activation = linear_activation
        self.linear_l1 = linear_l1
        self.linear_l2 = linear_l2
        self.linear_initializer = InitializerLayer(
            initializer=linear_initializer,
            activation=linear_activation,
            seed=seed,
        ).get()

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 2:
            raise MLGBError

        f = input_shape[1]
        self.linear_fn = Dense(
            units=self.linear_unit if self.linear_unit else f,
            activation=self.linear_activation,
            use_bias=self.linear_if_bias,
            kernel_initializer=self.linear_initializer,
            bias_initializer='zeros',
            kernel_regularizer=L1L2(self.linear_l1, self.linear_l2),
            bias_regularizer=L1L2(self.linear_l1, self.linear_l2),
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.linear_fn(x)
        return x


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, linear_activation=None, linear_if_bias=True, linear_l1=0.0, linear_l2=0.0,
                 linear_initializer=None, linear_unit=1, seed=None):
        super().__init__()
        self.linear_fn = LinearV2Layer(
            linear_unit=linear_unit,
            linear_activation=linear_activation,
            linear_if_bias=linear_if_bias,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
            seed=seed,
        )

    def build(self, input_shape):
        # tf.print('input_shape:', input_shape)
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 2:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.linear_fn(x)
        return x


class DeepNeuralNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None, dnn_if_bias=True, seed=None):
        super().__init__()
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_length = len(self.dnn_hidden_units)
        self.dnn_dropouts = [dnn_dropout] * self.dnn_length if isinstance(dnn_dropout, float) else dnn_dropout
        self.dnn_bns = [dnn_if_bn] * self.dnn_length if isinstance(dnn_if_bn, bool) else dnn_if_bn
        self.dnn_lns = [dnn_if_ln] * self.dnn_length if isinstance(dnn_if_ln, bool) else dnn_if_ln
        self.dnn_bias = [dnn_if_bias] * self.dnn_length if isinstance(dnn_if_bias, bool) else dnn_if_bias
        self.dnn_initializer_list = [
            InitializerLayer(
                initializer=dnn_initializer,
                activation=dnn_activation,
                seed=seed + i if isinstance(seed, int) else seed,
            ).get() for i in range(self.dnn_length)
        ]
        self.dnn_fn_list = [
            [
                Dense(
                    units=self.dnn_hidden_units[i],
                    kernel_initializer=self.dnn_initializer_list[i],
                    kernel_regularizer=L1L2(l1=0.0, l2=dnn_l2),
                    use_bias=self.dnn_bias[i],
                ),
                BatchNormalization(axis=1) if self.dnn_bns[i] else IdentityLayer(),
                LayerNormalization(axis=1) if self.dnn_lns[i] else IdentityLayer(),
                Dropout(rate=self.dnn_dropouts[i], seed=seed + i if isinstance(seed, int) else seed),
                ActivationLayer(activation=dnn_activation),
            ]
            for i in range(self.dnn_length)
        ]

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 2:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        for dnn_fn in self.dnn_fn_list:
            for fn in dnn_fn:
                x = fn(x)
        return x


class FeedForwardNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, ffn_linear_if_twice=True, ffn_if_bias=True, ffn_activation='relu',
                 ffn_dropout=0.0, ffn_if_bn=False, ffn_if_ln=False, ffn_l2=0.0, ffn_initializer=None,
                 ffn_last_factor=None, ffn_last_activation=None,
                 seed=None):
        super().__init__()
        self.ffn_linear_if_twice = ffn_linear_if_twice
        self.ffn_if_bias = ffn_if_bias
        self.ffn_l2 = ffn_l2
        self.ffn_last_factor = ffn_last_factor
        self.ffn_last_activation = ffn_last_activation

        self.bn_fn = BatchNormalization(axis=1) if ffn_if_bn else IdentityLayer()
        self.ln_fn = LayerNormalization(axis=1) if ffn_if_ln else IdentityLayer()
        self.drop_fn = Dropout(rate=ffn_dropout, seed=seed)
        self.ffn_initializer_list = [
            InitializerLayer(
                initializer=ffn_initializer,
                activation=None,
                seed=seed + i if isinstance(seed, int) else seed,
            ).get()
            for i in range(2 if self.ffn_linear_if_twice else 1)
        ]
        self.activation_fn = ActivationLayer(activation=ffn_activation)
        if self.ffn_linear_if_twice and self.ffn_last_activation:
            self.activation_last_fn = ActivationLayer(activation=ffn_last_activation)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        _, f, e = input_shape

        self.ffn_weight_list = [
            self.add_weight(
                name=f'ffn_weight_{i}',
                shape=[f, e],
                initializer=self.ffn_initializer_list[i],
                regularizer=L1L2(0.0, self.ffn_l2),
                trainable=True,
            )
            for i in range(2 if self.ffn_linear_if_twice else 1)
        ]
        if self.ffn_if_bias:
            self.ffn_bias_list = [
                self.add_weight(
                    name=f'ffn_bias_{i}',
                    shape=[f, 1],
                    initializer='zeros',
                    regularizer=L1L2(0.0, self.ffn_l2),
                    trainable=True,
                )
                for i in range(2 if self.ffn_linear_if_twice else 1)
            ]
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

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


class Linear3dParallelLayer(tf.keras.layers.Layer):  # vs Dense or EinsumDense with n-d
    def __init__(self, linear_unit=1, linear_activation=None, linear_dropout=0.0, linear_if_bn=False, linear_if_ln=False,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None, linear_if_bias=True, seed=None):
        super().__init__()
        self.linear_unit = linear_unit
        self.linear_if_bias = linear_if_bias
        self.linear_l1 = linear_l1
        self.linear_l2 = linear_l2
        self.linear_initializer = InitializerLayer(
            initializer=linear_initializer,
            activation=linear_activation,
            seed=seed,
        ).get()

        self.activation_fn = ActivationLayer(linear_activation)
        self.bn_fn = BatchNormalization(axis=1) if linear_if_bn else IdentityLayer()
        self.ln_fn = LayerNormalization(axis=1) if linear_if_ln else IdentityLayer()
        self.dropout_fn = Dropout(rate=linear_dropout)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        _, self.linear_parallel_num, self.inputs_width = input_shape

        self.linear_weight = self.add_weight(
            name='linear_weight',
            shape=[self.linear_parallel_num, self.inputs_width, self.linear_unit],
            initializer=self.linear_initializer,
            regularizer=L1L2(self.linear_l1, self.linear_l2),
            trainable=True,
        )
        self.linear_bias = self.add_weight(
            name='linear_bias',
            shape=[self.linear_parallel_num, self.linear_unit],
            initializer='zeros',
            regularizer=L1L2(self.linear_l1, self.linear_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        x = tf.einsum('ijk,jkl->ijl', x, self.linear_weight) + (self.linear_bias if self.linear_if_bias else 0.0)
        x = self.bn_fn(x)
        x = self.ln_fn(x)
        x = self.dropout_fn(x)
        x = self.activation_fn(x)  # (batch_size, linear_parallel_num, linear_unit)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'linear_weight': self.linear_weight.numpy(),
                'linear_bias': self.linear_bias.numpy(),
            }
        )
        return config


class DNN3dParallelLayer(tf.keras.layers.Layer):
    def __init__(self, dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_initializer=None, dnn_l2=0.0, dnn_if_bias=True,
                 dnn_if_output2d=False, seed=None):
        super().__init__()
        self.dnn_if_output2d = dnn_if_output2d
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_length = len(self.dnn_hidden_units)
        self.dnn_dropouts = [dnn_dropout] * self.dnn_length if isinstance(dnn_dropout, float) else dnn_dropout
        self.dnn_bns = [dnn_if_bn] * self.dnn_length if isinstance(dnn_if_bn, bool) else dnn_if_bn
        self.dnn_lns = [dnn_if_ln] * self.dnn_length if isinstance(dnn_if_ln, bool) else dnn_if_ln
        self.dnn_bias = [dnn_if_bias] * self.dnn_length if isinstance(dnn_if_bias, bool) else dnn_if_bias

        self.dnn_fn_list = [
            Linear3dParallelLayer(
                linear_activation=dnn_activation,
                linear_if_bias=self.dnn_bias[i],
                linear_unit=self.dnn_hidden_units[i],
                linear_dropout=self.dnn_dropouts[i],
                linear_if_bn=self.dnn_bns[i],
                linear_if_ln=self.dnn_lns[i],
                linear_l2=dnn_l2,
                linear_initializer=dnn_initializer,
                seed=seed,
            ) for i in range(self.dnn_length)
        ]
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        for dnn_fn in self.dnn_fn_list:
            x = dnn_fn(x)

        if self.dnn_if_output2d:
            x = self.flatten_fn(x)
        return x


class Linear2dParallelLayer(tf.keras.layers.Layer):
    def __init__(self, linear_parallel_num=1, linear_activation=None, linear_if_bias=True,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None, seed=None):
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
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
            seed=seed,
        )
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 2:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x = tf.stack([x] * self.linear_parallel_num, axis=1)  # (batch_size, linear_parallel_num, inputs_width)
        x = self.linear3d_parallel_fn(x)  # (batch_size, linear_parallel_num, linear_unit)
        x = self.flatten_fn(x)
        return x


class ConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, cnn_conv_mode='Conv1D', cnn_filter_num=64, cnn_kernel_height=32, cnn_kernel_width=1,
                 cnn_activation='tanh', cnn_l2=0.0, cnn_initializer=None,
                 cnn_if_max_pool=True, cnn_pool_size=2, seed=None):
        super().__init__()
        if cnn_conv_mode not in ('Conv1D', 'Conv2D'):
            raise MLGBError

        self.cnn_conv_mode = cnn_conv_mode
        self.cnn_if_max_pool = cnn_if_max_pool

        self.cnn_initializer = InitializerLayer(
            initializer=cnn_initializer,
            activation=cnn_activation,
            seed=seed,
        ).get()
        if self.cnn_conv_mode == 'Conv1D':
            self.conv_fn = Conv1D(
                filters=cnn_filter_num,
                kernel_size=cnn_kernel_height,
                strides=1,
                use_bias=True,
                padding='same',
                data_format='channels_last',
                activation=cnn_activation,
                kernel_initializer=self.cnn_initializer,
                kernel_regularizer=L1L2(0.0, cnn_l2),
            )  # (b, f, e) -> (b, f, cnn_filter_num)
            if self.cnn_if_max_pool:
                self.max_pool_fn = MaxPool1D(
                    pool_size=cnn_pool_size,
                    strides=cnn_pool_size,
                    padding='valid',
                    data_format='channels_last',
                )  # (b, f, cnn_filter_num) -> (b, f // cnn_pool_size, cnn_filter_num)
        else:
            self.conv_fn = Conv2D(
                filters=cnn_filter_num,
                kernel_size=(cnn_kernel_height, cnn_kernel_width),  # (k_h, k_w)
                strides=(1, 1),  # (s_h, s_w)
                use_bias=True,
                padding='same',
                data_format='channels_last',
                activation=cnn_activation,
                kernel_initializer=self.cnn_initializer,
                kernel_regularizer=L1L2(0.0, cnn_l2),
            )  # (b, f, e, 1) -> (b, f, e, cnn_filter_num)
            if self.cnn_if_max_pool:
                self.max_pool_fn = MaxPool2D(
                    pool_size=(cnn_pool_size, 1),  # (k_h, k_w)
                    strides=(cnn_pool_size, 1),  # (s_h, s_w)
                    padding='valid',
                    data_format='channels_last',
                )  # (b, f, e, cnn_filter_num) -> (b, f // cnn_pool_size, e, cnn_filter_num)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank not in (3, 4):
            raise MLGBError
        if self.cnn_conv_mode == 'Conv1D' and input_shape.rank != 3:
            raise MLGBError
        if self.cnn_conv_mode == 'Conv2D' and input_shape.rank != 4:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        x = self.conv_fn(x)
        x = self.max_pool_fn(x) if self.cnn_if_max_pool else x
        return x


class ConvolutionalNeuralNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, cnn_conv_mode='Conv1D', cnn_filter_nums=(64, 32), cnn_kernel_heights=(64, 32), cnn_kernel_widths=1,
                 cnn_activation='tanh', cnn_l2=0.0, cnn_initializer=None,
                 cnn_if_max_pool=True, cnn_pool_sizes=2, seed=None):
        super().__init__()
        if cnn_conv_mode not in ('Conv1D', 'Conv2D'):
            raise MLGBError

        self.cnn_conv_mode = cnn_conv_mode
        self.cnn_length = len(cnn_filter_nums)
        self.cnn_kernel_heights = [cnn_kernel_heights] * self.cnn_length if isinstance(cnn_kernel_heights, int) else cnn_kernel_heights
        self.cnn_kernel_widths = [cnn_kernel_widths] * self.cnn_length if isinstance(cnn_kernel_widths, int) else cnn_kernel_widths
        self.cnn_pool_sizes = [cnn_pool_sizes] * self.cnn_length if isinstance(cnn_pool_sizes, int) else cnn_pool_sizes

        self.cnn_fn_list = [
            ConvolutionLayer(
                cnn_conv_mode=cnn_conv_mode,
                cnn_filter_num=cnn_filter_nums[i],
                cnn_kernel_height=self.cnn_kernel_heights[i],
                cnn_kernel_width=self.cnn_kernel_widths[i],
                cnn_activation=cnn_activation,
                cnn_l2=cnn_l2,
                cnn_initializer=cnn_initializer,
                cnn_if_max_pool=cnn_if_max_pool,
                cnn_pool_size=self.cnn_pool_sizes[i],
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(self.cnn_length)
        ]

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank not in (3, 4):
            raise MLGBError
        if self.cnn_conv_mode == 'Conv1D' and input_shape.rank != 3:
            raise MLGBError
        if self.cnn_conv_mode == 'Conv2D' and input_shape.rank != 4:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        for cnn_fn in self.cnn_fn_list:
            x = cnn_fn(x)
        return x


class GatedRecurrentUnitLayer(tf.keras.layers.Layer):
    def __init__(self, gru_hidden_units=(64, 32),
                 gru_activation='tanh', gru_dropout=0.0, gru_l2=0.0, gru_initializer=None,
                 gru_rct_activation='sigmoid', gru_rct_dropout=0.0, gru_rct_l2=0.0, gru_rct_initializer='orthogonal',
                 gru_if_bias=True, gru_reset_after=True, gru_unroll=False, seed=None):
        super().__init__()
        self.gru_length = len(gru_hidden_units)
        self.gru_initializer_list = [
            InitializerLayer(
                initializer=gru_initializer,
                activation=gru_activation,
                seed=seed + i if isinstance(seed, int) else seed,
            ).get()
            for i in range(self.gru_length)
        ]
        self.gru_rct_initializer_list = [
            InitializerLayer(
                initializer=gru_rct_initializer,
                activation=gru_rct_activation,
                seed=seed + i if isinstance(seed, int) else seed,
            ).get()
            for i in range(self.gru_length)
        ]
        self.gru_fn_list = [
            GRU(
                units=gru_hidden_units[i],
                activation=gru_activation,
                dropout=gru_dropout,
                kernel_initializer=self.gru_initializer_list[i],
                kernel_regularizer=L1L2(0.0, gru_l2),
                recurrent_activation=gru_rct_activation,
                recurrent_dropout=gru_rct_dropout,
                recurrent_initializer=self.gru_rct_initializer_list[i],
                recurrent_regularizer=L1L2(0.0, gru_rct_l2),
                use_bias=gru_if_bias,
                reset_after=gru_reset_after,  # True if cuda or GRU_v1 else False(GRU_v3)
                unroll=gru_unroll,  # False if cuda or small memory else True
                return_sequences=True if i < self.gru_length - 1 else False,  # x[:, -1, :]
            )
            for i in range(self.gru_length)
        ]

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        for gru_fn in self.gru_fn_list:
            x = gru_fn(x)
        return x


class BiGatedRecurrentUnitLayer(tf.keras.layers.Layer):
    def __init__(self, gru_bi_mode='Frontward', gru_hidden_units=(64, 32),
                 gru_activation='tanh', gru_dropout=0.0, gru_l2=0.0, gru_initializer=None,
                 gru_rct_activation='sigmoid', gru_rct_dropout=0.0, gru_rct_l2=0.0, gru_rct_initializer='orthogonal',
                 gru_if_bias=True, gru_reset_after=True, gru_unroll=False, seed=None):
        super().__init__()
        if gru_bi_mode not in BiGRUModeList:
            raise MLGBError

        self.gru_bi_mode = gru_bi_mode

        self.gru_fn_list = [
            GatedRecurrentUnitLayer(
                gru_hidden_units=gru_hidden_units,
                gru_activation=gru_activation,
                gru_dropout=gru_dropout,
                gru_l2=gru_l2,
                gru_initializer=gru_initializer,
                gru_rct_activation=gru_rct_activation,
                gru_rct_dropout=gru_rct_dropout,
                gru_rct_l2=gru_rct_l2,
                gru_rct_initializer=gru_rct_initializer,
                gru_if_bias=gru_if_bias,
                gru_reset_after=gru_reset_after,
                gru_unroll=gru_unroll,
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(1 if self.gru_bi_mode in ('Frontward', 'Backward') else 2)
        ]

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        if self.gru_bi_mode == 'Frontward':
            x = self.gru_fn_list[0](x)
        elif self.gru_bi_mode == 'Backward':
            x = self.gru_fn_list[0](x[:, ::-1, :])
        elif self.gru_bi_mode == 'Frontward+Backward':
            x = self.gru_fn_list[0](x) + self.gru_fn_list[1](x[:, ::-1, :])
        elif self.gru_bi_mode == 'Frontward-Backward':
            x = self.gru_fn_list[0](x) - self.gru_fn_list[1](x[:, ::-1, :])
        elif self.gru_bi_mode == 'Frontward*Backward':
            x = self.gru_fn_list[0](x) * self.gru_fn_list[1](x[:, ::-1, :])
        elif self.gru_bi_mode == 'Frontward,Backward':
            x = tf.stack([self.gru_fn_list[0](x), self.gru_fn_list[1](x[:, ::-1, :])], axis=1)
        else:
            raise MLGBError

        x = x if self.gru_bi_mode == 'Frontward,Backward' else tf.expand_dims(x, axis=1)
        return x


class CapsuleNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, capsule_num=3, capsule_activation='squash', capsule_l2=0.0, capsule_initializer=None,
                 capsule_interest_num_if_dynamic=False, capsule_input_sequence_pad_mode='pre',
                 capsule_routing_initializer='random_normal', seed=None):
        super().__init__()
        if capsule_input_sequence_pad_mode not in ('pre', 'post'):
            raise MLGBError

        self.capsule_num = capsule_num
        self.capsule_l2 = capsule_l2
        self.capsule_interest_num_if_dynamic = capsule_interest_num_if_dynamic
        self.capsule_input_sequence_pad_mode = capsule_input_sequence_pad_mode
        self.capsule_initializer = InitializerLayer(
            initializer=capsule_initializer,
            activation=capsule_activation,
            seed=seed,
        ).get()
        self.capsule_routing_initializer = InitializerLayer(
            initializer=capsule_routing_initializer,
            activation=None,
            seed=seed,
        ).get()
        self.activation_fn = ActivationLayer(activation=capsule_activation)
        self.mask_fn = MaskLayer(att_if_mask=True)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        _, seq_len, embed_dim = input_shape
        if self.capsule_interest_num_if_dynamic:
            seq_len = self.get_dynamic_interest_num(seq_len, embed_dim)
            self.seq_len = seq_len

        self.capsule_bilinear_weight = self.add_weight(
            name='capsule_bilinear_weight',
            shape=[seq_len, embed_dim, embed_dim],
            initializer=self.capsule_initializer,
            regularizer=L1L2(0.0, self.capsule_l2),
            trainable=True,
        )
        self.capsule_routing_weight = self.add_weight(
            name='capsule_routing_weight',
            shape=[1, seq_len, embed_dim],
            initializer=self.capsule_routing_initializer,
            regularizer=L1L2(0.0, self.capsule_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        if self.capsule_interest_num_if_dynamic:
            x = x[:, -self.seq_len:, :] if self.capsule_input_sequence_pad_mode == 'pre' else x[:, :self.seq_len, :]

        w = self.capsule_routing_weight
        w_b = self.capsule_bilinear_weight
        for i in range(self.capsule_num):
            w = self.mask_fn(w)
            w = tf.nn.softmax(w, axis=1)
            x_h = tf.einsum('bfe,fee->bfe', x, w_b) * w  # high_level_capsule: x_h = w * w_b * x
            x_h = self.activation_fn(x_h)  # squash

            w_i = tf.einsum('bfe,fee->bfe', x_h, w_b) * x  # routing_logit: w_i = x_h * w_b * x
            w_i = tf.reduce_sum(w_i, axis=0, keepdims=True)
            w = w + w_i  # if `w` isn't be updated(not bp), it's like `w` of RNN.
            x = x_h
        return x

    def get_dynamic_interest_num(self, seq_len, embed_dim):
        seq_k = max(1, min(seq_len, int(numpy.log2(embed_dim))))
        return seq_k







