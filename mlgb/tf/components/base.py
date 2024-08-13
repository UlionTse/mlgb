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
    L1L2,
    Flatten,
    BatchNormalization,
    LayerNormalization,
    PoolModeList,
    EDCNModeList,
)
from mlgb.tf.functions import (
    FlattenAxesLayer,
    ActivationLayer,
    InitializerLayer,
    SimplePoolingLayer,
)
from mlgb.tf.components.linears import (
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


class ResidualUnitLayer(tf.keras.layers.Layer):
    def __init__(self, dcm_if_dnn=True, dcm_if_ln=False,
                 dnn_hidden_unit=32, dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.dcm_if_dnn = dcm_if_dnn
        self.dcm_if_ln = dcm_if_ln
        self.dnn_activation = dnn_activation
        self.dnn_dropout = dnn_dropout
        self.dnn_if_bn = dnn_if_bn
        self.dnn_if_ln = dnn_if_ln
        self.dnn_l2 = dnn_l2
        self.dnn_initializer = dnn_initializer
        self.seed = seed

        if self.dcm_if_dnn:
            self.dnn_fn = DeepNeuralNetworkLayer(
                dnn_hidden_units=[dnn_hidden_unit],
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed,
            )
        if self.dcm_if_ln:
            self.ln_fn = LayerNormalization(axis=1)
        self.activation_fn = ActivationLayer(activation=self.dnn_activation)

    def build(self, input_shape):
        if input_shape.rank != 2:
            raise MLGBError

        self.stack_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=[input_shape[-1]],
            dnn_activation=None,
            dnn_dropout=self.dnn_dropout,
            dnn_if_bn=self.dnn_if_bn,
            dnn_if_ln=self.dnn_if_ln,
            dnn_l2=self.dnn_l2,
            dnn_initializer=self.dnn_initializer,
            seed=self.seed,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x_dnn = self.dnn_fn(x) if self.dcm_if_dnn else x
        x_stack = self.stack_fn(x_dnn)

        x = tf.add(x, x_stack)  # x_stack = x_true - x_pred
        x = self.ln_fn(x) if self.dcm_if_ln else x
        x = self.activation_fn(x)
        return x


class CrossNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, dcn_version='v2', dcn_layer_num=1, dcn_l2=0.0, dcn_initializer=None, seed=None):
        super().__init__()
        if dcn_version not in ('v1', 'v2'):
            raise MLGBError

        self.dcn_version = dcn_version
        self.dcn_layer_num = dcn_layer_num
        self.dcn_l2 = dcn_l2
        self.dcn_unit = 1  # dcn_weight__length_of_latent_vector
        self.dcn_initializer_list = [
            InitializerLayer(
                initializer=dcn_initializer,
                activation=None,
                seed=seed + i if isinstance(seed, int) else seed,
            ).get() for i in range(self.dcn_layer_num)
        ]

    def build(self, input_shape):
        if input_shape.rank != 2:
            raise MLGBError

        self.embed_dim = input_shape[-1]

        self.dcn_weight_shap_map = {
            'v1': [self.embed_dim, self.dcn_unit],
            'v2': [self.embed_dim, self.embed_dim],
        }
        self.dcn_weight_list = [
            self.add_weight(
                name=f'dcn_weight_{i}',
                shape=self.dcn_weight_shap_map[self.dcn_version],
                initializer=self.dcn_initializer_list[i],
                regularizer=L1L2(0.0, self.dcn_l2),
                trainable=True,
            ) for i in range(self.dcn_layer_num)
        ]
        self.dcn_bias_list = [
            self.add_weight(
                name=f'dcn_bias_{i}',
                shape=[self.dcn_unit],
                initializer='zeros',
                regularizer=L1L2(0.0, self.dcn_l2),
                trainable=True,
            ) for i in range(self.dcn_layer_num)
        ]
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_i = x_0 = inputs
        for i in range(self.dcn_layer_num):
            if self.dcn_version == 'v2':
                x_i = x_0 * (x_i @ self.dcn_weight_list[i] + self.dcn_bias_list[i]) + x_i
            else:
                x_c = tf.einsum('ijk,ik->ij', tf.expand_dims(x_0, axis=-1), x_i @ self.dcn_weight_list[i])
                x_i = x_c + self.dcn_bias_list[i] + x_i
        return x_i


class RegulationModuleLayer(tf.keras.layers.Layer):
    def __init__(self, fgu_tau_ratio=1.0, fgu_initializer='ones', fgu_l2=0.0, seed=None):
        super().__init__()
        if not (0 < fgu_tau_ratio <= 1.0):
            raise MLGBError  # field-wise gating unit

        self.fgu_tau_ratio = fgu_tau_ratio
        self.fgu_l2 = fgu_l2
        self.fgu_initializer = InitializerLayer(
            initializer=fgu_initializer,
            activation=None,
            seed=seed,
        ).get()
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        _, self.fields_width, self.embed_dim = input_shape

        self.fgu_weight = self.add_weight(
            name='fgu_weight',
            shape=[1, self.fields_width, 1],
            initializer=self.fgu_initializer,
            regularizer=L1L2(0.0, self.fgu_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        fgu_w = self.fgu_weight * self.fgu_tau_ratio
        fgu_score = tf.nn.softmax(fgu_w, axis=1)
        fgu_outputs = x * fgu_score
        fgu_outputs = self.flatten_fn(fgu_outputs)
        return fgu_outputs


class BridgeModuleLayer(tf.keras.layers.Layer):
    def __init__(self, bdg_mode='EDCN:attention_pooling', bdg_layer_num=1,
                 dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        if bdg_mode not in EDCNModeList:
            raise MLGBError

        self.bdg_mode = bdg_mode
        self.bdg_layer_num = bdg_layer_num
        self.dnn_activation = dnn_activation
        self.dnn_dropout = dnn_dropout
        self.dnn_if_bn = dnn_if_bn
        self.dnn_if_ln = dnn_if_ln
        self.dnn_l2 = dnn_l2
        self.dnn_initializer = dnn_initializer
        self.seed = seed

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[0] != input_shape[1]:
            raise MLGBError

        _, self.inputs_width = input_shape[0]
        bdg_dnn_hidden_units = [self.inputs_width] * self.bdg_layer_num

        if self.bdg_mode in ('EDCN:attention_pooling', 'EDCN:concatenation'):
            self.dnn_fn = DeepNeuralNetworkLayer(
                dnn_hidden_units=bdg_dnn_hidden_units,
                dnn_activation=self.dnn_activation,
                dnn_dropout=self.dnn_dropout,
                dnn_if_bn=self.dnn_if_bn,
                dnn_if_ln=self.dnn_if_ln,
                dnn_l2=self.dnn_l2,
                dnn_initializer=self.dnn_initializer,
                seed=self.seed,
            )
        if self.bdg_mode == 'EDCN:attention_pooling':
            self.dnn2_fn = DeepNeuralNetworkLayer(
                dnn_hidden_units=bdg_dnn_hidden_units,
                dnn_activation=self.dnn_activation,
                dnn_dropout=self.dnn_dropout,
                dnn_if_bn=self.dnn_if_bn,
                dnn_if_ln=self.dnn_if_ln,
                dnn_l2=self.dnn_l2,
                dnn_initializer=self.dnn_initializer,
                seed=self.seed + 1 if isinstance(self.seed, int) else self.seed,
            )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_c, x_d = inputs  # x_cross, x_deep

        if self.bdg_mode == 'EDCN:pointwise_addition':
            x = x_c + x_d
        elif self.bdg_mode == 'EDCN:hadamard_product':
            x = x_c * x_d
        elif self.bdg_mode == 'EDCN:concatenation':
            x = tf.concat([x_c, x_d], axis=-1)
            x = self.dnn_fn(x)
        elif self.bdg_mode == 'EDCN:attention_pooling':
            att_c = tf.nn.log_softmax(self.dnn_fn(x_c), axis=1)  # dead softmax.
            att_d = tf.nn.log_softmax(self.dnn2_fn(x_d), axis=1)
            x = x_c * att_c + x_d * att_d
        else:
            raise MLGBError
        return x


class CompressedInteractionNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, cin_interaction_num=4, cin_interaction_ratio=1.0,
                 cnn_filter_num=64, cnn_kernel_size=64, cnn_activation='relu',
                 cnn_l2=0.0, cnn_initializer=None,
                 seed=None):
        super().__init__()
        if not (0.5 <= cin_interaction_ratio <= 1.0):
            raise MLGBError('0.5 <= cin_interaction_ratio <= 1.0')

        self.cin_interaction_num = cin_interaction_num
        self.cin_interaction_ratio = cin_interaction_ratio
        self.cnn_fn_list = [
            ConvolutionalNeuralNetworkLayer(
                cnn_conv_mode='Conv1D',
                cnn_filter_nums=[cnn_filter_num],
                cnn_kernel_heights=[cnn_kernel_size],
                cnn_activation=cnn_activation,
                cnn_l2=cnn_l2,
                cnn_initializer=cnn_initializer,
                cnn_if_max_pool=False,
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(cin_interaction_num)
        ]
        self.flatten_axes_fn_list = [FlattenAxesLayer(axes=[2, 3]) for _ in range(cin_interaction_num)]

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        x_0 = x_i = x
        x_list = []  # feature_map_list
        for i in range(self.cin_interaction_num):
            x_i = tf.einsum('bie,bje->beij', x_0, x_i)  # (b, e, f_0, f_i)
            x_i = self.flatten_axes_fn_list[i](x_i)  # (b, e, f_0 * f_i)
            x_i = self.cnn_fn_list[i](x_i)  # (b, e, cnn_filter_num)
            x_i = tf.transpose(x_i, perm=[0, 2, 1])

            x_list.append(x_i)
            x_i = self.get_center_x(x_i, self.cin_interaction_ratio)

        x = tf.concat(x_list, axis=1)
        x = tf.reduce_sum(x, axis=2, keepdims=False)  # (b, cin_interaction_num * cnn_filter_num)
        return x

    def get_center_x(self, x, center_ratio):
        f = x.shape[1]
        boundary_i = int(f * (1 - center_ratio) // 2)
        boundary_j = int(f - boundary_i)
        return x[:, boundary_i:boundary_j, :]


class SqueezeExcitationNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, sen_pool_mode='Pooling:average', sen_reduction_factor=2, sen_activation='relu',
                 sen_l2=0.0, sen_initializer=None, seed=None):
        super().__init__()
        if sen_pool_mode not in PoolModeList:
            raise MLGBError
        if not (isinstance(sen_reduction_factor, int) and sen_reduction_factor >= 1):
            raise MLGBError('sen_reduction_factor')

        self.sen_pool_mode = sen_pool_mode
        self.sen_reduction_factor = sen_reduction_factor
        self.sen_l2 = sen_l2
        self.sen_initializer = InitializerLayer(
            initializer=sen_initializer,
            activation=sen_activation,
            seed=seed,
        ).get()

        self.activation_fn = ActivationLayer(sen_activation)
        self.pooling_fn = SimplePoolingLayer(
            pool_mode=sen_pool_mode,
            pool_axis=2,
        )

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        _, self.fields_width, self.embed_dim = input_shape
        self.fields_unit = max(1, self.fields_width // self.sen_reduction_factor)

        self.sen_w1 = self.add_weight(
            name='sen_w1',
            shape=[self.fields_width, self.fields_unit],
            initializer=self.sen_initializer,
            regularizer=L1L2(0.0, self.sen_l2),
            trainable=True,
        )
        self.sen_w2 = self.add_weight(
            name='sen_w2',
            shape=[self.fields_unit, self.fields_width],
            initializer=self.sen_initializer,
            regularizer=L1L2(0.0, self.sen_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs  # (batch_size, fields_width, embed_dim)
        x_z = self.pooling_fn(x)  # (batch_size, fields_width)
        x_z = self.activation_fn(x_z @ self.sen_w1)
        x_z = self.activation_fn(x_z @ self.sen_w2)  # (batch_size, fields_width)
        x_z = tf.expand_dims(x_z, axis=2)
        x = x * x_z  # re_weight == attention, (batch_size, fields_width, embed_dim), dim=3.
        return x


class LocalActivationUnitLayer(tf.keras.layers.Layer):
    def __init__(self, lau_version='v4', lau_hidden_units=(16,), lau_activation='dice', seed=None):
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
            seed=seed,
        )
        self.flatten_axes_fn = FlattenAxesLayer(axes=[2, 3])

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == 3):
            raise MLGBError
        if input_shape[0] != input_shape[1]:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_q, x_k = inputs

        if self.lau_version in ('v3', 'v4'):
            x_op = tf.einsum('bfi,bfj->bfij', x_q, x_k)
            x_op = self.flatten_axes_fn(x_op)  # (b, f, e*e)
            x = tf.concat([x_op, x_q, x_k], axis=2)  # (b, f, (e+2)*e)
        else:
            x_ip = x_q * x_k
            x_sub = x_q - x_k
            x = tf.concat([x_ip, x_sub, x_q, x_k], axis=2)  # (b, f, 4*e)

        x = self.dnn_parallel_fn(x)  # (b, f, 1)
        return x


class FactorEstimatingNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, ifm_mode_if_dual=False,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.ifm_mode_if_dual = ifm_mode_if_dual
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_length = len(self.dnn_hidden_units)
        self.dnn_activation = dnn_activation
        self.dnn_dropout = dnn_dropout
        self.dnn_if_bn = dnn_if_bn
        self.dnn_if_ln = dnn_if_ln
        self.dnn_l2 = dnn_l2
        self.dnn_initializer = dnn_initializer
        self.seed = seed
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        self.fields_width = input_shape[1]

        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=self.dnn_hidden_units + (self.fields_width,),
            dnn_if_bias=[True] * self.dnn_length + [False],
            dnn_activation=self.dnn_activation,
            dnn_dropout=self.dnn_dropout,
            dnn_if_bn=self.dnn_if_bn,
            dnn_if_ln=self.dnn_if_ln,
            dnn_l2=self.dnn_l2,
            dnn_initializer=self.dnn_initializer,
            seed=self.seed,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.flatten_fn(x)  # (b, f*e)
        x = self.dnn_fn(x)

        if not self.ifm_mode_if_dual:
            x = tf.nn.softmax(x, axis=1) * self.fields_width  # (b, f)

        x = tf.expand_dims(x, axis=2)  # (b, f, 1)
        return x


class LogarithmicTransformationLayer(tf.keras.layers.Layer):
    def __init__(self, ltl_clip_min=1e-4, ltl_unit=32, ltl_l2=0.0, ltl_initializer=None, seed=None):
        super().__init__()
        self.ltl_clip_min = ltl_clip_min
        self.ltl_unit = ltl_unit
        self.ltl_l2 = ltl_l2
        self.ltl_initializer = InitializerLayer(
            initializer=ltl_initializer,
            activation=None,
            seed=seed,
        ).get()
        self.bn_fn_list = [BatchNormalization(axis=1) for _ in range(2)]
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        self.fields_width = input_shape[1]

        self.ltl_weight = self.add_weight(
            name='ltl_weight',
            shape=[self.fields_width, self.ltl_unit],
            initializer=self.ltl_initializer,
            regularizer=L1L2(0.0, self.ltl_l2),
            trainable=True,
        )
        self.ltl_bias = self.add_weight(
            name='ltl_bias',
            shape=[self.ltl_unit],
            initializer='zeros',
            regularizer=L1L2(0.0, self.ltl_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        x = tf.abs(x)
        x = tf.clip_by_value(x, clip_value_min=self.ltl_clip_min, clip_value_max=numpy.inf)
        x = self.bn_fn_list[0](x)  # (b, f, e)
        x = tf.einsum('bfe,fu->beu', x, self.ltl_weight) + self.ltl_bias
        x = tf.exp(x)
        x = self.bn_fn_list[1](x)
        x = self.flatten_fn(x)  # (b, u*e)
        return x


class MaskBlockLayer(tf.keras.layers.Layer):
    def __init__(self, ffn_activation='relu', ffn_if_bn=False, ffn_dropout=0.0, ffn_l2=0.0, ffn_initializer=None,
                 seed=None):
        super().__init__()
        self.igm_fn = FeedForwardNetworkLayer(
            ffn_linear_if_twice=True,
            ffn_if_bias=True,
            ffn_activation=ffn_activation,
            ffn_dropout=ffn_dropout,
            ffn_if_bn=ffn_if_bn,
            ffn_if_ln=False,
            ffn_l2=ffn_l2,
            ffn_initializer=ffn_initializer,
            seed=seed,
        )
        self.ln_hid_fn = FeedForwardNetworkLayer(
            ffn_linear_if_twice=False,
            ffn_if_bias=False,
            ffn_activation=ffn_activation,
            ffn_dropout=ffn_dropout,
            ffn_if_bn=ffn_if_bn,
            ffn_if_ln=True,  #
            ffn_l2=ffn_l2,
            ffn_initializer=ffn_initializer,
            seed=seed + 2 if isinstance(seed, int) else seed,
        )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 3:
            raise MLGBError
        if input_shape[0] != input_shape[1]:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x1, x2 = inputs

        x2 = self.igm_fn(x2)
        x = x1 * x2
        x = self.ln_hid_fn(x)
        return x












