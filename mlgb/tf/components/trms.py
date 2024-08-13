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
    tf,
    L1L2,
    Dropout,
    LayerNormalization,
)
from mlgb.tf.functions import (
    ActivationLayer,
    InitializerLayer,
    MaskLayer,
    PositionalEncoding,
)
from mlgb.error import MLGBError


__all__ = [
    'SimpleAttentionLayer',
    'LabelAttentionLayer',
    'TransformerLayer',
    'InteractingLayer',
]


class SimpleAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, softmax_axis=1, softmax_if_mask=True, softmax_pre_temperature_ratio=1.0):
        super().__init__()
        if not (1e-2 <= softmax_pre_temperature_ratio <= 1.0):
            raise MLGBError

        self.softmax_axis = softmax_axis
        self.softmax_pre_temperature_ratio = softmax_pre_temperature_ratio
        self.mask_fn = MaskLayer(att_if_mask=softmax_if_mask)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank not in (2, 3):
            raise MLGBError
        if input_shape[0] != input_shape[1]:
            raise MLGBError

        self.scale = input_shape[0][-1] ** 0.5  #
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_q, x_k = inputs
        x_v = x_k

        w = x_q * x_k / self.scale
        w = w / self.softmax_pre_temperature_ratio  # debug power_p of MIND model.
        w = self.mask_fn(w)
        w = tf.nn.softmax(w, axis=self.softmax_axis)
        x_v = x_v * w
        return x_v


class LabelAttentionLayer(SimpleAttentionLayer):
    def __init__(self, softmax_axis=1, softmax_pre_temperature_ratio=1.0):
        super().__init__(softmax_axis, True, softmax_pre_temperature_ratio)


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, mha_head_num=8, mha_head_dim=32, mha_if_mask=True, mha_l2=0.0, mha_initializer=None, seed=None):
        super().__init__()
        if not (mha_head_num > 0 and mha_head_dim > 0):
            raise MLGBError

        self.mha_head_num = mha_head_num
        self.mha_head_dim = mha_head_dim
        self.mha_model_dim = mha_head_num * mha_head_dim
        self.mha_l2 = mha_l2
        self.mha_initializer_list = [
            InitializerLayer(
                initializer=mha_initializer,
                activation=None,
                seed=seed + i if isinstance(seed, int) else seed,
            ).get() for i, _ in enumerate('qkv')
        ]
        self.mask_fn = MaskLayer(att_if_mask=mha_if_mask)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == 3):
            raise MLGBError
        if input_shape[0] != input_shape[1]:
            raise MLGBError

        _, self.fields_width, embed_dim = input_shape[1]

        self.qkv_input_weight_list = [
            self.add_weight(
                name=f'qkv_input_weight_{w}',
                shape=[embed_dim, self.mha_model_dim],
                initializer=self.mha_initializer_list[i],
                regularizer=L1L2(0.0, self.mha_l2),
                trainable=True,
            ) for i, w in enumerate('qkv')
        ]
        self.qkv_output_weight = self.add_weight(
            name=f'qkv_output_weight',
            shape=[self.mha_model_dim, embed_dim],
            initializer=self.mha_initializer_list[0],
            regularizer=L1L2(0.0, self.mha_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        # NLP inputs only has 1 vocabulary, 1 sequential feature, sequence_max_length == fields_width.
        # NLP's fields_width << CTR's fields_width, NLP's embed_dim >> CTR's embed_dim,
        # so there are different projection.

        x_q, x_k = inputs
        x_v = x_k  # (b, f, e)

        x_q @= self.qkv_input_weight_list[0]  # (batch_size, fields_width, model_dim)
        x_k @= self.qkv_input_weight_list[1]
        x_v @= self.qkv_input_weight_list[2]

        qkv_4d_shape = [-1, self.fields_width, self.mha_head_num, self.mha_head_dim]
        x_q = tf.transpose(tf.reshape(x_q, shape=qkv_4d_shape), perm=[0, 2, 1, 3])  # (b, head_num, f, head_dim)
        x_k = tf.transpose(tf.reshape(x_k, shape=qkv_4d_shape), perm=[0, 2, 1, 3])
        x_v = tf.transpose(tf.reshape(x_v, shape=qkv_4d_shape), perm=[0, 2, 1, 3])

        x_att = tf.einsum('bhid,bhjd->bhij', x_q, x_k) / (self.mha_head_dim ** 0.5)  # (b, head_num, f, f)
        x_att = self.mask_fn(x_att)
        x_att = tf.nn.softmax(x_att, axis=-1)  #
        x_v = tf.einsum('bhfi,bhfj->bhij', x_att, x_v)  # (b, head_num, f, head_dim)

        qkv_3d_shape = [-1, self.fields_width, self.mha_model_dim]
        x_v = tf.reshape(tf.transpose(x_v, perm=[0, 2, 1, 3]), shape=qkv_3d_shape)  # (b, f, model_dim)
        x_v @= self.qkv_output_weight  # (b, f, e)
        return x_v


class TransformerFeedForwardNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, ffn_activation='gelu', ffn_l2=0.0, ffn_initializer=None, seed=None):
        super().__init__()
        self.ffn_l2 = ffn_l2
        self.ffn_initializer_list = [
            InitializerLayer(
                initializer=ffn_initializer,
                activation=None,
                seed=seed + i if isinstance(seed, int) else seed,
            ).get() for i in range(2)
        ]
        self.activation_fn = ActivationLayer(activation=ffn_activation)

    def build(self, input_shape):
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
            ) for i in range(2)
        ]
        self.ffn_bias_list = [
            self.add_weight(
                name=f'ffn_bias_{i}',
                shape=[f, 1],
                initializer='zeros',
                regularizer=L1L2(0.0, self.ffn_l2),
                trainable=True,
            ) for i in range(2)
        ]
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        x = x * self.ffn_weight_list[0] + self.ffn_bias_list[0]
        x = self.activation_fn(x)
        x = x * self.ffn_weight_list[1] + self.ffn_bias_list[1]
        return x


class TransformerResidualLayer(tf.keras.layers.Layer):
    def __init__(self, residual_ln_axis=1, residual_dropout=0.0):
        super().__init__()
        self.ln_fn = LayerNormalization(axis=residual_ln_axis)
        self.dropout_fn = Dropout(rate=residual_dropout)

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
        x, x_stack = inputs

        x = tf.add(x, x_stack)
        x = self.ln_fn(x)
        x = self.dropout_fn(x)
        return x


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, trm_if_pe=False, trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_l2=0.0, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_l2=0.0, trm_ffn_initializer=None,
                 trm_residual_dropout=0.0, seed=None):
        super().__init__()
        self.trm_if_ffn = trm_if_ffn

        self.pe_fn = PositionalEncoding(att_if_pe=trm_if_pe)
        self.mha_fn = MultiHeadAttentionLayer(
            mha_head_num=trm_mha_head_num,
            mha_head_dim=trm_mha_head_dim,
            mha_if_mask=trm_mha_if_mask,
            mha_l2=trm_mha_l2,
            mha_initializer=trm_mha_initializer,
            seed=seed,
        )
        if self.trm_if_ffn:
            self.ffn_fn = TransformerFeedForwardNetworkLayer(
                ffn_activation=trm_ffn_activation,
                ffn_l2=trm_ffn_l2,
                ffn_initializer=trm_ffn_initializer,
                seed=seed,
            )
        self.residual_fn_list = [
            TransformerResidualLayer(
                residual_ln_axis=1,
                residual_dropout=trm_residual_dropout,
            ) for _ in range(2 if self.trm_if_ffn else 1)
        ]

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

        x_q = self.pe_fn(x_q)
        x_k = self.pe_fn(x_k)
        x = self.mha_fn([x_q, x_k])
        x_k = self.residual_fn_list[0]([x, x_k])

        if self.trm_if_ffn:
            x = self.ffn_fn(x_k)
            x_k = self.residual_fn_list[1]([x, x_k])
        return x_k


class InteractingLayer(TransformerLayer):
    def __init__(self, trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_l2=0.0, trm_mha_initializer=None,
                 trm_residual_dropout=0.0, seed=None):
        super().__init__(
            trm_if_pe=False,
            trm_mha_head_num=trm_mha_head_num, trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask, trm_mha_l2=trm_mha_l2, trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=False,
            trm_residual_dropout=trm_residual_dropout, seed=seed,
        )
















