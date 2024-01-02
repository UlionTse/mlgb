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
    Flatten,
    SeqRecPointwiseModeList,
)
from mlgb.tf.functions import (
    FlattenAxesLayer,
    MaskLayer,
    BiasEncoding,
    SimplePoolingLayer,
)
from mlgb.tf.components.linears import (
    DeepNeuralNetworkLayer,
    ConvolutionalNeuralNetworkLayer,
    GatedRecurrentUnitLayer,
    BiGatedRecurrentUnitLayer,
)
from mlgb.tf.components.trms import (
    LabelAttentionLayer,
    TransformerLayer,
)
from mlgb.tf.components.base import (
    LocalActivationUnitLayer,
)
from mlgb.error import MLGBError


__all__ = [
    'GatedRecurrentUnit4RecommendationLayer',
    'ConvolutionalSequenceEmbeddingRecommendationLayer',
    'SelfAttentiveSequentialRecommendationLayer',
    'BidirectionalEncoderRepresentationTransformer4RecommendationLayer',
    'BehaviorSequenceTransformerLayer',
    'DeepInterestNetworkLayer',
    'DeepInterestEvolutionNetworkLayer',
    'DeepSessionInterestNetworkLayer',
]


class GatedRecurrentUnit4RecommendationLayer(tf.keras.layers.Layer):
    def __init__(self, seq_rec_pointwise_mode='Add',
                 gru_hidden_units=(64, 32), gru_activation='tanh', gru_dropout=0.0, gru_l2=0.0, gru_initializer=None,
                 gru_rct_activation='sigmoid', gru_rct_dropout=0.0, gru_rct_l2=0.0, gru_rct_initializer='orthogonal',
                 gru_reset_after=True, gru_unroll=False,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        if seq_rec_pointwise_mode not in SeqRecPointwiseModeList:
            raise MLGBError
        
        self.seq_rec_pointwise_mode = seq_rec_pointwise_mode

        self.gru_fn = GatedRecurrentUnitLayer(
            gru_hidden_units=gru_hidden_units,
            gru_activation=gru_activation,
            gru_dropout=gru_dropout,
            gru_l2=gru_l2,
            gru_initializer=gru_initializer,
            gru_rct_activation=gru_rct_activation,
            gru_rct_dropout=gru_rct_dropout,
            gru_rct_l2=gru_rct_l2,
            gru_rct_initializer=gru_rct_initializer,
            gru_if_bias=True,
            gru_reset_after=gru_reset_after,
            gru_unroll=gru_unroll,
            seed=seed,
        )
        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            self.label_attention_fn = LabelAttentionLayer(
                softmax_axis=1,
                softmax_pre_temperature_ratio=1.0,
            )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
        self.flatten_fn_list = [Flatten() for _ in range(2 if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention') else 1)]

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == input_shape[2].rank == 3):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        # PointWise(not ListWise) by Add&LabelAttention.
        x_user_fea, x_user_seq, x_item_tgt = inputs

        x_seq = self.gru_fn(x_user_seq)  # (b, u == e)
        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            x_k = x_seq
            x_q = tf.reduce_sum(x_item_tgt, axis=1, keepdims=False)
            x_seq = self.label_attention_fn([x_q, x_k])

        x_fea = self.flatten_fn_list[0](x_user_fea)
        x = tf.concat([x_fea, x_seq], axis=1)
        if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention'):
            x_item = self.flatten_fn_list[1](x_item_tgt)
            x = tf.concat([x, x_item], axis=1)

        x = self.dnn_fn(x)
        return x


class ConvolutionalSequenceEmbeddingRecommendationLayer(tf.keras.layers.Layer):
    def __init__(self, seq_rec_pointwise_mode='Add',
                 cnn_filter_num=64, cnn_kernel_size=4, cnn_pool_size=2,
                 cnn_activation='tanh', cnn_l2=0.0, cnn_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        if seq_rec_pointwise_mode not in SeqRecPointwiseModeList:
            raise MLGBError

        self.seq_rec_pointwise_mode = seq_rec_pointwise_mode
        self.cnn_filter_num = cnn_filter_num
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_activation = cnn_activation
        self.cnn_l2 = cnn_l2
        self.cnn_initializer = cnn_initializer
        self.cnn_pool_size = cnn_pool_size
        self.seed = seed

        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            self.label_attention_fn = LabelAttentionLayer(
                softmax_axis=1,
                softmax_pre_temperature_ratio=1.0,
            )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
        self.flatten_fn_list = [Flatten() for _ in range(3 if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention') else 2)]
        self.flatten_axes_fn_list = [FlattenAxesLayer(axes=(1, 3)) for _ in range(2)]

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == input_shape[2].rank == 3):
            raise MLGBError

        self.embed_dim = input_shape[0][2]

        self.h_cnn_fn = ConvolutionalNeuralNetworkLayer(
            cnn_conv_mode='Conv2D',
            cnn_filter_nums=(self.cnn_filter_num,),
            cnn_kernel_heights=(self.cnn_kernel_size,),
            cnn_kernel_widths=(self.embed_dim,),  #
            cnn_activation=self.cnn_activation,
            cnn_l2=self.cnn_l2,
            cnn_initializer=self.cnn_initializer,
            cnn_if_max_pool=True,
            cnn_pool_sizes=(self.cnn_pool_size,),
            seed=self.seed,
        )
        self.v_cnn_fn = ConvolutionalNeuralNetworkLayer(
            cnn_conv_mode='Conv2D',
            cnn_filter_nums=(self.cnn_filter_num,),
            cnn_kernel_heights=(self.cnn_kernel_size,),
            cnn_kernel_widths=(1,),
            cnn_activation=self.cnn_activation,
            cnn_l2=self.cnn_l2,
            cnn_initializer=self.cnn_initializer,
            cnn_if_max_pool=False,  #
            seed=self.seed,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        # PointWise(not ListWise) by Add&LabelAttention.
        x_user_fea, x_user_seq, x_item_tgt = inputs

        x_user_seq = tf.expand_dims(x_user_seq, axis=2)
        x_h_seq = self.h_cnn_fn(x_user_seq)  # HorizontalConvolution
        x_h_seq = self.flatten_axes_fn_list[0](x_h_seq)
        x_v_seq = self.v_cnn_fn(x_user_seq)  # VerticalConvolution
        x_v_seq = self.flatten_axes_fn_list[1](x_v_seq)
        x_user_seq = tf.concat([x_h_seq, x_v_seq], axis=1)
        
        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            x_k = tf.reduce_sum(x_user_seq, axis=1, keepdims=False)
            x_q = tf.reduce_sum(x_item_tgt, axis=1, keepdims=False)
            x_user_seq = self.label_attention_fn([x_q, x_k])

        x_fea = self.flatten_fn_list[0](x_user_fea)
        x_seq = self.flatten_fn_list[1](x_user_seq)
        x = tf.concat([x_fea, x_seq], axis=1)
        if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention'):
            x_item = self.flatten_fn_list[2](x_item_tgt)
            x = tf.concat([x, x_item], axis=1)

        x = self.dnn_fn(x)
        return x


class Transformer4RecommendationLayer(tf.keras.layers.Layer):
    def __init__(self, seq_rec_pointwise_mode='Add',
                 trm_num=1, trm_if_pe=True, trm_mha_head_num=1, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_l2=0.0, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_l2=0.0, trm_ffn_initializer=None,
                 trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        if seq_rec_pointwise_mode not in SeqRecPointwiseModeList + ['OnlyAddTrm']:
            raise MLGBError

        self.seq_rec_pointwise_mode = seq_rec_pointwise_mode

        self.trm_fn_list = [
            TransformerLayer(
                trm_if_pe=True if (i == 0 and trm_if_pe) else False,
                trm_mha_head_num=trm_mha_head_num,
                trm_mha_head_dim=trm_mha_head_dim,
                trm_mha_if_mask=trm_mha_if_mask,
                trm_mha_l2=trm_mha_l2,
                trm_mha_initializer=trm_mha_initializer,
                trm_if_ffn=trm_if_ffn,
                trm_ffn_activation=trm_ffn_activation,
                trm_ffn_l2=trm_ffn_l2,
                trm_ffn_initializer=trm_ffn_initializer,
                trm_residual_dropout=trm_residual_dropout,
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(trm_num)
        ]
        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            self.label_attention_fn = LabelAttentionLayer(
                softmax_axis=1,
                softmax_pre_temperature_ratio=1.0,
            )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
        self.flatten_fn_list = [Flatten() for _ in range(3 if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention') else 2)]

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == input_shape[2].rank == 3):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        # PointWise(not ListWise) by Add&LabelAttention.
        x_user_fea, x_user_seq, x_item_tgt = inputs

        if self.seq_rec_pointwise_mode == 'OnlyAddTrm':
            x_user_seq = tf.concat([x_user_seq, x_item_tgt], axis=1)
        
        for trm_fn in self.trm_fn_list:
            x_user_seq = trm_fn([x_user_seq, x_user_seq])

        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            x_k = tf.reduce_sum(x_user_seq, axis=1, keepdims=False)
            x_q = tf.reduce_sum(x_item_tgt, axis=1, keepdims=False)
            x_user_seq = self.label_attention_fn([x_q, x_k])

        x_fea = self.flatten_fn_list[0](x_user_fea)
        x_seq = self.flatten_fn_list[1](x_user_seq)
        x = tf.concat([x_fea, x_seq], axis=1)
        if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention'):
            x_item = self.flatten_fn_list[2](x_item_tgt)
            x = tf.concat([x, x_item], axis=1)

        x = self.dnn_fn(x)
        return x
        

class SelfAttentiveSequentialRecommendationLayer(Transformer4RecommendationLayer):
    def __init__(self, seq_rec_pointwise_mode='Add',
                 trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_l2=0.0, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_l2=0.0, trm_ffn_initializer=None,
                 trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__(
            seq_rec_pointwise_mode=seq_rec_pointwise_mode,
            trm_num=1,
            trm_if_pe=False,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_l2=trm_mha_l2,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_l2=trm_ffn_l2,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )


class BidirectionalEncoderRepresentationTransformer4RecommendationLayer(Transformer4RecommendationLayer):
    def __init__(self, seq_rec_pointwise_mode='Add',
                 trm_num=4, trm_if_pe=True, trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_l2=0.0, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_l2=0.0, trm_ffn_initializer=None,
                 trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__(
            seq_rec_pointwise_mode=seq_rec_pointwise_mode,
            trm_num=trm_num,
            trm_if_pe=trm_if_pe,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_l2=trm_mha_l2,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,  # PositionWiseFFN but unnecessary, nonlinearity should rely on activation of FFN.
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_l2=trm_ffn_l2,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )


class BehaviorSequenceTransformerLayer(Transformer4RecommendationLayer):
    def __init__(self, trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_l2=0.0, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_l2=0.0, trm_ffn_initializer=None,
                 trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='selu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__(
            seq_rec_pointwise_mode='OnlyAddTrm',
            trm_num=1,
            trm_if_pe=True,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_l2=trm_mha_l2,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_l2=trm_ffn_l2,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )


class DeepInterestNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, lau_version='v4', lau_hidden_units=(16,), lau_if_softmax=False,
                 dnn_hidden_units=(64, 32), dnn_activation='dice', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.lau_if_softmax = lau_if_softmax

        self.lau_fn = LocalActivationUnitLayer(
            lau_version=lau_version,
            lau_hidden_units=lau_hidden_units,
            lau_activation=dnn_activation,
            seed=seed,
        )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
        if self.lau_if_softmax:
            self.mask_fn = MaskLayer(att_if_mask=True)
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == input_shape[2].rank == 3):
            raise MLGBError
        if input_shape[1] != input_shape[2]:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_b, x_q, x_k = inputs  # (base, query, key==value)
        x_v = x_k
        x_b = self.flatten_fn(x_b)

        w = self.lau_fn([x_q, x_k])  # (b, s, 1)
        if self.lau_if_softmax:
            w = self.mask_fn(w)
            w = tf.nn.softmax(w, axis=1)  # axis -> s

        x_v = x_v * w
        x_v = tf.reduce_sum(x_v, axis=2, keepdims=False)
        x = tf.concat([x_v, x_b], axis=1)
        x = self.dnn_fn(x)
        return x


class DeepInterestEvolutionNetworkLayer(GatedRecurrentUnit4RecommendationLayer):
    def __init__(self, gru_hidden_units=(64, 32), gru_activation='tanh', gru_dropout=0.0, gru_l2=0.0, gru_initializer=None,
                 gru_rct_activation='sigmoid', gru_rct_dropout=0.0, gru_rct_l2=0.0, gru_rct_initializer='orthogonal',
                 gru_reset_after=True, gru_unroll=False,
                 dnn_hidden_units=(64, 32), dnn_activation='dice', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__(
            seq_rec_pointwise_mode='Add&LabelAttention',  #
            gru_hidden_units=gru_hidden_units,
            gru_activation=gru_activation,
            gru_dropout=gru_dropout,
            gru_l2=gru_l2,
            gru_initializer=gru_initializer,
            gru_rct_activation=gru_rct_activation,
            gru_rct_dropout=gru_rct_dropout,
            gru_rct_l2=gru_rct_l2,
            gru_rct_initializer=gru_rct_initializer,
            gru_reset_after=gru_reset_after,
            gru_unroll=gru_unroll,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,  #
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )


class DeepSessionInterestNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, dsin_if_process_session=False, session_pool_mode='Pooling:average', session_size=4, session_stride=2,
                 bias_l2=0.0, bias_initializer='zeros',
                 trm_mha_head_num=4, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_l2=0.0, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_l2=0.0, trm_ffn_initializer=None,
                 trm_residual_dropout=0.0,
                 gru_bi_mode='Frontward+Backward',
                 gru_hidden_units=(64, 32), gru_activation='tanh', gru_dropout=0.0, gru_l2=0.0, gru_initializer=None,
                 gru_rct_activation='sigmoid', gru_rct_dropout=0.0, gru_rct_l2=0.0, gru_rct_initializer='orthogonal',
                 gru_reset_after=True, gru_unroll=False,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.dsin_if_process_session = dsin_if_process_session
        self.session_size = session_size
        self.session_stride = session_stride

        if self.dsin_if_process_session:
            self.pool_fn = SimplePoolingLayer(
                pool_mode=session_pool_mode,
                pool_axis=1,
                pool_axis_if_keep=True,  #
            )
        self.bias_fn = BiasEncoding(
            if_bias=True,
            bias_l2=bias_l2,
            bias_initializer=bias_initializer,
            seed=seed,
        )
        self.trm_fn = TransformerLayer(
            trm_if_pe=False,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_l2=trm_mha_l2,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_l2=trm_ffn_l2,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
            seed=seed,
        )
        self.gru_fn = BiGatedRecurrentUnitLayer(
            gru_bi_mode=gru_bi_mode,
            gru_hidden_units=gru_hidden_units,
            gru_activation=gru_activation,
            gru_dropout=gru_dropout,
            gru_l2=gru_l2,
            gru_initializer=gru_initializer,
            gru_rct_activation=gru_rct_activation,
            gru_rct_dropout=gru_rct_dropout,
            gru_rct_l2=gru_rct_l2,
            gru_rct_initializer=gru_rct_initializer,
            gru_if_bias=True,
            gru_reset_after=gru_reset_after,
            gru_unroll=gru_unroll,
            seed=seed,
        )
        self.label_attention_fn_list = [
            LabelAttentionLayer(
                softmax_axis=1,
                softmax_pre_temperature_ratio=1.0,
            )
            for _ in range(2)
        ]
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
        self.flatten_fn_list = [Flatten() for _ in range(2)]

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == input_shape[2].rank == 3):
            raise MLGBError

        if self.dsin_if_process_session:
            seq_length = input_shape[1][1]
            if seq_length < self.session_size + self.session_stride * 2:
                raise MLGBError('seq_length < self.session_size + self.session_stride * 2')

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_user_fea, x_user_seq, x_item_tgt = inputs

        x_user_seq = self.seq2session2seq(x_user_seq)
        x_user_seq = self.bias_fn(x_user_seq)

        x_trm_seq = self.trm_fn([x_user_seq, x_user_seq])
        x_gru_seq = self.gru_fn(x_trm_seq)

        x_q = tf.reduce_sum(x_item_tgt, axis=1, keepdims=False)
        x_trm_k = tf.reduce_sum(x_trm_seq, axis=1, keepdims=False)
        x_gru_k = x_gru_seq
        x_trm_seq = self.label_attention_fn_list[0]([x_q, x_trm_k])
        x_gru_seq = self.label_attention_fn_list[1]([x_q, x_gru_k])

        x_fea = self.flatten_fn_list[0](x_user_fea)
        x_item = self.flatten_fn_list[1](x_item_tgt)
        x = tf.concat([x_fea, x_item, x_trm_seq, x_gru_seq], axis=1)
        x = self.dnn_fn(x)
        return x

    def seq2session2seq(self, x):
        if self.dsin_if_process_session:
            x_list = [
                self.pool_fn(x[:, i: i+self.session_size, :])
                for i in range(0, x.shape[1] - (self.session_size + self.session_stride), self.session_stride)
            ]
            x = tf.concat(x_list, axis=1)
        return x






























