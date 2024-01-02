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
    torch,
    SeqRecPointwiseModeList,
)
from mlgb.torch.functions import (
    FlattenLayer,
    FlattenAxesLayer,
    MaskLayer,
    BiasEncoding,
    SimplePoolingLayer,
)
from mlgb.torch.components.linears import (
    DeepNeuralNetworkLayer,
    ConvolutionalNeuralNetworkLayer,
    GatedRecurrentUnitLayer,
    BiGatedRecurrentUnitLayer,
)
from mlgb.torch.components.trms import (
    LabelAttentionLayer,
    TransformerLayer,
)
from mlgb.torch.components.base import (
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


class GatedRecurrentUnit4RecommendationLayer(torch.nn.Module):
    def __init__(self, seq_rec_pointwise_mode='Add',
                 gru_hidden_units=(64, 32), gru_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
        super().__init__()
        if seq_rec_pointwise_mode not in SeqRecPointwiseModeList:
            raise MLGBError

        self.seq_rec_pointwise_mode = seq_rec_pointwise_mode

        self.gru_fn = GatedRecurrentUnitLayer(
            gru_hidden_units=gru_hidden_units,
            gru_dropout=gru_dropout,
            gru_if_bias=True,
            device=device,
        )
        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            self.label_attention_fn = LabelAttentionLayer(
                softmax_axis=1,
                softmax_pre_temperature_ratio=1.0,
                device=device,
            )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        self.flatten_fn_list = torch.nn.ModuleList([
            FlattenLayer(device=device) 
            for _ in range(2 if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention') else 1)
        ])
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 3:
                raise MLGBError
            if not (x[0].ndim == x[1].ndim == x[2].ndim == 3):
                raise MLGBError
            
            self.built = True
        return

    def forward(self, x):
        # PointWise(not ListWise) by Add&LabelAttention.
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_user_fea, x_user_seq, x_item_tgt = x

        x_seq = self.gru_fn(x_user_seq)  # (b, u == e)
        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            x_k = x_seq
            x_q = torch.sum(x_item_tgt, dim=1, keepdim=False)
            x_seq = self.label_attention_fn([x_q, x_k])

        x_fea = self.flatten_fn_list[0](x_user_fea)
        x = torch.concat([x_fea, x_seq], dim=1)
        if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention'):
            x_item = self.flatten_fn_list[1](x_item_tgt)
            x = torch.concat([x, x_item], dim=1)

        x = self.dnn_fn(x)
        return x


class ConvolutionalSequenceEmbeddingRecommendationLayer(torch.nn.Module):
    def __init__(self, seq_rec_pointwise_mode='Add',
                 cnn_filter_num=64, cnn_kernel_size=4, cnn_pool_size=2,
                 cnn_activation='tanh', cnn_l2=0.0, cnn_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
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
        self.device = device

        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            self.label_attention_fn = LabelAttentionLayer(
                softmax_axis=1,
                softmax_pre_temperature_ratio=1.0,
                device=device,
            )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        self.flatten_fn_list = torch.nn.ModuleList([
            FlattenLayer(device=device)
            for _ in range(3 if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention') else 2)
        ])
        self.flatten_axes_fn_list = torch.nn.ModuleList([
            FlattenAxesLayer(axes=(1, 3), device=device)
            for _ in range(2)
        ])
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 3:
                raise MLGBError
            if not (x[0].ndim == x[1].ndim == x[2].ndim == 3):
                raise MLGBError

            self.embed_dim = x.shape[0][2]

            self.h_cnn_fn = ConvolutionalNeuralNetworkLayer(
                cnn_conv_mode='Conv2D',
                cnn_filter_nums=(self.cnn_filter_num,),
                cnn_kernel_heights=(self.cnn_kernel_size,),
                cnn_kernel_widths=(self.embed_dim,),  #
                cnn_activation=self.cnn_activation,
                cnn_if_max_pool=True,
                cnn_pool_sizes=(self.cnn_pool_size,),
                device=self.device,
            )
            self.v_cnn_fn = ConvolutionalNeuralNetworkLayer(
                cnn_conv_mode='Conv2D',
                cnn_filter_nums=(self.cnn_filter_num,),
                cnn_kernel_heights=(self.cnn_kernel_size,),
                cnn_kernel_widths=(1,),
                cnn_activation=self.cnn_activation,
                cnn_if_max_pool=False,  #
                device=self.device,
            )

            self.built = True
        return

    def forward(self, x):
        # PointWise(not ListWise) by Add&LabelAttention.
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_user_fea, x_user_seq, x_item_tgt = x

        x_user_seq = torch.unsqueeze(x_user_seq, dim=2)
        x_h_seq = self.h_cnn_fn(x_user_seq)  # HorizontalConvolution
        x_h_seq = self.flatten_axes_fn_list[0](x_h_seq)
        x_v_seq = self.v_cnn_fn(x_user_seq)  # VerticalConvolution
        x_v_seq = self.flatten_axes_fn_list[1](x_v_seq)
        x_user_seq = torch.concat([x_h_seq, x_v_seq], dim=1)

        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            x_k = torch.sum(x_user_seq, dim=1, keepdim=False)
            x_q = torch.sum(x_item_tgt, dim=1, keepdim=False)
            x_user_seq = self.label_attention_fn([x_q, x_k])

        x_fea = self.flatten_fn_list[0](x_user_fea)
        x_seq = self.flatten_fn_list[1](x_user_seq)
        x = torch.concat([x_fea, x_seq], dim=1)
        if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention'):
            x_item = self.flatten_fn_list[2](x_item_tgt)
            x = torch.concat([x, x_item], dim=1)

        x = self.dnn_fn(x)
        return x


class Transformer4RecommendationLayer(torch.nn.Module):
    def __init__(self, seq_rec_pointwise_mode='Add',
                 trm_num=1, trm_if_pe=True, trm_mha_head_num=1, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_initializer=None,
                 trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
        super().__init__()
        if seq_rec_pointwise_mode not in SeqRecPointwiseModeList + ['OnlyAddTrm']:  #
            raise MLGBError

        self.seq_rec_pointwise_mode = seq_rec_pointwise_mode

        self.trm_fn_list = torch.nn.ModuleList([
            TransformerLayer(
                trm_if_pe=True if (i == 0 and trm_if_pe) else False,
                trm_mha_head_num=trm_mha_head_num,
                trm_mha_head_dim=trm_mha_head_dim,
                trm_mha_if_mask=trm_mha_if_mask,
                trm_mha_initializer=trm_mha_initializer,
                trm_if_ffn=trm_if_ffn,
                trm_ffn_activation=trm_ffn_activation,
                trm_ffn_initializer=trm_ffn_initializer,
                trm_residual_dropout=trm_residual_dropout,
                device=device,
            )
            for i in range(trm_num)
        ])
        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            self.label_attention_fn = LabelAttentionLayer(
                softmax_axis=1,
                softmax_pre_temperature_ratio=1.0,
                device=device,
            )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        self.flatten_fn_list = torch.nn.ModuleList([
            FlattenLayer(device=device)
            for _ in range(3 if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention') else 2)
        ])
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 3:
                raise MLGBError
            if not (x[0].ndim == x[1].ndim == x[2].ndim == 3):
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        # PointWise(not ListWise) by Add&LabelAttention.
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_user_fea, x_user_seq, x_item_tgt = x

        if self.seq_rec_pointwise_mode == 'OnlyAddTrm':
            x_user_seq = torch.concat([x_user_seq, x_item_tgt], dim=1)

        for trm_fn in self.trm_fn_list:
            x_user_seq = trm_fn([x_user_seq, x_user_seq])

        if self.seq_rec_pointwise_mode in ('LabelAttention', 'Add&LabelAttention'):
            x_k = torch.sum(x_user_seq, dim=1, keepdim=False)
            x_q = torch.sum(x_item_tgt, dim=1, keepdim=False)
            x_user_seq = self.label_attention_fn([x_q, x_k])

        x_fea = self.flatten_fn_list[0](x_user_fea)
        x_seq = self.flatten_fn_list[1](x_user_seq)
        x = torch.concat([x_fea, x_seq], dim=1)
        if self.seq_rec_pointwise_mode in ('Add', 'Add&LabelAttention'):
            x_item = self.flatten_fn_list[2](x_item_tgt)
            x = torch.concat([x, x_item], dim=1)

        x = self.dnn_fn(x)
        return x


class SelfAttentiveSequentialRecommendationLayer(Transformer4RecommendationLayer):
    def __init__(self, seq_rec_pointwise_mode='Add',
                 trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_initializer=None, trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
        super().__init__(
            seq_rec_pointwise_mode=seq_rec_pointwise_mode,
            trm_num=1,
            trm_if_pe=False,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )


class BidirectionalEncoderRepresentationTransformer4RecommendationLayer(Transformer4RecommendationLayer):
    def __init__(self, seq_rec_pointwise_mode='Add',
                 trm_num=4, trm_if_pe=True, trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_initializer=None, trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
        super().__init__(
            seq_rec_pointwise_mode=seq_rec_pointwise_mode,
            trm_num=trm_num,
            trm_if_pe=trm_if_pe,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,  # PositionWiseFFN but unnecessary, nonlinearity should rely on activation of FFN.
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )


class BehaviorSequenceTransformerLayer(Transformer4RecommendationLayer):
    def __init__(self, trm_mha_head_num=8, trm_mha_head_dim=32, trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_initializer=None, trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='selu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
        super().__init__(
            seq_rec_pointwise_mode='OnlyAddTrm',
            trm_num=1,
            trm_if_pe=True,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )


class DeepInterestNetworkLayer(torch.nn.Module):
    def __init__(self, lau_version='v4', lau_hidden_units=(16,), lau_if_softmax=False,
                 dnn_hidden_units=(64, 32), dnn_activation='dice', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
        super().__init__()
        self.lau_if_softmax = lau_if_softmax

        self.lau_fn = LocalActivationUnitLayer(
            lau_version=lau_version,
            lau_hidden_units=lau_hidden_units,
            lau_activation=dnn_activation,
            device=device,
        )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        if self.lau_if_softmax:
            self.mask_fn = MaskLayer(att_if_mask=True)
        self.flatten_fn = FlattenLayer(device=device)
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 3:
                raise MLGBError
            if not (x[0].ndim == x[1].ndim == x[2].ndim == 3):
                raise MLGBError
            if x[1].shape != x[2].shape:
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_b, x_q, x_k = x  # (base, query, key==value)
        x_v = x_k
        x_b = self.flatten_fn(x_b)

        w = self.lau_fn([x_q, x_k])  # (b, s, 1)
        if self.lau_if_softmax:
            w = self.mask_fn(w)
            w = torch.softmax(w, dim=1)  # axis -> s

        x_v = x_v * w
        x_v = torch.sum(x_v, dim=2, keepdim=False)
        x = torch.concat([x_v, x_b], dim=1)
        x = self.dnn_fn(x)
        return x


class DeepInterestEvolutionNetworkLayer(GatedRecurrentUnit4RecommendationLayer):
    def __init__(self, gru_hidden_units=(64, 32), gru_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='dice', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
        super().__init__(
            seq_rec_pointwise_mode='Add&LabelAttention',  #
            gru_hidden_units=gru_hidden_units,
            gru_dropout=gru_dropout,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,  #
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )


class DeepSessionInterestNetworkLayer(torch.nn.Module):
    def __init__(self, dsin_if_process_session=False, session_pool_mode='Pooling:average', session_size=4, session_stride=2,
                 bias_initializer='zeros',
                 trm_mha_head_num=4, trm_mha_head_dim=32, trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_initializer=None, trm_residual_dropout=0.0,
                 gru_bi_mode='Frontward+Backward', gru_hidden_units=(64, 32), gru_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 device='cpu'):
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
            bias_initializer=bias_initializer,
            device=device,
        )
        self.trm_fn = TransformerLayer(
            trm_if_pe=False,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
            device=device,
        )
        self.gru_fn = BiGatedRecurrentUnitLayer(
            gru_bi_mode=gru_bi_mode,
            gru_hidden_units=gru_hidden_units,
            gru_dropout=gru_dropout,
            gru_if_bias=True,
            device=device,
        )
        self.label_attention_fn_list = torch.nn.ModuleList([
            LabelAttentionLayer(
                softmax_axis=1,
                softmax_pre_temperature_ratio=1.0,
                device=device,
            )
            for _ in range(2)
        ])
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        self.flatten_fn_list = torch.nn.ModuleList([FlattenLayer(device=device) for _ in range(2)])
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 3:
                raise MLGBError
            if not (x[0].ndim == x[1].ndim == x[2].ndim == 3):
                raise MLGBError
            if self.dsin_if_process_session:
                seq_length = x[1].shape[1]
                if seq_length < self.session_size + self.session_stride * 2:
                    raise MLGBError('seq_length < self.session_size + self.session_stride * 2')

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_user_fea, x_user_seq, x_item_tgt = x

        x_user_seq = self.seq2session2seq(x_user_seq)
        x_user_seq = self.bias_fn(x_user_seq)

        x_trm_seq = self.trm_fn([x_user_seq, x_user_seq])
        x_gru_seq = self.gru_fn(x_trm_seq)

        x_q = torch.sum(x_item_tgt, dim=1, keepdim=False)
        x_trm_k = torch.sum(x_trm_seq, dim=1, keepdim=False)
        x_gru_k = x_gru_seq
        x_trm_seq = self.label_attention_fn_list[0]([x_q, x_trm_k])
        x_gru_seq = self.label_attention_fn_list[1]([x_q, x_gru_k])

        x_fea = self.flatten_fn_list[0](x_user_fea)
        x_item = self.flatten_fn_list[1](x_item_tgt)
        x = torch.concat([x_fea, x_item, x_trm_seq, x_gru_seq], dim=1)
        x = self.dnn_fn(x)
        return x

    def seq2session2seq(self, x):
        if self.dsin_if_process_session:
            x_list = [
                self.pool_fn(x[:, i: i + self.session_size, :])
                for i in range(0, x.shape[1] - (self.session_size + self.session_stride), self.session_stride)
            ]
            x = torch.concat(x_list, dim=1)
        return x






























