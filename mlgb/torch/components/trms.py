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

from mlgb.torch.configs import torch
from mlgb.torch.functions import (
    LayerNormalizationLayer,
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


class SimpleAttentionLayer(torch.nn.Module):
    def __init__(self, softmax_axis=1, softmax_if_mask=True, softmax_pre_temperature_ratio=1.0, device='cpu'):
        super().__init__()
        if not (1e-2 <= softmax_pre_temperature_ratio <= 1.0):
            raise MLGBError

        self.softmax_axis = softmax_axis
        self.softmax_pre_temperature_ratio = softmax_pre_temperature_ratio
        self.mask_fn = MaskLayer(att_if_mask=softmax_if_mask, device=device)
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 2:
                raise MLGBError
            if x[0].ndim not in (2, 3):
                raise MLGBError
            if x[0].shape != x[1].shape:
                raise MLGBError

            self.scale = x[0].shape[-1] ** 0.5  #
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_q, x_k = x
        x_v = x_k

        w = x_q * x_k / self.scale
        w = w / self.softmax_pre_temperature_ratio  # debug power_p of MIND model.
        w = self.mask_fn(w)
        w = torch.softmax(w, dim=self.softmax_axis)
        x_v = x_v * w
        return x_v


class LabelAttentionLayer(SimpleAttentionLayer):
    def __init__(self, softmax_axis=1, softmax_pre_temperature_ratio=1.0, device='cpu'):
        super().__init__(softmax_axis, True, softmax_pre_temperature_ratio, device)


class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, mha_head_num=8, mha_head_dim=32, mha_if_mask=True, mha_initializer=None, device='cpu'):
        super().__init__()
        if not (mha_head_num > 0 and mha_head_dim > 0):
            raise MLGBError

        self.mha_head_num = mha_head_num
        self.mha_head_dim = mha_head_dim
        self.mha_model_dim = mha_head_num * mha_head_dim
        self.initializer_fn = InitializerLayer(
            initializer=mha_initializer,
            activation=None,
        ).get()
        self.mask_fn = MaskLayer(att_if_mask=mha_if_mask, device=device)
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

            _, self.fields_width, embed_dim = x[1].shape

            self.qkv_input_weight_list = [
                self.initializer_fn(torch.nn.parameter.Parameter(
                    data=torch.empty(
                        size=[embed_dim, self.mha_model_dim],
                        device=self.device,
                    ),
                    requires_grad=True,
                ))
                for _ in range(3)  # 'qkv'
            ]
            self.qkv_output_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.mha_model_dim, embed_dim],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        # NLP inputs only has 1 vocabulary, 1 sequential feature, sequence_max_length == fields_width.
        # NLP's fields_width << CTR's fields_width, NLP's embed_dim >> CTR's embed_dim,
        # so there are different projection.

        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_q, x_k = x
        x_v = x_k  # (b, f, e)

        x_q @= self.qkv_input_weight_list[0]  # (batch_size, fields_width, model_dim)
        x_k @= self.qkv_input_weight_list[1]
        x_v @= self.qkv_input_weight_list[2]

        qkv_4d_shape = [-1, self.fields_width, self.mha_head_num, self.mha_head_dim]
        x_q = torch.transpose(torch.reshape(x_q, shape=qkv_4d_shape), *[1, 2])  # (b, head_num, f, head_dim)
        x_k = torch.transpose(torch.reshape(x_k, shape=qkv_4d_shape), *[1, 2])
        x_v = torch.transpose(torch.reshape(x_v, shape=qkv_4d_shape), *[1, 2])

        x_att = torch.einsum('bhid,bhjd->bhij', x_q, x_k) / (self.mha_head_dim ** 0.5)  # (b, head_num, f, f)
        x_att = self.mask_fn(x_att)
        x_att = torch.softmax(x_att, dim=-1)  #
        x_v = torch.einsum('bhfi,bhfj->bhij', x_att, x_v)  # (b, head_num, f, head_dim)

        qkv_3d_shape = [-1, self.fields_width, self.mha_model_dim]
        x_v = torch.reshape(torch.transpose(x_v, *[1, 2]), shape=qkv_3d_shape)  # (b, f, model_dim)
        x_v @= self.qkv_output_weight  # (b, f, e)
        return x_v


class TransformerFeedForwardNetworkLayer(torch.nn.Module):
    def __init__(self, ffn_activation='gelu', ffn_initializer=None, device='cpu'):
        super().__init__()
        self.activation_fn = ActivationLayer(
            activation=ffn_activation,
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
                for _ in range(2)
            ]
            self.ffn_bias_list = [
                self.initializer_zeros_fn(torch.nn.parameter.Parameter(
                    data=torch.empty(
                        size=[f, 1],
                        device=self.device,
                    ),
                    requires_grad=True,
                ))
                for _ in range(2)
            ]
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = x * self.ffn_weight_list[0] + self.ffn_bias_list[0]
        x = self.activation_fn(x)
        x = x * self.ffn_weight_list[1] + self.ffn_bias_list[1]
        return x


class TransformerResidualLayer(torch.nn.Module):
    def __init__(self, residual_ln_axis=1, residual_dropout=0.0, device='cpu'):
        super().__init__()
        self.ln_fn = LayerNormalizationLayer(axis=residual_ln_axis, device=device)
        self.drop_fn = torch.nn.Dropout(p=residual_dropout)
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
        x, x_stack = x

        x = torch.add(x, x_stack)
        x = self.ln_fn(x)
        x = self.drop_fn(x)
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(self, trm_if_pe=False, trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_initializer=None, trm_residual_dropout=0.0,
                 device='cpu'):
        super().__init__()
        self.trm_if_ffn = trm_if_ffn

        self.pe_fn = PositionalEncoding(att_if_pe=trm_if_pe)
        self.mha_fn = MultiHeadAttentionLayer(
            mha_head_num=trm_mha_head_num,
            mha_head_dim=trm_mha_head_dim,
            mha_if_mask=trm_mha_if_mask,
            mha_initializer=trm_mha_initializer,
            device=device,
        )
        if self.trm_if_ffn:
            self.ffn_fn = TransformerFeedForwardNetworkLayer(
                ffn_activation=trm_ffn_activation,
                ffn_initializer=trm_ffn_initializer,
                device=device,
            )
        self.residual_fn_list = torch.nn.ModuleList([
            TransformerResidualLayer(
                residual_ln_axis=1,
                residual_dropout=trm_residual_dropout,
                device=device,
            ) for _ in range(2 if self.trm_if_ffn else 1)
        ])
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
                 trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_residual_dropout=0.0, device='cpu'):
        super().__init__(
            trm_if_pe=False,
            trm_mha_head_num=trm_mha_head_num, trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask, trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=False,
            trm_residual_dropout=trm_residual_dropout, device=device,
        )






