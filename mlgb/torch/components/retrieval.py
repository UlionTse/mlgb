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
    SampleModeList,
)
from mlgb.torch.functions import (
    FlattenLayer,
)
from mlgb.torch.inputs import InputsLayer
from mlgb.torch.components.linears import DeepNeuralNetworkLayer
from mlgb.error import MLGBError


__all__ = [
    'SampledSoftmaxLossLayer',
    'BaseInputsEmbeddingLayer',
]


class SampledSoftmaxLossLayer(torch.nn.Module):
    def __init__(self, sample_mode='Sample:batch', sample_num=None, sample_item_distribution_list=None,
                 sample_fixed_unigram_frequency_list=None, sample_fixed_unigram_distortion=1.0,
                 device='cpu'):
        """
        Paper Link:
            https://arxiv.org/pdf/1310.4546.pdf,
            https://arxiv.org/pdf/1412.2007.pdf,
            https://www.tensorflow.org/extras/candidate_sampling.pdf
        """
        super().__init__()
        if sample_mode not in SampleModeList:
            raise MLGBError

        self.sample_mode = sample_mode
        self.sample_num = sample_num
        self.sample_item_distribution_list = sample_item_distribution_list
        self.sample_fixed_unigram_frequency_list = sample_fixed_unigram_frequency_list
        self.sample_fixed_unigram_distortion = sample_fixed_unigram_distortion
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, y_true, y_pred):
        if not self.built:
            if not (y_true.ndim == y_pred.ndim == 2):
                raise MLGBError
            if y_true.shape != y_pred.shape:
                raise MLGBError
            if y_true.shape[1] <= 2:
                raise MLGBError

            self.built = True
        return

    def forward(self, y_true, y_pred):
        y_true = torch.as_tensor(y_true, dtype=torch.float32, device=self.device)
        y_pred = torch.as_tensor(y_pred, dtype=torch.float32, device=self.device)
        self.build(y_true, y_pred)

        if self.sample_mode != 'Sample:batch':
            y_true_int = y_true.to(dtype=torch.int64)
            item_indices = torch.where(torch.sum(y_true_int, dim=0, keepdim=False) > 0)[0]  #

            if self.sample_item_distribution_list:
                q = torch.as_tensor(self.sample_item_distribution_list, dtype=torch.float32, device=self.device)
                q = torch.unsqueeze(q, dim=0)
            else:
                q = self.get_item_distribution(y_true)
            q = torch.index_select(q, dim=1, index=item_indices)

            y_true = torch.index_select(y_true, dim=1, index=item_indices)
            y_pred = torch.index_select(y_pred, dim=1, index=item_indices)
            y_pred -= torch.log(q)

        loss = torch.nn.functional.cross_entropy(input=y_pred, target=y_true, reduction='mean')
        return loss

    def get_item_distribution(self, y_true, add_bias=1):  # Batch Q like BN, not global Q.
        q_top = torch.sum(y_true, dim=0, keepdim=True)
        q_bottom = torch.sum(y_true, dim=None, keepdim=False)
        q = (q_top + add_bias) / (q_bottom + add_bias)  # add 1 avoid log(0) = -inf.
        return q


class BaseInputsEmbeddingLayer(torch.nn.Module):
    def __init__(self, user_feature_names,
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=False, user_inputs_if_embed_dense=True,
                 embed_dim=32, embed_initializer=None,
                 pool_mv_mode='Attention', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Attention', pool_seq_axis=2, pool_seq_initializer=None,
                 user_dnn_hidden_units=(64, 32), dnn_activation=None, dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False,
                 tower_embeds_flatten_mode='flatten', tower_embeds_if_l2_norm=True,
                 device='cpu'):
        super().__init__()
        if tower_embeds_flatten_mode not in ('flatten', 'sum'):
            raise MLGBError

        self.user_dnn_hidden_units = user_dnn_hidden_units
        self.tower_embeds_flatten_mode = tower_embeds_flatten_mode
        self.tower_embeds_if_l2_norm = tower_embeds_if_l2_norm

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,
            inputs_if_embed_dense=user_inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_2d_dim=None,
            embed_cate_if_output2d=False,  #
            embed_initializer=embed_initializer,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        if self.user_dnn_hidden_units:
            self.dnn_fn = DeepNeuralNetworkLayer(
                dnn_hidden_units=user_dnn_hidden_units,
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                device=device,
            )
        self.flatten_fn = FlattenLayer(device=device)
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):  # OriginInputs
        if not self.built:
            if len(x) not in (2, 3, 4):
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        _, x = self.user_input_fn(x)

        if self.tower_embeds_flatten_mode == 'flatten':
            x = self.flatten_fn(x)
        else:
            x = torch.sum(x, dim=1, keepdim=False)

        if self.user_dnn_hidden_units:
            x = self.dnn_fn(x)

        if self.tower_embeds_if_l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x



























